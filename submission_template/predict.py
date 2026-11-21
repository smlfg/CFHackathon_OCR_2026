#!/usr/bin/env python3
"""Official TableSalt submission entrypoint (multi-variant locked at build time).

Contract (README §7):
    python submission_template/predict.py <input_dir> <output_dir>

Reads TABLESALT_VARIANT (baked into the Docker image via ARG VARIANT) and loads
the immutable config at submission_template/variants/<variant>.json.
Writes one valid <table_id>.json atomically as soon as each table finishes.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytesseract
import torch
from PIL import Image, ImageFilter
from transformers import TableTransformerForObjectDetection

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import my_tatr_fusion as mf  # noqa: E402
from baselines import baseline_tesseract as bt  # noqa: E402

VARIANTS_DIR = Path(__file__).resolve().parent / "variants"
MODEL_DIR = Path(os.environ.get("TATR_MODEL_DIR", "/app/models/table-transformer-structure-recognition"))
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}


def load_variant_config() -> tuple[dict, str]:
    variant_id = os.environ.get("TABLESALT_VARIANT", "v1_tatr_clean").strip()
    cfg_path = VARIANTS_DIR / f"{variant_id}.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Unknown TABLESALT_VARIANT={variant_id!r} (missing {cfg_path})")
    raw = cfg_path.read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()[:16]


def resolve_device(cfg_device: str) -> str:
    override = os.environ.get("TATR_DEVICE", "").strip().lower()
    want = override or cfg_device.lower()
    if want == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if want == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return want


def load_image_safe(path: Path) -> Image.Image:
    """Load raster image or first page of a PDF safely as RGB."""
    if path.suffix.lower() == ".pdf":
        with tempfile.TemporaryDirectory() as td:
            prefix = Path(td) / "page"
            subprocess.run(
                ["pdftoppm", "-png", "-r", "200", "-f", "1", "-l", "1", str(path), str(prefix)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            pages = sorted(Path(td).glob("page*.png"))
            if not pages:
                raise RuntimeError(f"pdftoppm produced no page for {path}")
            return Image.open(pages[0]).convert("RGB")
    return Image.open(path).convert("RGB")


def ocr_cells_naive(img: Image.Image, cells: list[dict], lang: str = "eng", workers: int = 6) -> list[dict]:
    """V0 baseline OCR: reproduces scripts/tatr_experiment.py ocr_cells."""
    def _one(c: dict) -> dict:
        x0, y0, x1, y1 = [int(round(v)) for v in c["bbox"]]
        pad = 2
        crop = img.crop((
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(img.width, x1 + pad),
            min(img.height, y1 + pad),
        ))
        if crop.width < 40 and crop.width > 0 and crop.height > 0:
            crop = crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS)
        try:
            text = pytesseract.image_to_string(crop, lang=lang, config="--psm 7").strip()
        except Exception:
            text = ""
        return {
            "text": " ".join(text.split()),
            "r": c["r"],
            "c": c["c"],
            "rs": c["rs"],
            "cs": c["cs"],
            "header": c["header"],
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, cells))


def fallback_tesseract_cells(img: Image.Image) -> list[dict]:
    """Defensive fallback when TATR returns 0 cells: simple Tesseract grid."""
    words = bt.words_from_image(img.convert("RGB"))
    if not words:
        return [{"text": "", "r": 0, "c": 0, "rs": 1, "cs": 1}]
    rows = bt.cluster_rows(words)
    spans, cols = bt.column_edges(rows)
    out = []
    for r, row in enumerate(spans):
        cells = [""] * len(cols)
        for s in row:
            ci = min(range(len(cols)), key=lambda i: abs(cols[i] - s["x"]))
            cells[ci] = (cells[ci] + " " + s["text"]).strip()
        for c, txt in enumerate(cells):
            out.append({"text": txt, "r": r, "c": c, "rs": 1, "cs": 1})
    return out or [{"text": "", "r": 0, "c": 0, "rs": 1, "cs": 1}]


def run_single_table(
    img_path: Path,
    model: TableTransformerForObjectDetection,
    device: str,
    cfg: dict,
) -> tuple[list[dict], dict]:
    t0 = time.perf_counter()
    img = load_image_safe(img_path)

    # 1) Structure stage (TATR)
    t_s0 = time.perf_counter()
    tatr_img = img.filter(ImageFilter.MedianFilter(3)) if cfg.get("tatr_denoise") == "median3" else img
    objs = mf.detect_structure_objects(
        tatr_img,
        model,
        device,
        score_threshold=float(cfg.get("score_threshold", 0.5)),
        nms_threshold=float(cfg.get("nms_iou", 0.3)),
    )
    cells = mf.structure_to_cells(objs)
    t_struct = time.perf_counter() - t_s0

    if not cells:
        fallback = fallback_tesseract_cells(img)
        return fallback, {
            "total_s": time.perf_counter() - t0,
            "struct_s": t_struct,
            "ocr_s": 0.0,
            "cells": len(fallback),
            "fallback": True,
        }

    # 2) OCR stage
    t_o0 = time.perf_counter()
    ocr_denoise = cfg.get("ocr_denoise", "speckle_gate")
    if ocr_denoise == "always_median3":
        ocr_img = tatr_img
    elif ocr_denoise == "speckle_gate":
        thr = float(cfg.get("speckle_gate_threshold", 0.010))
        ocr_img = tatr_img if mf.has_speckle_noise(img, threshold=thr) else img
    else:
        ocr_img = img

    workers = int(cfg.get("workers", 6))
    lang = str(cfg.get("ocr_lang", "deu+eng"))
    ocr_mode = cfg.get("ocr_mode", "clean_crop")

    if ocr_mode == "naive_crop":
        filled = ocr_cells_naive(ocr_img, cells, lang=lang, workers=workers)
    elif ocr_mode == "clean_crop":
        mf.TESS_LANG = lang
        filled = mf.fill_cells_with_tesseract(ocr_img, cells, workers=workers)
    else:
        raise ValueError(f"Unsupported ocr_mode={ocr_mode!r}")

    t_ocr = time.perf_counter() - t_o0

    # 3) Optional deterministic postprocess
    if cfg.get("postprocess") == "coherence_l1":
        import coherence_postprocess as cp
        filled, _ = cp.apply_coherence(filled)

    # Strip extra keys only if needed; keep clean official schema
    out_cells = [
        {
            "text": c.get("text", ""),
            "r": int(c["r"]),
            "c": int(c["c"]),
            "rs": int(c.get("rs", 1)),
            "cs": int(c.get("cs", 1)),
        }
        for c in filled
    ]
    return out_cells, {
        "total_s": time.perf_counter() - t0,
        "struct_s": t_struct,
        "ocr_s": t_ocr,
        "cells": len(out_cells),
        "fallback": False,
    }


def write_json_atomic(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes
    return ru / (1024.0 * 1024.0) if sys.platform == "darwin" else ru / 1024.0


def main(inp_arg: str, out_arg: str) -> int:
    t_boot = time.perf_counter()
    cfg, cfg_sha = load_variant_config()
    device = resolve_device(str(cfg.get("device", "auto")))
    git_sha = os.environ.get("TABLESALT_GIT_SHA", "unknown")

    model_source = str(MODEL_DIR) if MODEL_DIR.is_dir() else mf.MODEL_ID
    model = TableTransformerForObjectDetection.from_pretrained(
        model_source, local_files_only=True
    ).to(device).eval()
    startup_s = time.perf_counter() - t_boot

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    inp, out = Path(inp_arg), Path(out_arg)
    out.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in inp.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    print(
        f"[predict] variant={cfg['variant_id']} cfg_sha={cfg_sha} git={git_sha} "
        f"device={device} startup={startup_s:.2f}s images={len(images)}",
        flush=True,
    )

    per_table = []
    fallbacks = 0
    for img_path in images:
        dst = out / f"{img_path.stem}.json"
        try:
            cells, stats = run_single_table(img_path, model, device, cfg)
            if stats["fallback"]:
                fallbacks += 1
        except Exception as exc:
            fallbacks += 1
            traceback.print_exc(file=sys.stderr)
            cells = [{"text": "", "r": 0, "c": 0, "rs": 1, "cs": 1}]
            stats = {"total_s": 0.0, "struct_s": 0.0, "ocr_s": 0.0, "cells": 1, "fallback": True, "error": str(exc)}

        write_json_atomic(dst, cells)
        per_table.append({"id": img_path.stem, **stats})
        print(
            f"  {img_path.stem:12s} cells={stats['cells']:3d} "
            f"total={stats['total_s']:.2f}s (struct={stats['struct_s']:.2f}s ocr={stats['ocr_s']:.2f}s)"
            f"{' [FALLBACK]' if stats['fallback'] else ''}",
            flush=True,
        )

    times = [r["total_s"] for r in per_table]
    peak_vram_mb = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else 0.0
    summary = {
        "variant_id": cfg["variant_id"],
        "config_sha256": cfg_sha,
        "config": cfg,
        "git_sha": git_sha,
        "device": device,
        "startup_s": round(startup_s, 3),
        "n_tables": len(per_table),
        "fallbacks": fallbacks,
        "mean_s": round(float(np.mean(times)) if times else 0.0, 3),
        "p90_s": round(float(np.percentile(times, 90)) if times else 0.0, 3),
        "max_s": round(float(max(times)) if times else 0.0, 3),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "per_table": per_table,
    }
    print(
        f"[predict] DONE variant={cfg['variant_id']} n={summary['n_tables']} "
        f"mean={summary['mean_s']:.2f}s p90={summary['p90_s']:.2f}s max={summary['max_s']:.2f}s "
        f"rss={summary['peak_rss_mb']:.0f}MB vram={summary['peak_vram_mb']:.0f}MB fallbacks={fallbacks}",
        flush=True,
    )

    meta_path = os.environ.get("TABLESALT_META_PATH", "").strip()
    if meta_path:
        mp = Path(meta_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(mp, summary)

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: predict.py <input_dir> <output_dir>")
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
