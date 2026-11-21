#!/usr/bin/env python3
"""TATR -> Tesseract with Team-14 'polish' before and after, as switchable variants.

    raw            Samuel's my_tatr_fusion as is (median denoise -> TATR -> cell Tesseract)
    enh            our enhancement first (polish.enhance: shadow, deskew, perspective,
                   despeckle, crop; scale 1, grid lines KEPT so TATR can see them)
    repair         raw + our learned repair (vocabulary snapping, number-format rebuild)
    enh_repair     both

All variants share one TATR model load. Output follows score_all.py --teams:
    <out>/<variant>/predictions/<id>.json  +  <out>/<variant>/run.json

Run (Samuel's TATR venv, model already in the HF cache):
    .venv-tatr/bin/python scripts/tatr_polish.py --gt data/dev --out out/tatr_polish
    .venv/bin/python score_all.py --gt data/dev --teams out/tatr_polish

Self-test without torch/TATR (checks the polish pieces only):
    python scripts/tatr_polish.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from polish.enhance import enhance  # noqa: E402
from polish.repair import repair_cells  # noqa: E402

VARIANTS = ("raw", "enh", "repair", "enh_repair")


def polish_input(img: Image.Image, scale: float) -> Image.Image:
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    g, _ = enhance(bgr, scale=scale, no_lines=False)
    return Image.fromarray(g).convert("RGB")


def run_tatr(img: Image.Image, tf, model, device, workers: int, denoise: bool):
    """Samuel's pipeline for one image (same calls as my_tatr_fusion.main)."""
    clean = img.filter(ImageFilter.MedianFilter(3)) if denoise else img
    cells = tf.structure_to_cells(tf.detect_structure_objects(clean, model, device))
    ocr_img = clean if (denoise and tf.has_speckle_noise(img)) else img
    return tf.fill_cells_with_tesseract(ocr_img, cells, workers=workers)


def selftest() -> None:
    img = Image.open(ROOT / "data/dev/table_003/image.jpg").convert("RGB")
    out = polish_input(img, 1.0)
    assert out.size[0] > 100 and out.mode == "RGB"
    cells = [dict(text="Item", r=0, c=0, rs=1, cs=1, header=True),
             dict(text="Q1", r=0, c=1, rs=1, cs=1, header=True),
             dict(text="Lithlum", r=1, c=0, rs=1, cs=1, header=False),
             dict(text="1.234,56", r=1, c=1, rs=1, cs=1, header=False),
             dict(text="Carbon", r=2, c=0, rs=1, cs=1, header=False),
             dict(text="73:840:33", r=2, c=1, rs=1, cs=1, header=False),
             dict(text="5.120,00", r=3, c=1, rs=1, cs=1, header=False),
             dict(text="Silicon", r=3, c=0, rs=1, cs=1, header=False)]
    fixed = {(c["r"], c["c"]): c["text"] for c in repair_cells(cells)}
    assert fixed[(1, 0)] == "Lithium", fixed
    assert fixed[(2, 1)] == "73.840,33", fixed
    print("OK: polish enhancement + repair work (TATR not needed for this test)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/dev")
    ap.add_argument("--out", default="out/tatr_polish")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--scale", type=float, default=1.0, help="enhancement upscale (TATR resizes to 1000 px anyway)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return

    import torch
    import my_tatr_fusion as tf
    from transformers import TableTransformerForObjectDetection

    gt, out = Path(a.gt), Path(a.out)
    manifest = json.loads((gt / "manifest.json").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = TableTransformerForObjectDetection.from_pretrained(tf.MODEL_ID, local_files_only=True).to(device).eval()

    # repair variants reuse the OCR of their base variant, so TATR runs at most twice per table
    bases = sorted({"enh" if v.startswith("enh") else "raw" for v in a.variants})
    secs = {v: 0.0 for v in a.variants}
    for v in a.variants:
        (out / v / "predictions").mkdir(parents=True, exist_ok=True)

    for k, e in enumerate(manifest, 1):
        img = Image.open(gt / e["image"]).convert("RGB")
        for base in bases:
            t0 = time.perf_counter()
            src = polish_input(img, a.scale) if base == "enh" else img
            # our enhancement already removed specks; Samuel's median only for raw input
            cells = run_tatr(src, tf, model, device, a.workers, denoise=(base == "raw"))
            t_base = time.perf_counter() - t0
            for v in a.variants:
                if v not in (base, f"{base}_repair") and not (base == "raw" and v == "repair"):
                    continue
                t1 = time.perf_counter()
                result = repair_cells([dict(c) for c in cells]) if "repair" in v else cells
                secs[v] += t_base + (time.perf_counter() - t1)
                (out / v / "predictions" / f"{e['id']}.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{k}/{len(manifest)}] {e['id']} {e['tier']} {e['style']}", flush=True)

    for v in a.variants:
        (out / v / "run.json").write_text(json.dumps(
            dict(team=f"tatr_{v}", seconds=round(secs[v], 1), status="ok")), encoding="utf-8")
    print(f"\nDone. Score all variants:\n  python score_all.py --gt {gt} --teams {out}")


if __name__ == "__main__":
    main()
