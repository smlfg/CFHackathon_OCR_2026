#!/usr/bin/env python3
"""Ablate T3 fax image preprocessing for already-loaded MinerU2.5 1.2B on MLX server."""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval_qa as eq
import evaluate as ev
from pipeline import convert, providers


def apply_variant(img: Image.Image, variant: str) -> Image.Image:
    """Apply a named preprocessing variant to an RGB PIL image."""
    if variant == "median3":
        return img.filter(ImageFilter.MedianFilter(size=3))
    if variant == "upscale15_median3":
        w, h = img.size
        up = img.resize((round(w * 1.5), round(h * 1.5)), Image.LANCZOS)
        return up.filter(ImageFilter.MedianFilter(size=3))
    if variant == "despeckle1":
        arr = np.asarray(img, dtype=np.int16)
        med = np.asarray(img.filter(ImageFilter.MedianFilter(size=3)), dtype=np.int16)
        diff = np.max(np.abs(arr - med), axis=2) > 60
        padded = np.pad(arr, ((1, 1), (1, 1), (0, 0)), mode="edge")
        h, w = arr.shape[:2]
        near_med = np.zeros((h, w), dtype=np.int16)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                near_med += (np.max(np.abs(nb - med), axis=2) <= 40)
        mask = diff & (near_med >= 7)
        out = arr.copy()
        out[mask] = med[mask]
        return Image.fromarray(out.astype(np.uint8))
    raise ValueError(f"Unknown variant: {variant}")


def img_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_variant(provider: providers.MlxProvider, gt_dir: Path, ds_name: str,
                variant: str, force: bool = False) -> None:
    manifest = json.loads((gt_dir / "manifest.json").read_text(encoding="utf-8"))
    t3_entries = [e for e in manifest if e["tier"] == "T3"]
    out_dir = ROOT / "out" / "t3_preproc" / variant / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    for e in t3_entries:
        tid = e["id"]
        raw_path = out_dir / f"{tid}.txt"
        html_path = out_dir / f"{tid}.html"
        if raw_path.exists() and html_path.exists() and not force:
            continue
        img = Image.open(gt_dir / e["image"]).convert("RGB")
        proc_img = apply_variant(img, variant)
        png_bytes = img_to_png_bytes(proc_img)
        text, usage = provider.generate(
            png_bytes,
            "Table Recognition:",
            max_tokens=4096,
            temperature=0.0,
        )
        raw_path.write_text(text, encoding="utf-8")
        html, _ = convert.convert(text, "auto")
        html_path.write_text(html or "", encoding="utf-8")
        meta[tid] = {
            "seconds": usage.get("seconds"),
            "chars": len(text),
            "size": list(proc_img.size),
        }
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"  [{variant}/{ds_name}] {tid} {usage.get('seconds', 0):.2f}s {len(text)} chars",
              flush=True)


def score_t3(gt_dir: Path, pred_dir: Path, postprocess_fn=None) -> dict:
    """Compute official TEDS, TEDS-S, CellF1, QA over T3 tables in gt_dir."""
    manifest = json.loads((gt_dir / "manifest.json").read_text(encoding="utf-8"))
    t3_entries = [e for e in manifest if e["tier"] == "T3"]
    teds_vals, teds_s_vals, f1_vals = [], [], []
    qa_ok, qa_n = 0, 0
    for e in t3_entries:
        tid = e["id"]
        gt_html = (gt_dir / tid / "gt" / "table.html").read_text(encoding="utf-8")
        pred_html = ev.load_prediction(pred_dir, tid)
        img_path = gt_dir / e["image"]
        if pred_html and postprocess_fn is not None:
            pred_html = postprocess_fn(pred_html, img_path)
        if pred_html:
            teds_vals.append(ev.teds(gt_html, pred_html, False))
            teds_s_vals.append(ev.teds(gt_html, pred_html, True))
            f1_vals.append(ev.cell_f1(gt_html, pred_html))
            grid = eq.grid_from_html(pred_html)
        else:
            teds_vals.append(0.0)
            teds_s_vals.append(0.0)
            f1_vals.append(0.0)
            grid = None
        qas = json.loads((gt_dir / tid / "gt" / "qa.json").read_text(encoding="utf-8"))
        for q in qas:
            qa_ok += int(eq.grade_one(q, eq.auto_answer(q, grid)))
            qa_n += 1
    meta_path = pred_dir / "meta.json"
    s_per_tab = None
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        if isinstance(m, dict):
            secs = [v["seconds"] for v in m.values() if v.get("seconds") is not None]
            s_per_tab = sum(secs) / len(secs) if secs else None
    n = len(t3_entries)
    return {
        "n": n,
        "teds": sum(teds_vals) / n,
        "teds_s": sum(teds_s_vals) / n,
        "cell_f1": sum(f1_vals) / n,
        "qa": qa_ok / qa_n if qa_n else 0.0,
        "qa_ok": qa_ok,
        "qa_n": qa_n,
        "s_per_table": s_per_tab,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["median3", "upscale15_median3"])
    ap.add_argument("--datasets", nargs="+", default=["dev", "val96"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--with-postprocess", action="store_true")
    args = ap.parse_args()

    provider = providers.MlxProvider()
    if not args.score_only:
        if not provider.healthy():
            raise SystemExit("MLX server not reachable")
        provider.load("mlx-community/MinerU2.5-2509-1.2B-bf16")
        for variant in args.variants:
            for ds_name in args.datasets:
                gt_dir = ROOT / "data" / ds_name
                print(f"=== Running {variant} on {ds_name} T3 ===", flush=True)
                run_variant(provider, gt_dir, ds_name, variant, force=args.force)

    pp_fn = None
    if args.with_postprocess:
        from pipeline.postprocess import postprocess_html
        pp_fn = postprocess_html

    print("\n=== T3 PREPROCESSING ABLATION RESULTS ===")
    print(f"{'dataset':<8} {'variant':<20} {'T3_TEDS':>8} {'T3_TEDS-S':>9} {'T3_CellF1':>9} {'T3_QA':>10} {'s/tab':>6}")
    for ds_name in args.datasets:
        gt_dir = ROOT / "data" / ds_name
        base_dir = (ROOT / "predictions" / "mineru2.5-1.2b"
                    if ds_name == "dev"
                    else ROOT / "predictions" / ds_name / "mineru2.5-1.2b")
        rows = [("raw", base_dir)] + [
            (v, ROOT / "out" / "t3_preproc" / v / ds_name) for v in args.variants
        ]
        for label, pdir in rows:
            for pp_tag, fn in ([("raw_eval", None)] + ([("+postproc", pp_fn)] if pp_fn else [])):
                s = score_t3(gt_dir, pdir, fn)
                name = label if pp_tag == "raw_eval" else f"{label}+pp"
                st = f"{s['s_per_table']:.2f}" if s["s_per_table"] is not None else "—"
                print(f"{ds_name:<8} {name:<20} {s['teds']:8.4f} {s['teds_s']:9.4f} "
                      f"{s['cell_f1']:9.4f} {s['qa']:5.3f}({s['qa_ok']}/{s['qa_n']}) {st:>6}")


if __name__ == "__main__":
    main()
