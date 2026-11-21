#!/usr/bin/env python3
"""CUDA benchmark harness: TATR (structure) -> Tesseract (text).

Answers three questions with numbers, not adjectives:

  1. How fast is the whole pipeline on CUDA, per table and per stage?
  2. How good is the output, scored with the official scorer?
  3. Does CUDA actually help, or is the bottleneck somewhere else?

The algorithm is `scripts/my_tatr_fusion.py`, unchanged and imported — this
harness only measures it. Per-stage timing matters because the two halves have
completely different cost profiles:

  - TATR structure inference: one DETR forward pass per table -> GPU-bound.
  - Tesseract cell OCR: N cells x subprocess call -> CPU-bound, and it is the
    part CUDA cannot touch.

So a fast CUDA number for stage 1 does not automatically mean a fast table.

Usage (inside the container):

  python3 bench_tatr_cuda.py --gt data/dev --out out/cuda_dev
  python3 bench_tatr_cuda.py --gt data/dev --out out/cuda_dev --device cpu
  python3 bench_tatr_cuda.py --gt data/dev --out out/cuda_dev --repeat 3

Output:
  <out>/<table_id>.json   cells in the official TableSalt format
  <out>/_report.json      per-table timings + shapes
  <out>/_bench.json       summary: throughput, stage split, GPU info
  stdout                  human-readable table
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

def find_app_root() -> Path:
    """Locate the app root, whether we run from the repo or from the image.

    In the repo the harness lives at ``benchmarks/cuda/bench_tatr_cuda.py`` and
    the app root is two levels up. In the image it is flattened to
    ``/app/bench_tatr_cuda.py`` and the root is one level up. We recognise the
    root by the marker files that must be there (``scripts/my_tatr_fusion.py``
    and ``score_all.py``) rather than by counting levels.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent, here.parent.parent, Path("/app")):
        if (candidate / "scripts" / "my_tatr_fusion.py").is_file() and \
           (candidate / "score_all.py").is_file():
            return candidate
    raise SystemExit(
        "cannot locate the app root: expected scripts/my_tatr_fusion.py and "
        "score_all.py in one of " + ", ".join(str(p) for p in
                                             (here, here.parent, here.parent.parent))
    )


APP = find_app_root()
for candidate in (APP, APP / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402


# --------------------------------------------------------------------------- env
def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def gpu_info(device: str) -> dict:
    info = {"device": device, "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available()}
    if device == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu_name": props.name,
            "gpu_total_mem_gb": round(props.total_memory / 1e9, 2),
            "gpu_capability": f"{props.major}.{props.minor}",
            "cudnn": torch.backends.cudnn.version(),
        })
    info["platform"] = platform.platform()
    info["cpu_count"] = os.cpu_count()
    return info


def sync(device: str) -> None:
    """CUDA is async. Without this the timer measures kernel launch, not work."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# --------------------------------------------------------------------------- run
def run_table(img: Image.Image, model, device: str, workers: int) -> tuple[list[dict], dict]:
    """One table through the pipeline, timed per stage. Returns cells + timings."""
    import my_tatr_fusion as fusion

    t: dict[str, float] = {}
    wall0 = time.perf_counter()

    t0 = time.perf_counter()
    clean = img.filter(ImageFilter.MedianFilter(3))
    t["median"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    objs = fusion.detect_structure_objects(clean, model, device)
    sync(device)
    t["tatr"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cells = fusion.structure_to_cells(objs)
    t["cells"] = time.perf_counter() - t0

    # Median only for OCR when there is real salt-and-pepper noise (keeps sharp text).
    ocr_img = clean if fusion.has_speckle_noise(img) else img
    t0 = time.perf_counter()
    filled = fusion.fill_cells_with_tesseract(ocr_img, cells, workers=workers)
    t["ocr"] = time.perf_counter() - t0

    # Wall clock for the table, including everything the stages do not cover
    # (image decode, JSON write, speckle probe).
    t["total"] = time.perf_counter() - wall0
    return filled, t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/dev",
                    help="dataset dir with manifest.json (mounted into the container)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    ap.add_argument("--workers", type=int, default=6,
                    help="Tesseract threads per table (CPU part)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="passes over the dataset; >1 warms CUDA/caches, last pass is timed")
    ap.add_argument("--limit", type=int, default=0, help="only first N tables")
    ap.add_argument("--model", default="",
                    help="override model path (default: baked-in /app/models/...)")
    ap.add_argument("--no-score", action="store_true",
                    help="skip the official scorer (predictions are still written)")
    args = ap.parse_args()

    device = resolve_device(args.device)
    gt_dir = Path(args.gt)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((gt_dir / "manifest.json").read_text(encoding="utf-8"))
    if args.limit:
        manifest = manifest[: args.limit]

    # The image bakes the checkpoint in; the repo falls back to the HF cache.
    baked = APP / "models" / "table-transformer-structure-recognition"
    if args.model:
        model_path = args.model
        local_only = Path(args.model).is_dir()
    elif baked.is_dir():
        model_path = str(baked)
        local_only = True
    else:
        # Repo fallback: let transformers resolve it from the HF cache/network.
        model_path = "microsoft/table-transformer-structure-recognition"
        local_only = False
        if Path("/app").is_dir() and Path("/.dockerenv").exists():
            raise SystemExit(
                f"TATR checkpoint missing at {baked}\n"
                "the image is supposed to contain it (benchmarks/cuda/fetch_weights.py). "
                "Rebuild the image."
            )

    from transformers import TableTransformerForObjectDetection

    print(f"device   {device}")
    print(f"model    {model_path}")
    print(f"dataset  {gt_dir}  ({len(manifest)} tables)")
    print(f"out      {out_dir}\n", flush=True)

    load_t0 = time.perf_counter()
    model = TableTransformerForObjectDetection.from_pretrained(
        model_path, local_files_only=local_only
    ).to(device).eval()
    load_seconds = time.perf_counter() - load_t0
    print(f"model loaded in {load_seconds:.2f}s\n", flush=True)

    rows: list[dict] = []
    for rep in range(args.repeat):
        rows = []
        warm = rep < args.repeat - 1
        for entry in manifest:
            tid = entry["id"]
            img = Image.open(gt_dir / entry["image"]).convert("RGB")
            cells, timings = run_table(img, model, device, args.workers)
            (out_dir / f"{tid}.json").write_text(
                json.dumps(cells, ensure_ascii=False, indent=1), encoding="utf-8")
            rows.append({
                "id": tid, "tier": entry["tier"], "style": entry["style"],
                "n_cells": len(cells),
                "n_rows": max((c["r"] + c["rs"]) for c in cells) if cells else 0,
                "n_cols": max((c["c"] + c["cs"]) for c in cells) if cells else 0,
                "gt_rows": entry.get("n_rows"), "gt_cols": entry.get("n_cols"),
                **{f"t_{k}": round(v, 4) for k, v in timings.items()},
            })
            print(f"  {tid} {entry['tier']:<3} {entry['style']:<10} "
                  f"cells={len(cells):3d}  tatr={timings['tatr']:.3f}s "
                  f"ocr={timings['ocr']:.2f}s  total={timings['total']:.2f}s"
                  + ("  (warmup)" if warm else ""), flush=True)

        if not warm:
            wall = sum(r["t_total"] for r in rows)
            print(f"\npass {rep + 1}/{args.repeat}: {wall:.1f}s total, "
                  f"{wall / len(rows):.2f}s/table\n", flush=True)

    (out_dir / "_report.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    # ------------------------------------------------------------- summary
    def stage(key: str) -> list[float]:
        return [r[f"t_{key}"] for r in rows]

    def pct(values: list[float], q: float) -> float:
        return float(np.percentile(values, q)) if values else 0.0

    wall_total = sum(stage("total"))
    summary = {
        "gpu_info": gpu_info(device),
        "dataset": str(gt_dir),
        "n_tables": len(rows),
        "model_load_seconds": round(load_seconds, 2),
        "repeat": args.repeat,
        "workers": args.workers,
        "throughput": {
            "wall_seconds": round(wall_total, 2),
            "s_per_table_mean": round(wall_total / len(rows), 3),
            "s_per_table_p50": round(pct(stage("total"), 50), 3),
            "s_per_table_p90": round(pct(stage("total"), 90), 3),
            "s_per_table_min": round(min(stage("total")), 3),
            "s_per_table_max": round(max(stage("total")), 3),
            "tables_per_second": round(len(rows) / wall_total, 3) if wall_total else None,
        },
        "stages": {
            key: {
                "mean_s": round(statistics.mean(stage(key)), 4),
                "p50_s": round(pct(stage(key), 50), 4),
                "p90_s": round(pct(stage(key), 90), 4),
                "share_pct": round(100 * sum(stage(key)) / wall_total, 1) if wall_total else 0.0,
            }
            for key in ("median", "tatr", "cells", "ocr")
        },
        "per_tier_s_per_table": {
            tier: round(statistics.mean([r["t_total"] for r in rows if r["tier"] == tier]), 3)
            for tier in sorted({r["tier"] for r in rows})
        },
        "shape_exact": {
            "correct": sum(1 for r in rows
                           if r["gt_rows"] == r["n_rows"] and r["gt_cols"] == r["n_cols"]),
            "total": len(rows),
        },
    }

    # ------------------------------------------------- official score
    if not args.no_score:
        summary["score"] = score_with_official_scorer(gt_dir, out_dir)

    (out_dir / "_bench.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    # ------------------------------------------------------------- print
    print("\n" + "=" * 74)
    print("TIMING")
    print("=" * 74)
    print(f"  device            {device}"
          + (f"  ({summary['gpu_info'].get('gpu_name', '?')})"
             if summary["gpu_info"].get("gpu_name") else ""))
    print(f"  model load        {load_seconds:.2f}s")
    print(f"  wall total        {wall_total:.1f}s  for {len(rows)} tables")
    print(f"  s/table           mean {summary['throughput']['s_per_table_mean']:.3f}  "
          f"p50 {summary['throughput']['s_per_table_p50']:.3f}  "
          f"p90 {summary['throughput']['s_per_table_p90']:.3f}  "
          f"max {summary['throughput']['s_per_table_max']:.3f}")
    print(f"  throughput        {summary['throughput']['tables_per_second']} tables/s")
    print()
    print(f"  {'stage':<10} {'mean':>9} {'p50':>9} {'p90':>9} {'share':>8}")
    for key, s in summary["stages"].items():
        print(f"  {key:<10} {s['mean_s']:9.4f} {s['p50_s']:9.4f} "
              f"{s['p90_s']:9.4f} {s['share_pct']:7.1f}%")
    print()
    shape = summary["shape_exact"]
    print(f"  shape exactly right {shape['correct']}/{shape['total']}")
    print(f"  per tier s/table   {summary['per_tier_s_per_table']}")

    if "score" in summary:
        print("\n" + "=" * 74)
        print("QUALITY (official scorer)")
        print("=" * 74)
        print(json.dumps(summary["score"], indent=1))

    print(f"\nreport -> {out_dir / '_bench.json'}")
    return 0


def score_with_official_scorer(gt_dir: Path, out_dir: Path) -> dict:
    """Run score_all.py unchanged. It is the truth; we never re-implement it."""
    cmd = [sys.executable, str(APP / "score_all.py"),
           "--gt", str(gt_dir), "--pred", str(out_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(APP))
    if proc.returncode != 0:
        return {"error": "score_all.py failed",
                "stderr": proc.stderr[-2000:], "stdout": proc.stdout[-2000:]}
    payload: dict = {"stdout": proc.stdout.strip()}
    for line in proc.stdout.splitlines():
        line = line.strip()
        for key in ("FINAL", "TEDS", "QA"):
            if line.upper().startswith(key):
                payload.setdefault("raw", []).append(line)
    try:
        import evaluate
        import eval_qa

        manifest = json.loads((gt_dir / "manifest.json").read_text(encoding="utf-8"))
        per_tier: dict[str, list[float]] = {}
        f1s, qa_ok = [], []
        for e in manifest:
            tid = e["id"]
            gt_html = (gt_dir / tid / "gt" / "table.html").read_text(encoding="utf-8")
            pred_html = evaluate.load_prediction(out_dir, tid)
            per_tier.setdefault(e["tier"], []).append(
                evaluate.teds(gt_html, pred_html) if pred_html else 0.0)
            f1s.append(evaluate.cell_f1(gt_html, pred_html) if pred_html else 0.0)
            grid = eval_qa.load_grid(out_dir, tid)
            for q in json.loads((gt_dir / tid / "gt" / "qa.json").read_text(encoding="utf-8")):
                qa_ok.append(eval_qa.grade_one(q, eval_qa.auto_answer(q, grid)))
        per_tier_mean = {t: sum(v) / len(v) for t, v in sorted(per_tier.items())}
        teds_w = evaluate.tier_weighted(per_tier_mean)
        qa_acc = sum(qa_ok) / len(qa_ok) if qa_ok else 0.0
        payload.update({
            "tier_teds": {t: round(v, 4) for t, v in per_tier_mean.items()},
            "teds_weighted": round(teds_w, 4),
            "qa_accuracy": round(qa_acc, 4),
            "cell_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
            "final": round(0.85 * teds_w + 0.15 * qa_acc, 4),
        })
    except Exception as exc:  # scoring is a bonus, timing is the point
        payload["parse_error"] = repr(exc)
    return payload


if __name__ == "__main__":
    sys.exit(main())
