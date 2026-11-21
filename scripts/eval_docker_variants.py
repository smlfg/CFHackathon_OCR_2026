#!/usr/bin/env python3
"""Controlled multi-variant Docker evaluation harness (official offline contract).

Runs one or more built Docker images (`tablesalt:v0`, `tablesalt:v1`, ...) with
`docker run --rm --network none`, passing ONLY `/data/images` (ro) and
`/data/predictions` plus an optional sidecar metadata file, then scores with the
official `score_all.py` / `evaluate.py` / `eval_qa.py` and writes a
machine-readable experiment manifest.

Usage:
  .venv/bin/python scripts/eval_docker_variants.py \
    --images tablesalt:v0 tablesalt:v1 \
    --gt data/dev --flat out/flat_dev --out out/docker_eval/dev
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_qa as qa  # noqa: E402
import evaluate as ev  # noqa: E402
import score_all as sa  # noqa: E402


def docker_image_info(image: str) -> dict:
    fmt = "{{.Id}}|{{.Size}}|{{.Architecture}}|{{.Os}}"
    raw = subprocess.check_output(["docker", "image", "inspect", "-f", fmt, image], text=True).strip()
    img_id, size_b, arch, os_name = raw.split("|")
    return {
        "image": image,
        "image_id": img_id,
        "size_mb": round(int(size_b) / 1e6, 1),
        "size_gb": round(int(size_b) / 1e9, 2),
        "platform": f"{os_name}/{arch}",
    }


def evaluate_predictions(gt_dir: Path, pred_dir: Path) -> dict:
    manifest = json.loads((gt_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = sa.score_team(pred_dir.name, gt_dir, pred_dir)

    by_style: dict[str, list[float]] = {}
    per_table: dict[str, dict] = {}
    for e in manifest:
        tid = e["id"]
        gt_html = (gt_dir / tid / "gt" / "table.html").read_text(encoding="utf-8")
        pred_html = ev.load_prediction(pred_dir, tid)
        teds = ev.teds(gt_html, pred_html) if pred_html else 0.0
        struct_teds = ev.teds(gt_html, pred_html, structure_only=True) if pred_html else 0.0
        f1 = ev.cell_f1(gt_html, pred_html) if pred_html else 0.0
        grid = qa.load_grid(pred_dir, tid)
        qas = json.loads((gt_dir / tid / "gt" / "qa.json").read_text(encoding="utf-8"))
        q_ok = sum(1 for q in qas if qa.grade_one(q, qa.auto_answer(q, grid)))

        by_style.setdefault(e["style"], []).append(teds)
        per_table[tid] = {
            "tier": e["tier"],
            "style": e["style"],
            "teds": round(teds, 4),
            "struct_teds": round(struct_teds, 4),
            "cell_f1": round(f1, 4),
            "qa_ok": q_ok,
            "qa_total": len(qas),
        }

    style_means = {k: round(sum(v) / len(v), 4) for k, v in sorted(by_style.items())}
    return {
        "final": round(summary["final"], 4),
        "teds_weighted": round(summary["teds_weighted"], 4),
        "qa_accuracy": round(summary["qa_accuracy"], 4),
        "cell_f1": round(summary["cell_f1"], 4),
        "per_tier": {k: round(v, 4) for k, v in sorted(summary["per_tier"].items())},
        "per_style": style_means,
        "per_table": per_table,
    }


def run_container(image: str, flat_dir: Path, pred_dir: Path, meta_dir: Path, use_gpu: bool) -> tuple[float, dict]:
    pred_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    for old in pred_dir.glob("*.json"):
        old.unlink()
    meta_file = meta_dir / "run_meta.json"
    if meta_file.exists():
        meta_file.unlink()

    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "-e", "TABLESALT_META_PATH=/meta/run_meta.json",
        "-v", f"{flat_dir.resolve()}:/data/images:ro",
        "-v", f"{pred_dir.resolve()}:/data/predictions",
        "-v", f"{meta_dir.resolve()}:/meta",
    ]
    if use_gpu:
        cmd.extend(["--gpus", "all"])
    cmd.append(image)

    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    wall_s = time.perf_counter() - t0
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    return round(wall_s, 2), meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--flat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", action="store_true")
    args = ap.parse_args()

    gt_dir, flat_dir, out_dir = Path(args.gt), Path(args.flat), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for img in args.images:
        slug = img.replace(":", "_").replace("/", "_")
        pred_dir = out_dir / slug / "predictions"
        meta_dir = out_dir / slug / "meta"
        img_info = docker_image_info(img)
        print(f"\n=== Running {img} ({img_info['size_gb']} GB, {img_info['platform']}) offline on {gt_dir} ===", flush=True)
        wall_s, run_meta = run_container(img, flat_dir, pred_dir, meta_dir, use_gpu=args.gpus)
        scores = evaluate_predictions(gt_dir, pred_dir)
        rec = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": str(gt_dir),
            "docker": img_info,
            "wall_s": wall_s,
            "runtime": {k: v for k, v in run_meta.items() if k != "per_table"},
            "scores": scores,
        }
        records.append(rec)
        pt = scores["per_tier"]
        ps = scores["per_style"]
        rt = rec["runtime"]
        print(
            f"RESULT {img:16s} FINAL={scores['final']:.4f} TEDSw={scores['teds_weighted']:.4f} "
            f"QA={scores['qa_accuracy']:.3f} F1={scores['cell_f1']:.4f} | "
            f"T0={pt['T0']:.3f} T1={pt['T1']:.3f} T2={pt['T2']:.3f} T3={pt['T3']:.3f} | "
            f"grid={ps.get('grid',0):.3f} hlines={ps.get('hlines',0):.3f} borderless={ps.get('borderless',0):.3f} | "
            f"mean={rt.get('mean_s',0):.2f}s startup={rt.get('startup_s',0):.1f}s rss={rt.get('peak_rss_mb',0):.0f}MB",
            flush=True,
        )

    # Compute pairwise regressions relative to first image (baseline)
    comparisons = []
    if len(records) >= 2:
        base_tables = records[0]["scores"]["per_table"]
        for rec in records[1:]:
            cur_tables = rec["scores"]["per_table"]
            improved, regressed = [], []
            for tid, bt in base_tables.items():
                ct = cur_tables[tid]
                delta = round(ct["teds"] - bt["teds"], 4)
                if delta > 1e-4:
                    improved.append({"id": tid, "delta": delta, "before": bt["teds"], "after": ct["teds"]})
                elif delta < -1e-4:
                    regressed.append({"id": tid, "delta": delta, "before": bt["teds"], "after": ct["teds"]})
            comparisons.append({
                "baseline": records[0]["docker"]["image"],
                "candidate": rec["docker"]["image"],
                "delta_final": round(rec["scores"]["final"] - records[0]["scores"]["final"], 4),
                "improved_count": len(improved),
                "regressed_count": len(regressed),
                "improved": sorted(improved, key=lambda x: -x["delta"]),
                "regressed": sorted(regressed, key=lambda x: x["delta"]),
            })
            print(
                f"\nDIFF {records[0]['docker']['image']} -> {rec['docker']['image']}: "
                f"delta_FINAL={comparisons[-1]['delta_final']:+.4f}  "
                f"improved={len(improved)}  regressed={len(regressed)}"
            )

    manifest_out = {
        "schema_version": 1,
        "dataset": str(gt_dir),
        "runs": records,
        "comparisons": comparisons,
    }
    (out_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nManifest written -> {out_dir / 'experiment_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
