#!/usr/bin/env python3
"""Run the main-branch TableSalt pipeline on each real-data sample and score TEDS.

Runs this repository's submission pipeline (or REALDATA_PIPELINE=<checkout>), never a
copy of it. Reports unweighted mean TEDS / TEDS-S / Cell-F1 per dataset.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# The pipeline under test: this repository (override with REALDATA_PIPELINE=/path/to/checkout).
SNAPSHOT = Path(os.environ.get("REALDATA_PIPELINE", ROOT.parents[1])).resolve()
SAMPLES = ROOT / "samples"
PREDS = ROOT / "predictions"
REPORTS = ROOT / "reports"

sys.path.insert(0, str(SNAPSHOT))
from evaluate import cell_f1, load_prediction, teds  # noqa: E402


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def eval_dataset(name: str, pred_dir: Path) -> dict:
    gt_dir = SAMPLES / name
    manifest = json.loads((gt_dir / "manifest.json").read_text(encoding="utf-8"))
    source = {}
    src_path = gt_dir / "SOURCE.json"
    if src_path.exists():
        source = json.loads(src_path.read_text(encoding="utf-8"))

    rows = []
    for entry in manifest:
        tid = entry["id"]
        gt_html = (gt_dir / tid / "gt" / "table.html").read_text(encoding="utf-8")
        pred_html = load_prediction(pred_dir, tid)
        if pred_html is not None:
            scores = dict(
                teds=round(teds(gt_html, pred_html), 4),
                teds_s=round(teds(gt_html, pred_html, True), 4),
                cell_f1=round(cell_f1(gt_html, pred_html), 4),
                missing=False,
            )
        else:
            scores = dict(teds=0.0, teds_s=0.0, cell_f1=0.0, missing=True)
        rows.append({**entry, **scores})

    present = [r for r in rows if not r["missing"]]
    summary = {
        "dataset": name,
        "n": len(rows),
        "n_scored": len(present),
        "n_missing": sum(1 for r in rows if r["missing"]),
        "mean_teds": round(mean([r["teds"] for r in rows]), 4),
        "mean_teds_s": round(mean([r["teds_s"] for r in rows]), 4),
        "mean_cell_f1": round(mean([r["cell_f1"] for r in rows]), 4),
        "has_text_gt": source.get("has_text_gt", name != "icdar2019_ctdar"),
        "license": source.get("license"),
        "link": source.get("link"),
        "domain": source.get("domain"),
        "real_or_degraded": source.get("real_or_degraded"),
        "notes": source.get("notes"),
        "per_table": rows,
    }
    return summary


def run_predict(name: str, workers: int, timeout_s: int) -> dict:
    img_dir = SAMPLES / name / "images"
    pred_dir = PREDS / name
    if pred_dir.exists():
        shutil.rmtree(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    predict = SNAPSHOT / "submission_template" / "predict.py"
    n = len(list(img_dir.glob("*.png"))) + len(list(img_dir.glob("*.jpg")))
    cmd = [sys.executable, str(predict), str(img_dir), str(pred_dir)]
    # predict.py uses available_cpus(); cap via env if we want fewer workers.
    env = os.environ.copy()
    env.setdefault("OMP_THREAD_LIMIT", "1")
    # We cannot pass workers to this predict.py (it always uses all CPUs).
    # Limit visibility of CPUs so it won't oversubscribe RAM.
    if workers and hasattr(os, "sched_setaffinity"):
        pass  # set on the child only
    print(f"[{name}] predict {n} images -> {pred_dir}  timeout={timeout_s}s")
    t0 = time.time()
    preexec = None
    if workers and hasattr(os, "sched_setaffinity"):

        def preexec():
            with contextlib.suppress(Exception):
                os.sched_setaffinity(0, set(range(max(1, workers))))

    proc = subprocess.run(
        cmd,
        cwd=str(SNAPSHOT),
        env=env,
        timeout=timeout_s,
        capture_output=True,
        text=True,
        preexec_fn=preexec,
    )
    elapsed = time.time() - t0
    log = REPORTS / f"{name}_predict.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        f"cmd: {cmd}\nexit: {proc.returncode}\nelapsed: {elapsed:.1f}s\n\n"
        f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        print(f"[{name}] predict failed rc={proc.returncode}  see {log}")
        # Keep going: evaluate whatever files were written.
    else:
        print(f"[{name}] predict done in {elapsed:.1f}s  ({elapsed / max(n, 1):.1f}s/table)")
    return {
        "elapsed_s": round(elapsed, 1),
        "returncode": proc.returncode,
        "n_images": n,
        "seconds_per_table": round(elapsed / max(n, 1), 2),
    }


def print_summary(summary: dict) -> None:
    print(f"\n== {summary['dataset']}  n={summary['n']} missing={summary['n_missing']} ==")
    print(f"{'id':40s} {'TEDS':>7s} {'TEDS-S':>7s} {'CellF1':>7s}")
    for r in summary["per_table"]:
        flag = "  MISSING" if r["missing"] else ""
        print(f"{r['id'][:40]:40s} {r['teds']:7.4f} {r['teds_s']:7.4f} {r['cell_f1']:7.4f}{flag}")
    print(
        f"MEAN TEDS={summary['mean_teds']:.4f}  TEDS-S={summary['mean_teds_s']:.4f}  "
        f"CellF1={summary['mean_cell_f1']:.4f}"
    )
    if summary["dataset"] == "icdar2019_ctdar" or summary.get("has_text_gt") is False:
        print(
            "NOTE: this dataset has no (or incomplete) cell-text ground truth; "
            "TEDS and CellF1 are not fair content scores. Use TEDS-S."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument(
        "--budget-s",
        type=int,
        default=90 * 60,
        help="total wall-clock budget in seconds for all predict runs",
    )
    ap.add_argument("--skip-predict", action="store_true")
    args = ap.parse_args()

    names = args.only or sorted(p.name for p in SAMPLES.iterdir() if (p / "manifest.json").exists())
    REPORTS.mkdir(parents=True, exist_ok=True)
    remaining = args.budget_s
    all_summaries = []
    timing = {}

    for name in names:
        n = len(json.loads((SAMPLES / name / "manifest.json").read_text()))
        # leave a little time for the last datasets
        left_ds = max(1, len(names) - len(timing))
        timeout = max(120, min(remaining, max(30 * n, remaining // left_ds)))
        if not args.skip_predict:
            if remaining <= 30:
                print(f"[{name}] skipped predict, budget exhausted")
                timing[name] = {"elapsed_s": 0, "returncode": None, "skipped": True}
            else:
                try:
                    timing[name] = run_predict(name, workers=args.workers, timeout_s=timeout)
                except subprocess.TimeoutExpired:
                    print(f"[{name}] predict timed out after {timeout}s")
                    timing[name] = {"elapsed_s": timeout, "returncode": "timeout", "n_images": n}
                remaining = max(0, remaining - timing[name]["elapsed_s"])
        summary = eval_dataset(name, PREDS / name)
        summary["timing"] = timing.get(name, {})
        print_summary(summary)
        all_summaries.append(summary)
        (REPORTS / f"{name}_report.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    headline = []
    for s in all_summaries:
        headline.append(
            {
                "dataset": s["dataset"],
                "n": s["n"],
                "mean_teds": s["mean_teds"],
                "mean_teds_s": s["mean_teds_s"],
                "mean_cell_f1": s["mean_cell_f1"],
                "n_missing": s["n_missing"],
                "has_text_gt": s["has_text_gt"],
                "link": s["link"],
                "license": s["license"],
                "domain": s["domain"],
                "real_or_degraded": s["real_or_degraded"],
                "timing": s.get("timing"),
                "gap_to_0.99_teds": round(0.99 - s["mean_teds"], 4),
            }
        )
    out = {
        "target": 0.99,
        "target_note": (
            "Samuel's success target is 99% TEDS on five new real datasets. "
            "Numbers below are measured, not invented. The current pipeline was "
            "tuned on 24 synthetic dev tables and is not expected to hit 99% on real scans."
        ),
        "datasets": headline,
        "n_at_or_above_99": sum(
            1 for h in headline if h["mean_teds"] >= 0.99 and h["has_text_gt"] is not False
        ),
    }
    (REPORTS / "summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n======== OVERALL ========")
    print(
        f"{'dataset':20s} {'n':>4s} {'TEDS':>8s} {'TEDS-S':>8s} {'CellF1':>8s} {'gap to 99%':>10s}"
    )
    for h in headline:
        print(
            f"{h['dataset']:20s} {h['n']:4d} {h['mean_teds']:8.4f} {h['mean_teds_s']:8.4f} "
            f"{h['mean_cell_f1']:8.4f} {h['gap_to_0.99_teds']:10.4f}"
        )
    print(f"datasets with TEDS >= 0.99 (and text GT): {out['n_at_or_above_99']} / {len(headline)}")
    print(f"wrote {REPORTS / 'summary.json'}")


if __name__ == "__main__":
    main()
