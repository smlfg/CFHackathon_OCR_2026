#!/usr/bin/env python3
"""One command to measure: flatten -> pipeline -> score_all + evaluate -> docs/RESULTS.md.

    python tools/eval_local.py --gt data/val --note "what changed"
    python tools/eval_local.py --gt data/dev --note "..." --no-log   # do not append a row

Predictions land in out_eval/<dataset name>/, the per-table report next to them.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tablesalt import run as ts_run  # noqa: E402


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="dataset dir with manifest.json")
    ap.add_argument("--note", default="", help="what changed (goes into RESULTS.md)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    gt = Path(a.gt)
    name = gt.name
    flat = ROOT / "out_eval" / f"{name}_flat"
    pred = ROOT / "out_eval" / name
    load("flatten_dataset").flatten(gt, flat)
    if pred.exists():
        for p in pred.glob("*.json"):
            p.unlink()

    t = time.perf_counter()
    infos = ts_run.main(flat, pred, a.workers, log=str(pred.parent / f"{name}_run.json"))
    wall = time.perf_counter() - t
    per_table = statistics.mean(i["seconds"] for i in infos) if infos else 0.0

    sa = load("score_all")
    res = sa.score_team(name, gt, pred)
    ev = sa.ev
    manifest = json.loads((gt / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for e in manifest:
        gt_html = (gt / e["id"] / "gt" / "table.html").read_text(encoding="utf-8")
        ph = ev.load_prediction(pred, e["id"])
        rows.append(dict(id=e["id"], tier=e["tier"], style=e["style"],
                         teds=ev.teds(gt_html, ph) if ph else 0.0,
                         teds_s=ev.teds(gt_html, ph, True) if ph else 0.0))
    (pred.parent / f"{name}_report.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    def mean(key, **flt):
        v = [r[key] for r in rows if all(r[k] == x for k, x in flt.items())]
        return sum(v) / len(v) if v else 0.0

    tiers = " ".join(f"{t}:{res['per_tier'].get(t, 0):.3f}" for t in ("T0", "T1", "T2", "T3"))
    styles = " ".join(f"{s}:{mean('teds', style=s):.3f}" for s in ("grid", "hlines", "borderless"))
    fb = sum(1 for i in infos if i.get("mode") != "pipeline")
    print(f"\nFINAL {res['final']:.4f}  TEDSw {res['teds_weighted']:.4f}  QA {res['qa_accuracy']:.4f}"
          f"  CellF1 {res['cell_f1']:.4f}  TEDS-S {mean('teds_s'):.4f}")
    print(f"per tier  {tiers}\nper style {styles}")
    print(f"{per_table:.2f} s/table (per worker), wall {wall:.1f}s, fallbacks {fb}")

    if not a.no_log:
        p = ROOT / "docs" / "RESULTS.md"
        line = (f"| {dt.date.today().isoformat()} | {a.note or '-'} | {name} ({len(rows)}) "
                f"| {res['final']:.4f} | {res['per_tier'].get('T0', 0):.3f} "
                f"| {res['per_tier'].get('T1', 0):.3f} | {res['per_tier'].get('T2', 0):.3f} "
                f"| {res['per_tier'].get('T3', 0):.3f} | {res['qa_accuracy']:.3f} "
                f"| {mean('teds_s'):.3f} | {per_table:.2f} | {fb} |\n")
        with p.open("a", encoding="utf-8") as f:
            f.write(line)
        print(f"appended to {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
