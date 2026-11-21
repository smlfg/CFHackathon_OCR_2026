#!/usr/bin/env python3
"""Worst tables per tier and every failed QA question, for a scored prediction folder.

    python tools/worst_tables.py --gt data/val --pred out_eval/val [--n 5] [--cells]

--cells prints the differing cells (gt vs. prediction) of each worst table.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cell_diff(gt_dir: Path, pred_dir: Path, tid: str) -> list[str]:
    gt = {(c["r"], c["c"]): c for c in json.loads(
        (gt_dir / tid / "gt" / "cells.json").read_text(encoding="utf-8"))}
    pp = pred_dir / f"{tid}.json"
    pr = {(c["r"], c["c"]): c for c in json.loads(pp.read_text(encoding="utf-8"))} \
        if pp.exists() else {}
    out = []
    for k in sorted(set(gt) | set(pr)):
        g, p = gt.get(k), pr.get(k)
        gs = (g["text"], g.get("rs", 1), g.get("cs", 1)) if g else None
        ps = (p["text"], p.get("rs", 1), p.get("cs", 1)) if p else None
        if gs != ps:
            out.append(f"    {k}: gt={gs!r:32s} pred={ps!r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--cells", action="store_true")
    a = ap.parse_args()
    gt, pred = Path(a.gt), Path(a.pred)
    ev, qa = load("evaluate"), load("eval_qa")
    manifest = json.loads((gt / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for e in manifest:
        g = (gt / e["id"] / "gt" / "table.html").read_text(encoding="utf-8")
        p = ev.load_prediction(pred, e["id"])
        rows.append(dict(e, teds=ev.teds(g, p) if p else 0.0,
                         teds_s=ev.teds(g, p, True) if p else 0.0))
    for tier in ("T0", "T1", "T2", "T3"):
        worst = sorted((r for r in rows if r["tier"] == tier), key=lambda r: r["teds"])[:a.n]
        print(f"\n== {tier} worst ==")
        for r in worst:
            print(f"  {r['id']} {r['style']:10s} TEDS {r['teds']:.3f}  TEDS-S {r['teds_s']:.3f}")
            if a.cells and r["teds"] < 0.999:
                print("\n".join(cell_diff(gt, pred, r["id"])))
    print("\n== failed QA ==")
    for e in manifest:
        grid = qa.load_grid(pred, e["id"])
        for q in json.loads((gt / e["id"] / "gt" / "qa.json").read_text(encoding="utf-8")):
            ans = qa.auto_answer(q, grid)
            if not qa.grade_one(q, ans):
                print(f"  {e['id']} {e['tier']} {e['style']:10s} {q['type']:9s} "
                      f"want {q['answer']!r} got {ans!r}  | {q['question'][:70]}")


if __name__ == "__main__":
    main()
