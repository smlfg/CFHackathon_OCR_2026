#!/usr/bin/env python3
"""Ablate principled post-processing passes P0..P5 on dev and val96."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval_qa as eq
import evaluate as ev
from pipeline.postprocess import postprocess_html


def check_gt_zero_harm(datasets: list[str], flags: dict[str, bool]) -> int:
    """Verify postprocess_html leaves all 120 GT tables unchanged."""
    changed = 0
    total = 0
    for ds_name in datasets:
        gt_dir = ROOT / "data" / ds_name
        manifest = json.loads((gt_dir / "manifest.json").read_text(encoding="utf-8"))
        for e in manifest:
            total += 1
            tid = e["id"]
            gt_html = (gt_dir / tid / "gt" / "table.html").read_text(encoding="utf-8")
            img_path = gt_dir / e["image"]
            pp_html = postprocess_html(gt_html, img_path, **flags)
            t = ev.teds(gt_html, pp_html, False)
            f1 = ev.cell_f1(gt_html, pp_html)
            if t < 0.99999 or f1 < 0.99999:
                changed += 1
                print(f"  [GT HARM!] {ds_name}/{tid} ({e['tier']} {e['style']}): TEDS={t:.4f} CellF1={f1:.4f}")
                # Identify which single pass triggered
                for k in ("p0", "p1", "p2", "p3", "p4", "p5"):
                    if flags.get(k):
                        sub = {m: (m == k) for m in ("p0", "p1", "p2", "p3", "p4", "p5")}
                        sh = postprocess_html(gt_html, img_path, **sub)
                        if ev.teds(gt_html, sh, False) < 0.99999 or ev.cell_f1(gt_html, sh) < 0.99999:
                            print(f"    culprit: {k.upper()} -> {sh[:200]}")
    print(f"GT Zero-Harm Check ({datasets}): {total - changed}/{total} unchanged ({changed} harmed)")
    return changed


def eval_config(ds_name: str, model_id: str, flags: dict[str, bool],
                return_per_table: bool = False) -> tuple[dict, dict[str, tuple[float, float, int]]]:
    gt_dir = ROOT / "data" / ds_name
    pred_dir = (ROOT / "predictions" / model_id if ds_name == "dev"
                else ROOT / "predictions" / ds_name / model_id)
    manifest = json.loads((gt_dir / "manifest.json").read_text(encoding="utf-8"))

    per_tier_t = defaultdict(list)
    per_tier_s = defaultdict(list)
    f1s = []
    qa_ok = 0
    qa_n = 0
    per_table = {}
    t_start = time.time()

    use_any = any(flags.values())
    for e in manifest:
        tid = e["id"]
        gt_html = (gt_dir / tid / "gt" / "table.html").read_text(encoding="utf-8")
        pred_html = ev.load_prediction(pred_dir, tid)
        if pred_html and use_any:
            pred_html = postprocess_html(pred_html, gt_dir / e["image"], **flags)
        if pred_html:
            t = ev.teds(gt_html, pred_html, False)
            ts = ev.teds(gt_html, pred_html, True)
            f1 = ev.cell_f1(gt_html, pred_html)
            grid = eq.grid_from_html(pred_html)
        else:
            t = ts = f1 = 0.0
            grid = None
        per_tier_t[e["tier"]].append(t)
        per_tier_s[e["tier"]].append(ts)
        f1s.append(f1)
        qas = json.loads((gt_dir / tid / "gt" / "qa.json").read_text(encoding="utf-8"))
        ok = sum(1 for q in qas if eq.grade_one(q, eq.auto_answer(q, grid)))
        qa_ok += ok
        qa_n += len(qas)
        if return_per_table:
            per_table[tid] = (t, f1, ok)

    tm = {k: sum(v) / len(v) for k, v in sorted(per_tier_t.items())}
    sm = {k: sum(v) / len(v) for k, v in sorted(per_tier_s.items())}
    tw = ev.tier_weighted(tm)
    sw = ev.tier_weighted(sm)
    qa_acc = qa_ok / qa_n if qa_n else 0.0
    res = {
        "final": ev.final_score(tw, qa_acc),
        "teds_w": tw,
        "teds_sw": sw,
        "qa": qa_acc,
        "qa_ok": qa_ok,
        "qa_n": qa_n,
        "cell_f1": sum(f1s) / len(f1s),
        "T0": tm["T0"],
        "T1": tm["T1"],
        "T2": tm["T2"],
        "T3": tm["T3"],
        "secs": time.time() - t_start,
    }
    return res, per_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mineru2.5-1.2b")
    ap.add_argument("--datasets", nargs="+", default=["dev", "val96"])
    args = ap.parse_args()

    all_on = {f"p{i}": True for i in range(6)}
    check_gt_zero_harm(args.datasets, all_on)

    configs = [
        ("raw", {}),
        ("+P0 (phantom col)", {"p0": True}),
        ("+P1 (dash norm)", {"p1": True}),
        ("+P2 (num fmt)", {"p2": True}),
        ("+P3 (num cs + px)", {"p3": True}),
        ("+P4 (hdr corner)", {"p4": True}),
        ("+P5 (split ph + px)", {"p5": True}),
        ("cum P0..P1", {"p0": True, "p1": True}),
        ("cum P0..P2", {"p0": True, "p1": True, "p2": True}),
        ("cum P0..P3", {"p0": True, "p1": True, "p2": True, "p3": True}),
        ("cum P0..P4", {"p0": True, "p1": True, "p2": True, "p3": True, "p4": True}),
        ("cum P0..P5", {"p0": True, "p1": True, "p2": True, "p3": True, "p4": True, "p5": True}),
    ]

    for ds_name in args.datasets:
        print(f"\n### Post-processing ablation on `{ds_name}` ({args.model})")
        print("| Config | FINAL | TEDSw | TEDS-Sw | QA | CellF1 | T0 | T1 | T2 | T3 | ΔFINAL | ΔT3 |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        base_res = None
        base_pt = None
        for label, part in configs:
            flags = {f"p{i}": part.get(f"p{i}", False) for i in range(6)}
            res, pt = eval_config(ds_name, args.model, flags, return_per_table=(label in ("raw", "cum P0..P4", "cum P0..P5")))
            if base_res is None:
                base_res = res
                base_pt = pt
            df = res["final"] - base_res["final"]
            dt3 = res["T3"] - base_res["T3"]
            print(
                f"| {label} | {res['final']:.4f} | {res['teds_w']:.4f} | {res['teds_sw']:.4f} | "
                f"{res['qa']:.3f} | {res['cell_f1']:.4f} | {res['T0']:.3f} | {res['T1']:.3f} | "
                f"{res['T2']:.3f} | {res['T3']:.4f} | {df:+.4f} | {dt3:+.4f} |"
            )
            if label in ("cum P0..P4", "cum P0..P5"):
                diffs = []
                for tid, (t, f1, ok) in pt.items():
                    bt, bf1, bok = base_pt[tid]
                    if abs(t - bt) > 1e-4 or abs(f1 - bf1) > 1e-4 or ok != bok:
                        diffs.append((tid, t - bt, f1 - bf1, ok - bok, bt, t))
                worse = [d for d in diffs if d[1] < -1e-4 or d[3] < 0]
                print(f"  [{label}] changed tables={len(diffs)}, regressions={len(worse)}")
                for d in worse:
                    print(f"    REGRESSION {d[0]}: TEDS {d[4]:.4f}->{d[5]:.4f} ({d[1]:+.4f}) F1{d[2]:+.3f} QA{d[3]:+d}")


if __name__ == "__main__":
    main()
