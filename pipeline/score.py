"""Scoring: same official functions the leaderboard uses, plus per-table detail.

Imports score_all.py / evaluate.py / eval_qa.py from the repo root so there is
exactly one scoring implementation, and this pipeline can never drift from the
official number.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from . import config


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, config.ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = _load("evaluate")
qa = _load("eval_qa")


def score(gt_dir: Path, pred_dir: Path) -> tuple[dict, list[dict]]:
    """-> (aggregate metrics, per-table rows) using the official formulas."""
    manifest = json.loads((gt_dir / "manifest.json").read_text(encoding="utf-8"))
    per_tier: dict[str, list[float]] = {}
    f1s: list[float] = []
    qa_ok: list[bool] = []
    rows: list[dict] = []
    empty = 0

    for entry in manifest:
        tid = entry["id"]
        gt_html = (gt_dir / tid / "gt" / "table.html").read_text(encoding="utf-8")
        pred_html = ev.load_prediction(pred_dir, tid)
        empty += 1 if not pred_html else 0
        teds = ev.teds(gt_html, pred_html) if pred_html else 0.0
        f1 = ev.cell_f1(gt_html, pred_html) if pred_html else 0.0
        per_tier.setdefault(entry["tier"], []).append(teds)
        f1s.append(f1)
        grid = qa.load_grid(pred_dir, tid)
        correct = 0
        questions = json.loads((gt_dir / tid / "gt" / "qa.json").read_text(encoding="utf-8"))
        for question in questions:
            ok = qa.grade_one(question, qa.auto_answer(question, grid))
            qa_ok.append(ok)
            correct += 1 if ok else 0
        rows.append({
            "tid": tid,
            "tier": entry["tier"],
            "style": entry.get("style", ""),
            "has_spans": entry.get("has_spans", False),
            "teds": round(teds, 4),
            "cell_f1": round(f1, 4),
            "qa": round(correct / len(questions), 4) if questions else 0.0,
            "empty": not pred_html,
        })

    tier_mean = {t: sum(v) / len(v) for t, v in sorted(per_tier.items())}
    teds_w = ev.tier_weighted(tier_mean)
    qa_acc = sum(qa_ok) / len(qa_ok) if qa_ok else 0.0
    aggregate = {
        "n_tables": len(manifest),
        "final": round(ev.final_score(teds_w, qa_acc), 4),
        "teds_weighted": round(teds_w, 4),
        "qa_accuracy": round(qa_acc, 4),
        "cell_f1": round(sum(f1s) / len(f1s), 4),
        "per_tier": {t: round(v, 4) for t, v in tier_mean.items()},
        "empty_tables": empty,
    }
    return aggregate, rows
