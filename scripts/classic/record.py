#!/usr/bin/env python3
"""Put the classic extractors on the metrics board, scored by the official code.

The MLX board only knows models served over HTTP. The classic tools are local
libraries, so their predictions are scored with ``pipeline.score`` (which imports
``score_all.py`` / ``evaluate.py`` / ``eval_qa.py`` unchanged) and stored as
normal runs, so ``board`` and ``verify`` treat them like any other row.

  python scripts/classic/record.py --dataset dev
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from pipeline import config, score as scoring, store

# predictions dir -> board identity
TOOLS = {
    "pred_tatr_tess": {
        "id": "tatr-tesseract",
        "name": "TATR + Tesseract (structure + cell OCR)",
        "params_b": 0.0288,
        "quant": "fp32",
        "fmt": "html",
        "note": "structure only (Table Transformer 28.8M), text from Tesseract per cell; "
                "grid alone reaches TEDS_w 0.9365 with ground-truth text",
    },
    "ppstructure": {
        "id": "pp-structure",
        "name": "PP-Structure (PaddleOCR 3.7, stock)",
        "params_b": None,
        "quant": "fp32",
        "fmt": "html",
        "note": "TableRecognitionPipelineV2, use_layout_detection=False; built-in PP-OCRv4 text",
    },
    "img2table": {
        "id": "img2table",
        "name": "img2table (classic CV + Tesseract)",
        "params_b": None,
        "quant": "–",
        "fmt": "html",
        "note": "TableExtractor with borderless_tables; text from Tesseract per cell (psm 7)",
    },
    "docling": {
        "id": "docling",
        "name": "Docling 2.15 (TableFormer)",
        "params_b": None,
        "quant": "fp32",
        "fmt": "html",
        "note": "the toolkit (layout model + TableFormer + RapidOCR), not the Docling VLMs; CPU, stock settings",
    },
    "camelot": {
        "id": "camelot",
        "name": "Camelot 2.0 (PDF tool)",
        "params_b": None,
        "quant": "–",
        "fmt": "html",
        "note": "needs a PDF with a text layer: image -> img2pdf page -> OCRmyPDF text layer -> "
                "camelot lattice, else stream; not a model",
    },
}


def seconds_per_table(pred_dir: Path) -> float | None:
    log = pred_dir.parent / f"{pred_dir.name}.log.json"
    if not log.exists():
        return None
    data = json.loads(log.read_text())
    values = [v["seconds"] for v in data.values() if isinstance(v.get("seconds"), (int, float))]
    return round(sum(values) / len(values), 2) if values else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dev")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    gt = root / "data" / args.dataset
    manifest = json.loads((gt / "manifest.json").read_text())

    for key, meta in TOOLS.items():
        pred_dir = root / "predictions" / key
        if not pred_dir.is_dir():
            print(f"skip {key}: {pred_dir} missing")
            continue
        have = len(list(pred_dir.glob("table_*.html")))
        if have < len(manifest):
            print(f"skip {key}: only {have}/{len(manifest)} predictions")
            continue
        agg, rows = scoring.score(gt, pred_dir)
        model = {"id": meta["id"]}
        prompt_hash = hashlib.sha256(f"classic:{key}".encode()).hexdigest()
        run = {
            "run_id": store.run_id(model, args.dataset, prompt_hash),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_id": meta["id"],
            "model_name": meta["name"],
            "hf_repo": "",
            "params_b": meta["params_b"],
            "active_b": meta["params_b"],
            "quant": meta["quant"],
            "runtime": "local",
            "prompt_hash": prompt_hash,
            "dataset": args.dataset,
            "n_tables": agg["n_tables"],
            "final": agg["final"],
            "teds_weighted": agg["teds_weighted"],
            "qa_accuracy": agg["qa_accuracy"],
            "cell_f1": agg["cell_f1"],
            "per_tier": agg["per_tier"],
            "s_per_table": seconds_per_table(pred_dir),
            "empty_rate": round(agg["empty_tables"] / agg["n_tables"], 3),
            "status": "ok",
            "meta": {"fmt": meta["fmt"], "note": meta["note"], "predictions": str(pred_dir.name)},
        }
        store.save(run, rows)
        print(f"{meta['id']:16s} FINAL {agg['final']:.4f}  TEDS_w {agg['teds_weighted']:.4f}  "
              f"QA {agg['qa_accuracy']:.3f}  CellF1 {agg['cell_f1']:.4f}  "
              f"s/table {run['s_per_table']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
