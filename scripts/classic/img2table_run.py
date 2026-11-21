#!/usr/bin/env python3
"""img2table (classic CV, no weights) on the dev set.

Structure: ``img2table.tables.extractor.TableExtractor`` — line-aware, with
``borderless_tables=True`` so the borderless style has a chance. Text: per-cell
Tesseract (see ``tess.py``), because img2table's own OCR hooking returned no
content on these single-table crops and would have hidden the grid behind an
empty table.

Writes ``predictions/img2table/<table_id>.html`` in the competition format
(``<table><tr><td>…``), which is what ``score_all.py`` expects.

  .venv-img2table/bin/python scripts/classic/img2table_run.py \
      --gt data/dev --out predictions/img2table
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from tess import cell_text  # noqa: E402

from img2table.tables.extractor import TableExtractor  # noqa: E402


def extract(img_path: Path, borderless: bool):
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    tables = TableExtractor(img=img).extract_tables(borderless_tables=borderless)
    if not tables:
        return None, img
    # A crop may trigger more than one detection; keep the largest by area.
    table = max(tables, key=lambda t: t.area)
    return table, img


def run_table(img_path: Path) -> tuple[str, dict]:
    table, img = extract(img_path, borderless=True)
    used_borderless = True
    if table is None:
        table, img = extract(img_path, borderless=False)
        used_borderless = False
    if table is None or img is None:
        return "<table></table>", {"cells": 0, "borderless": used_borderless, "detected": False}

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # A merged cell appears in every row it covers; OCR each unique bbox once.
    seen: dict[tuple, str] = {}
    for row in table.rows:
        for cell in row.cells:
            key = (cell.x1, cell.y1, cell.x2, cell.y2)
            if key in seen:
                cell.content = seen[key]
                continue
            text = cell_text(pil, key)
            seen[key] = text
            cell.content = text

    et = table.extracted_table(image_width=img.shape[1], image_height=img.shape[0])
    html = et.html
    return html, {
        "cells": len(seen),
        "rows": len(table.rows),
        "cols": table.nb_columns,
        "borderless": used_borderless,
        "detected": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/dev")
    ap.add_argument("--out", default="predictions/img2table")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    gt = Path(args.gt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((gt / "manifest.json").read_text())
    ids = [m["id"] for m in manifest][: args.limit or None]

    log, t0 = {}, time.time()
    for i, tid in enumerate(ids, 1):
        entry = next(m for m in manifest if m["id"] == tid)
        t = time.time()
        try:
            html, info = run_table(gt / entry["image"])
        except Exception as exc:  # keep going, record why
            html, info = "<table></table>", {"error": f"{type(exc).__name__}: {exc}"}
        info["seconds"] = round(time.time() - t, 2)
        (out / f"{tid}.html").write_text(html)
        log[tid] = info
        print(f"[{i}/{len(ids)}] {tid} {entry['tier']:2s} rows={info.get('rows')} "
              f"cells={info.get('cells')} {info['seconds']}s {info.get('error', '')}")

    elapsed = time.time() - t0
    (out.parent / f"{out.name}.log.json").write_text(json.dumps(log, indent=1))
    ok = sum(1 for v in log.values() if v.get("detected"))
    print(f"\n{ok}/{len(ids)} tables detected · {elapsed / max(1, len(ids)):.2f}s/table")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
