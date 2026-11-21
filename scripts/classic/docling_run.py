#!/usr/bin/env python3
"""Docling (IBM) on the dev set — the toolkit, not the tiny Docling VLMs.

SmolDocling-256M and Granite-Docling-258M are on the MLX board; this is the
other thing with the same name: the Docling document pipeline with its layout
model and TableFormer, run on one table crop per image. Stock settings, CPU,
RapidOCR as shipped.

Writes ``predictions/docling/<table_id>.html``.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

_TBODY = re.compile(r"</?tbody>")


def html_rows(html: str) -> int:
    return html.count("<tr")


def run_table(conv, img_path: Path) -> tuple[str, dict]:
    res = conv.convert(str(img_path))
    doc = res.document
    if not doc.tables:
        return "<table></table>", {"detected": False, "tables": 0, "cells": 0}

    best_html, best_cells = "", 0
    for table in doc.tables:
        html = table.export_to_html(doc=doc)
        cells = html.count("<td") + html.count("<th")
        if cells > best_cells:
            best_html, best_cells = html, cells
    html = _TBODY.sub("", best_html)
    return html, {"detected": True, "tables": len(doc.tables), "cells": best_cells}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/dev")
    ap.add_argument("--out", default="predictions/docling")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    gt, out = Path(args.gt), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from docling.document_converter import DocumentConverter

    conv = DocumentConverter()  # weights load once, reused for every table

    manifest = json.loads((gt / "manifest.json").read_text())
    ids = [m["id"] for m in manifest][: args.limit or None]

    log, t0 = {}, time.time()
    for i, tid in enumerate(ids, 1):
        entry = next(m for m in manifest if m["id"] == tid)
        t = time.time()
        try:
            html, info = run_table(conv, gt / entry["image"])
        except Exception as exc:
            html, info = "<table></table>", {"error": f"{type(exc).__name__}: {exc}"}
        info["seconds"] = round(time.time() - t, 2)
        (out / f"{tid}.html").write_text(html)
        log[tid] = info
        print(f"[{i}/{len(ids)}] {tid} {entry['tier']:2s} {entry['style']:10s} "
              f"cells={info.get('cells')} {info['seconds']}s {info.get('error', '')}",
              flush=True)

    elapsed = time.time() - t0
    (out.parent / f"{out.name}.log.json").write_text(json.dumps(log, indent=1))
    ok = sum(1 for v in log.values() if v.get("detected"))
    print(f"\n{ok}/{len(ids)} tables detected · {elapsed / max(1, len(ids)):.2f}s/table")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
