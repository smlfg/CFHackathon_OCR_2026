#!/usr/bin/env python3
"""Camelot on the dev set — and what it costs to use it on raster images.

Camelot reads *PDFs* and takes text from the PDF text layer; on an image-only
page it refuses outright ("camelot only works on text-based pages"). The
documented way to use Camelot on scans is therefore a small chain:

    image → lossless one-page PDF (img2pdf, 1 px = 1 pt)
          → invisible OCR text layer (OCRmyPDF 17 / Tesseract 5.5.3, deu+eng)
          → Camelot (lattice, then stream) → cells → HTML

The chain matters: embedding the same image with Pillow's lossy JPEG PDF made
Camelot see a 3×3 grid on a 13×4 table. With the lossless page it recovers the
full grid, which is the only fair way to quote a Camelot number.

Both flavours are tried, the one with more cells wins. Nothing is tuned.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import img2pdf

sys.path.insert(0, str(Path(__file__).parent))


def cells_to_html(cells) -> tuple[str, int]:
    """Camelot cells -> <table> with rowspan/colspan derived from geometry."""
    boxes = []
    for cell in cells:
        boxes.append((round(cell.x1), round(cell.y1), round(cell.x2), round(cell.y2),
                      (cell.text or "").strip()))
    if not boxes:
        return "<table></table>", 0

    xs = sorted({b[0] for b in boxes} | {b[2] for b in boxes})
    # Camelot reports PDF coordinates: y grows upwards. Sort descending so row 0
    # is the top row of the table, not the bottom one.
    ys = sorted({b[1] for b in boxes} | {b[3] for b in boxes}, reverse=True)
    col = {x: i for i, x in enumerate(xs)}
    row = {y: i for i, y in enumerate(ys)}
    n_rows, n_cols = len(ys) - 1, len(xs) - 1
    if n_rows <= 0 or n_cols <= 0:
        return "<table></table>", 0

    grid: dict[tuple[int, int], dict] = {}
    for b in boxes:
        c0, c1 = col[b[0]], max(col[b[2]], col[b[0]] + 1)
        r0, r1 = row[b[3]], max(row[b[1]], row[b[3]] + 1)  # b[3] is the top edge
        grid[(r0, c0)] = {"rs": r1 - r0, "cs": c1 - c0, "text": b[4]}

    parts = ["<table>"]
    for r in range(n_rows):
        parts.append("<tr>")
        for c in range(n_cols):
            cell = grid.get((r, c))
            if cell is None:
                parts.append("<td></td>")
                continue
            attrs = ""
            if cell["rs"] > 1:
                attrs += f' rowspan="{cell["rs"]}"'
            if cell["cs"] > 1:
                attrs += f' colspan="{cell["cs"]}"'
            parts.append(f"<td{attrs}>{cell['text']}</td>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts), len(grid)


def run_table(img_path: Path, workdir: Path) -> tuple[str, dict]:
    import camelot

    pdf = workdir / "page.pdf"
    pdf.write_bytes(img2pdf.convert(str(img_path)))

    ocr_pdf = workdir / "page_ocr.pdf"
    subprocess.run(
        [str(Path(sys.executable).with_name("ocrmypdf")),
         "--output-type", "pdf", "--optimize", "0", "--quiet",
         "--language", "deu+eng", str(pdf), str(ocr_pdf)],
        check=True, capture_output=True,
    )

    best, flavour = None, None
    for flavor in ("lattice", "stream"):
        try:
            found = camelot.read_pdf(str(ocr_pdf), pages="1", flavor=flavor)
        except Exception:
            continue
        # Documented rule: lattice for ruled tables, stream only when no ruling
        # lines are found. "Most cells" would pick stream's over-segmentation.
        for table in found:
            # camelot 2.x returns rows of cells; flatten so a table is one list.
            flat = [c for row in table.cells for c in row]
            if best is None or len(flat) > len(best):
                best, flavour = flat, flavor
        if best:
            break  # lattice won; do not fall through to stream
    if not best:
        return "<table></table>", {"detected": False, "flavour": None, "cells": 0}

    html, n = cells_to_html(best)
    return html, {"detected": True, "flavour": flavour, "cells": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/dev")
    ap.add_argument("--out", default="predictions/camelot")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    gt, out = Path(args.gt), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    workdir = out.parent / "_camelot_pdf"
    workdir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((gt / "manifest.json").read_text())
    ids = [m["id"] for m in manifest][: args.limit or None]

    log, t0 = {}, time.time()
    for i, tid in enumerate(ids, 1):
        entry = next(m for m in manifest if m["id"] == tid)
        t = time.time()
        try:
            html, info = run_table(gt / entry["image"], workdir)
        except Exception as exc:
            html, info = "<table></table>", {"error": f"{type(exc).__name__}: {exc}"}
        info["seconds"] = round(time.time() - t, 2)
        (out / f"{tid}.html").write_text(html)
        log[tid] = info
        print(f"[{i}/{len(ids)}] {tid} {entry['tier']:2s} {entry['style']:10s} "
              f"flavour={info.get('flavour')} cells={info.get('cells')} "
              f"{info['seconds']}s {info.get('error', '')}")

    elapsed = time.time() - t0
    (out.parent / f"{out.name}.log.json").write_text(json.dumps(log, indent=1))
    ok = sum(1 for v in log.values() if v.get("detected"))
    print(f"\n{ok}/{len(ids)} tables detected · {elapsed / max(1, len(ids)):.2f}s/table")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
