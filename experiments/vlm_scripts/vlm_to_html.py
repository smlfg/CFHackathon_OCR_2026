#!/usr/bin/env python3
"""Turn raw model text into TableSalt prediction files (HTML, for the gallery).

    python experiments/vlm_scripts/vlm_to_html.py --raw raw/paddle  --out pred_paddle  --format otsl
    python experiments/vlm_scripts/vlm_to_html.py --raw raw/lighton --out pred_lighton --format auto

`otsl` understands PaddleOCR-VL's OTSL cells (<fcel> <lcel> <ucel> <xcel> <ecel> <nl>)
and reconstructs rowspan/colspan. `auto` uses lighton_cells, which keeps a real HTML
table if the model wrote one and otherwise reads a markdown pipe table (spans are
impossible there). For the official cells JSON use lighton_cells.py directly.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: official scorer
from lighton_cells import text_to_cells

from evaluate import cells_json_to_html

TAG = re.compile(r"<(fcel|ecel|lcel|ucel|xcel|nl)>")


def otsl_to_cells(text: str) -> list[dict]:
    parts = TAG.split(text)
    cells: list[dict] = []
    cover: dict[tuple[int, int], dict] = {}
    r = c = 0
    for i in range(1, len(parts) - 1, 2):
        tag, body = parts[i], parts[i + 1]
        if tag == "nl":
            r, c = r + 1, 0
        elif tag in ("fcel", "ecel"):
            cell = {"text": "" if tag == "ecel" else body.strip(), "r": r, "c": c, "rs": 1, "cs": 1}
            cells.append(cell)
            cover[(r, c)] = cell
            c += 1
        elif tag == "lcel":
            left = cover.get((r, c - 1))
            if left is not None:  # extend that cell right
                if left["r"] == r:
                    left["cs"] += 1
                cover[(r, c)] = left
            c += 1
        elif tag == "ucel":
            up = cover.get((r - 1, c))
            if up is not None:  # extend that cell down
                if up["c"] == c:
                    up["rs"] += 1
                cover[(r, c)] = up
            c += 1
        elif tag == "xcel":
            up = cover.get((r - 1, c - 1))
            if up is not None:
                up["rs"] += 1
                up["cs"] += 1
                cover[(r, c)] = up
            c += 1
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", default="auto", choices=["auto", "otsl"])
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    for path in sorted(raw.glob("*.txt")):
        text = path.read_text()
        cells = otsl_to_cells(text) if args.format == "otsl" else text_to_cells(text)
        if not cells:
            print(f"{path.stem}: EMPTY")
            continue
        (out / f"{path.stem}.html").write_text(cells_json_to_html(cells))
        done += 1
    print(f"{done} predictions -> {out}")


if __name__ == "__main__":
    main()
