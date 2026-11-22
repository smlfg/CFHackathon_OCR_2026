#!/usr/bin/env python3
"""Draw what the pipeline sees on top of the rectified image.

    python tools/debug_overlay.py data/val/table_036/image.png [--out dbg.png] [--ocr]

green boxes = text blobs (with OCR text if --ocr), blue = column bounds,
red = row bounds, magenta = removed rules, orange = rs=2 stub.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tablesalt import preprocess, structure
from tablesalt.io_utils import load_gray


def overlay(img_path: Path, with_ocr: bool = False) -> np.ndarray:
    prep = preprocess.prepare(load_gray(img_path))
    lay = structure.analyze(prep.binary)
    vis = cv2.cvtColor(prep.gray, cv2.COLOR_GRAY2BGR)
    rules = cv2.bitwise_or(lay.hline_mask, lay.vline_mask) > 0
    vis[rules] = (200, 0, 200)
    texts = {}
    if with_ocr:
        from tablesalt import ocr

        res = ocr.read_blobs(prep.gray, lay.blobs, lay.h, rules_mask=rules)
        texts = {i: t for i, (t, _) in enumerate(res)}
    for i, b in enumerate(lay.blobs):
        cv2.rectangle(vis, (b.x0, b.y0), (b.x1 - 1, b.y1 - 1), (0, 170, 0), 1)
        if i in texts:
            cv2.putText(
                vis,
                texts[i],
                (b.x0, b.y0 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 120, 255),
                1,
                cv2.LINE_AA,
            )
    H, W = vis.shape[:2]
    for x in lay.grid.col_bounds:
        cv2.line(vis, (int(x), 0), (int(x), H - 1), (255, 0, 0), 1)
    for y in lay.grid.row_bounds:
        cv2.line(vis, (0, int(y)), (W - 1, int(y)), (0, 0, 255), 1)
    if lay.stub_row is not None:
        for i in lay.rows[lay.stub_row]:
            b = lay.blobs[i]
            cv2.rectangle(vis, (b.x0 - 2, b.y0 - 2), (b.x1 + 1, b.y1 + 1), (0, 140, 255), 2)
    cv2.putText(
        vis,
        f"angle={prep.angle} H={prep.homography} h={lay.h:.1f} "
        f"rows={len(lay.rows)} cols={lay.grid.n_cols}",
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ocr", action="store_true")
    a = ap.parse_args()
    out = Path(a.out or Path(a.image).with_name("debug_overlay.png"))
    cv2.imwrite(str(out), overlay(Path(a.image), a.ocr))
    print(out)


if __name__ == "__main__":
    main()
