#!/usr/bin/env python3
"""Per-cell Tesseract text, used by every structure-only classic tool.

The classic extractors (img2table, Camelot) return geometry but no text, or
text from a PDF layer that a raster image does not have. To make their scores
comparable with the TATR run (challenge/TATR_RESEARCH.md §6 row D), text is
read the same way there: one Tesseract call per cell crop, ``--psm 7``.

So a score from these tools answers "how good is the grid, with the same
text engine TATR used", not "how good is this library's own OCR".
"""
from __future__ import annotations

import re

import pytesseract
from PIL import Image

_WHITESPACE = re.compile(r"\s+")

TESS_CONFIG = "--psm 7"
TESS_LANG = "deu+eng"


def cell_text(img: Image.Image, bbox: tuple[int, int, int, int], margin: int = 2) -> str:
    """OCR one cell crop. Returns '' when Tesseract sees nothing."""
    x1, y1, x2, y2 = bbox
    w, h = img.size
    box = (max(0, x1 - margin), max(0, y1 - margin), min(w, x2 + margin), min(h, y2 + margin))
    if box[2] - box[0] < 3 or box[3] - box[1] < 3:
        return ""
    text = pytesseract.image_to_string(img.crop(box), lang=TESS_LANG, config=TESS_CONFIG)
    return _WHITESPACE.sub(" ", text).strip()
