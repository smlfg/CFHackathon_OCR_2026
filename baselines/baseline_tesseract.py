#!/usr/bin/env python3
"""
Floor baseline: Tesseract + geometric heuristics.

Words from Tesseract's TSV output are clustered into rows (by y-center)
and columns (by x-gap analysis), then emitted as a plain grid <table>
(no span detection — merged cells will be wrong by construction).

This is deliberately the weakest sensible pipeline: it defines the floor
every team should beat. Requires: apt install tesseract-ocr,
pip install pytesseract pillow.

Usage:
  python baselines/baseline_tesseract.py --gt data/dev --out predictions_tess/
"""

import argparse
import html
import json
import statistics
from pathlib import Path

import pytesseract
from PIL import Image


def words_from_image(image):
    """Tesseract TSV words. `image` is a path or a PIL.Image."""
    data = pytesseract.image_to_data(
        Image.open(image) if isinstance(image, (str, Path)) else image,
        output_type=pytesseract.Output.DICT)
    words = []
    for i, txt in enumerate(data["text"]):
        txt = txt.strip()
        if not txt or int(data["conf"][i]) < 0:
            continue
        x, y = data["left"][i], data["top"][i]
        w, h = data["width"][i], data["height"][i]
        words.append(dict(text=txt, x=x, y=y, w=w, h=h, cy=y + h / 2))
    return words


def cluster_rows(words):
    rows = []
    for wd in sorted(words, key=lambda d: d["cy"]):
        for row in rows:
            if abs(row["cy"] - wd["cy"]) < max(10, wd["h"] * 0.6):
                row["words"].append(wd)
                row["cy"] = statistics.mean(w["cy"] for w in row["words"])
                break
        else:
            rows.append(dict(cy=wd["cy"], words=[wd]))
    return [sorted(r["words"], key=lambda d: d["x"]) for r in rows]


def column_edges(rows):
    """Find column boundaries from the global x-gap histogram."""
    spans = []
    for row in rows:
        merged = []
        for wd in row:  # merge words separated by small gaps (same cell)
            if merged and wd["x"] - (merged[-1]["x"] + merged[-1]["w"]) < 18:
                merged[-1] = dict(text=merged[-1]["text"] + " " + wd["text"],
                                  x=merged[-1]["x"],
                                  w=wd["x"] + wd["w"] - merged[-1]["x"],
                                  y=wd["y"], h=wd["h"], cy=wd["cy"])
            else:
                merged.append(dict(wd))
        spans.append(merged)
    xs = sorted({s["x"] for row in spans for s in row})
    cols = []
    for x in xs:
        for c in cols:
            if abs(c - x) < 40:
                break
        else:
            cols.append(x)
    return spans, sorted(cols)


def to_html(img_path):
    return to_html_image(Image.open(img_path))


def to_html_image(image):
    """Naive image -> grid <table>. No spans, no preprocessing, no postprocess.

    `image` is normalized to RGB so pytesseract writes a lossless PNG temp file.
    Handing it a JPEG-backed image makes pytesseract re-encode the JPEG before
    OCR (PIL default quality), which lowers the score for a reason that has
    nothing to do with table recognition.
    """
    words = words_from_image(image.convert("RGB"))
    if not words:
        return "<table></table>"
    rows = cluster_rows(words)
    spans, cols = column_edges(rows)
    out = ["<table>"]
    for row in spans:
        cells = [""] * len(cols)
        for s in row:
            ci = min(range(len(cols)), key=lambda i: abs(cols[i] - s["x"]))
            cells[ci] = (cells[ci] + " " + s["text"]).strip()
        out.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>"
                                    for c in cells) + "</tr>")
    out.append("</table>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(args.gt) / "manifest.json").read_text(encoding="utf-8"))
    for e in manifest:
        img = Path(args.gt) / e["id"] / Path(e["image"]).name
        (out / f"{e['id']}.html").write_text(to_html(img), encoding="utf-8")
        print(e["id"], "done")


if __name__ == "__main__":
    main()
