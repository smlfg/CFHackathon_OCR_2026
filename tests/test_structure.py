"""Structure detection on small self-drawn tables with known geometry."""
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from tablesalt import preprocess, spans, structure

FONT = None
for p in ("C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
          "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
    try:
        FONT = ImageFont.truetype(p, 20)
        break
    except OSError:
        continue

pytestmark = pytest.mark.skipif(FONT is None, reason="no TrueType font available")

ROWS = [["Item", "Q1", "Q2", "Q3", "Q4"]] + [
    [lab, "1,234.56", "-98.70", "45,000.00", "7,725.25"]
    for lab in ("Revenue", "Taxes", "EBIT", "Capex", "Inventory", "Cash flow", "Net income")]


def draw(style: str, merge: bool = False) -> np.ndarray:
    col_x = [60, 200, 330, 460, 590, 720]
    row_y = [60 + 46 * i for i in range(len(ROWS) + 1)]
    img = Image.new("L", (col_x[-1] + 60, row_y[-1] + 90), 252)
    d = ImageDraw.Draw(img)
    for r, row in enumerate(ROWS):
        for c, text in enumerate(row):
            if merge and r == 3 and c in (2, 3):
                continue
            bb = d.textbbox((0, 0), text, font=FONT)
            w = bb[2] - bb[0]
            if r == 0:
                x = (col_x[c] + col_x[c + 1] - w) / 2
            elif c == 0:
                x = col_x[0] + 14
            else:
                x = col_x[c + 1] - 14 - w
            d.text((x, row_y[r] + 12), text, font=FONT, fill=25)
        if merge and r == 3:
            bb = d.textbbox((0, 0), "see note 3", font=FONT)
            d.text(((col_x[2] + col_x[4] - (bb[2] - bb[0])) / 2, row_y[r] + 12), "see note 3",
                   font=FONT, fill=25)
    if style == "grid":
        for y in row_y:
            d.line([(col_x[0], y), (col_x[-1], y)], fill=25, width=2)
        for x in col_x:
            d.line([(x, row_y[0]), (x, row_y[-1])], fill=25, width=2)
    elif style == "hlines":
        for y in (row_y[0], row_y[1], row_y[-1]):
            d.line([(col_x[0], y), (col_x[-1], y)], fill=25, width=2)
    return np.asarray(img)


@pytest.mark.parametrize("style", ["grid", "hlines", "borderless"])
def test_rows_and_columns(style):
    prep = preprocess.prepare(draw(style))
    lay = structure.analyze(prep.binary)
    assert len(lay.rows) == len(ROWS)
    assert lay.grid.n_cols == 5
    assert lay.extra["style"] == style


def test_rotated_table_is_straightened():
    img = Image.fromarray(draw("borderless")).rotate(2.5, expand=True, fillcolor=250)
    prep = preprocess.prepare(np.asarray(img))
    assert abs(abs(prep.angle) - 2.5) < 0.6
    lay = structure.analyze(prep.binary)
    assert len(lay.rows) == len(ROWS) and lay.grid.n_cols == 5


def test_merged_body_cell_gets_colspan():
    prep = preprocess.prepare(draw("hlines", merge=True))
    lay = structure.analyze(prep.binary)
    texts = [""] * len(lay.blobs)
    for i, b in enumerate(lay.blobs):
        if b.w > 80 and lay.rows[3][-1] != i and b.x0 > 300:
            texts[i] = "see note 3"
    cells, hr = spans.build_cells(lay, texts)
    merged = [c for c in cells if c.cs == 2]
    assert hr == 1
    assert len(merged) == 1 and (merged[0].r, merged[0].c) == (3, 2)
