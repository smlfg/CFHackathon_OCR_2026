"""Small data containers passed between the pipeline steps."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Blob:
    """One connected text region in the rectified image, normally one cell's text."""
    x0: int
    y0: int
    x1: int          # exclusive
    y1: int          # exclusive
    ink: int         # number of ink pixels
    text: str = ""
    conf: float = -1.0
    lead_dash: bool = False   # left-most glyph is a flat mid-height bar (minus sign)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


@dataclass
class Prepared:
    """Output of preprocess: rectified grayscale + binary ink mask."""
    gray: np.ndarray            # uint8, illumination-normalized, rectified
    binary: np.ndarray          # uint8 0/255, 255 = ink, rectified, despeckled
    angle: float = 0.0          # deskew angle applied (deg)
    homography: bool = False    # True if a perspective correction was applied


@dataclass
class Grid:
    """Logical grid in rectified pixel coordinates."""
    row_bounds: list[float]     # len n_rows + 1
    col_bounds: list[float]     # len n_cols + 1
    row_centers: list[float]

    @property
    def n_rows(self) -> int:
        return len(self.row_bounds) - 1

    @property
    def n_cols(self) -> int:
        return len(self.col_bounds) - 1


@dataclass
class Layout:
    """Everything structure.py found: blobs grouped into rows, plus the grid."""
    h: float                                    # median text height (px)
    blobs: list[Blob]
    rows: list[list[int]]                       # blob indices per text row, top to bottom
    grid: Grid
    hline_mask: np.ndarray
    vline_mask: np.ndarray
    stub_row: int | None = None                 # index into rows of a mid-way rs=2 stub
    extra: dict = field(default_factory=dict)


@dataclass
class Cell:
    text: str
    r: int
    c: int
    rs: int = 1
    cs: int = 1
    blobs: list[int] = field(default_factory=list)
    header: bool = False
    neg_hint: bool = False    # geometry saw a leading minus

    def to_json(self) -> dict:
        return {"text": self.text, "r": self.r, "c": self.c, "rs": self.rs, "cs": self.cs}
