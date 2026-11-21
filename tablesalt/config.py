"""All thresholds in one place. Lengths are in units of the median text height ``h``
(cap height of the body font, ~14 px at the generator's native scale) unless noted.

Change a value only together with a measurement on data/val (docs/RESULTS.md).
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- preprocess ---------------------------------------------------------
    bg_kernel: int = 31               # px, closing kernel for the illumination estimate
    bin_thresh_min: int = 120         # clamp for Otsu on the normalized image
    bin_thresh_max: int = 200
    speck_area: int = 5               # px, components up to this size are noise
    deskew_max_deg: float = 5.0       # generator rotates at most 4 deg
    deskew_coarse_step: float = 0.25
    deskew_fine_step: float = 0.05
    rectify_inlier_px: float = 0.30   # x h, line-fit inlier tolerance
    rectify_min_rows: int = 5         # need this many text rows to trust a homography

    # --- lines / style ------------------------------------------------------
    hline_len: float = 2.6            # x h, opening kernel for horizontal rules
    vline_len: float = 2.2            # x h, opening kernel for vertical rules
    hrule_min_len: float = 12.0       # x h, a horizontal rule is at least this long
    line_gap_close: float = 0.6       # x h, bridge broken rules (T3) before opening
    vline_min_count: int = 2          # inner vertical rules needed to call it "grid"
    vline_min_cover: float = 0.5      # share of table height a vertical rule must cover

    # --- blobs / rows / columns ---------------------------------------------
    blob_join_x: float = 0.85         # x h, horizontal dilation: joins words of one cell
    blob_join_y: int = 3              # px, joins i-dots / umlauts to their letters
    blob_min_ink: int = 10            # px of ink a blob needs
    blob_min_width: float = 0.45      # x h  (an en dash is ~0.75 h wide)
    row_split: float = 0.55           # x h, y-centre gap that starts a new row
    col_min_gap: float = 0.9          # x h, minimum empty gap between two columns
    col_cover_frac: float = 0.2       # share of body rows allowed to cross a gap (merges)
    edge_tol: float = 0.3             # x h, right edges closer than this are one column
    edge_support: float = 0.35        # share of body rows that must share a right edge
    edge_min_width: float = 2.4       # x h, median blob width on an edge (numbers, not n/a)
    min_cols: int = 4
    max_cols: int = 7

    # --- spans ----------------------------------------------------------------
    stub_mid_tol: float = 0.3         # share of the row pitch: rs=2 stub sits mid-way
    cross_px: float = 0.25            # x h a blob must reach into a neighbour column
    em_dash_min_w: float = 1.05       # x h, wider dashes are em dashes (merged cells)

    # --- OCR ------------------------------------------------------------------
    tess_lang: str = os.environ.get("TABLESALT_TESS_LANG", "eng")
    tess_cmd: str = os.environ.get("TABLESALT_TESSERACT", "tesseract")
    ocr_cap_height: int = 40          # px, crops are scaled so h becomes this
    ocr_pad: int = 12                 # px white border around each crop (after scaling)
    ocr_margin: int = 1               # px kept around the blob box before scaling
    ocr_median: int = 3               # median filter after scaling (0 = off); kills T3 specks
    tess_psm: int = 7                 # single text line
    # every blob is read once per variant (margin, median, psm); postprocess picks per role
    ocr_variants: tuple = tuple(
        tuple(int(x) for x in v.split(":"))
        for v in os.environ.get("TABLESALT_OCR_VARIANTS", "1:3:7,3:0:7,2:0:7").split(","))
    # psm 7 sometimes returns nothing for a clean line (layout analysis rejects it); psm 13
    # (raw line) reads those but drops leading '5's elsewhere -> only a rescue read
    rescue_variant: tuple = (3, 0, 13)
    num_whitelist: str = "0123456789.,-()%kgms"
    second_pass: bool = False         # whitelist pass for numeric cells

    # --- postprocess ----------------------------------------------------------
    label_min_score: float = 60.0     # rapidfuzz ratio needed to snap to the lexicon
    row_label_min_score: float = 40.0 # once the row-label pool is identified
    group_min_score: float = 30.0     # only 5 group names exist
    pool_min_share: float = 0.5       # share of headers that must agree on a pool

    # --- robustness -------------------------------------------------------------
    table_budget_s: float = 10.0      # skip optional passes after this


CFG = Config()
