"""Step 3 - structure: text blobs, text rows, column bounds.

The generator writes every cell on one line, keeps >= 28 px between cells and ~6 px between
words.  A horizontal dilation of ~0.85 h therefore turns each cell's text into exactly one
connected *blob*.  Rows are clusters of blob y-centres; columns are the empty vertical gaps
in the body rows.  Works the same for grid, hlines and borderless once rules are removed.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import style as style_mod
from .config import CFG, Config
from .types import Blob, Grid, Layout


# ----------------------------------------------------------------------------
# text height and rules
# ----------------------------------------------------------------------------

def text_height(binary: np.ndarray) -> float:
    """Median height of character-sized components (~cap height of the body font)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    ws = stats[1:, cv2.CC_STAT_WIDTH]
    keep = (hs >= 8) & (hs <= 60) & (ws <= 3 * hs)
    if keep.sum() < 5:
        return 14.0
    return float(np.median(hs[keep]))


def line_masks(binary: np.ndarray, h: float, cfg: Config = CFG) -> tuple[np.ndarray, np.ndarray]:
    """Long horizontal / vertical rules as two masks (255 = rule)."""
    hl = max(25, int(cfg.hline_len * h))
    vl = max(22, int(cfg.vline_len * h))
    # fill single-pixel holes across the rule (T3 specks) without joining letters sideways
    hb = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    hmask = cv2.morphologyEx(hb, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hl, 1)))
    vb = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    vmask = cv2.morphologyEx(vb, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vl)))
    # Horizontal rules span (almost) the whole table; blurred words can form long runs too,
    # so a rule must be >= hrule_min_len*h AND >= half the longest rule.
    hmask = _keep_long(hmask, axis=1, min_len=1.3 * hl,
                       abs_min=cfg.hrule_min_len * h, rel_min=0.5)
    vmask = _keep_long(vmask, axis=0, min_len=1.2 * vl)
    return hmask, vmask


def _keep_long(mask: np.ndarray, axis: int, min_len: float, abs_min: float = 0.0,
               rel_min: float = 0.0) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    if n <= 1:
        return out
    key = cv2.CC_STAT_WIDTH if axis == 1 else cv2.CC_STAT_HEIGHT
    lengths = stats[1:, key]
    longest = float(lengths.max())
    if longest < abs_min:
        return out
    thr = max(min_len, rel_min * longest)
    keep = np.zeros(n, bool)
    keep[1:] = lengths >= thr
    out[keep[lab]] = 255
    return out


def remove_lines(binary: np.ndarray, hmask: np.ndarray, vmask: np.ndarray) -> np.ndarray:
    rules = cv2.dilate(cv2.bitwise_or(hmask, vmask), np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(binary, cv2.bitwise_not(rules))


# ----------------------------------------------------------------------------
# blobs and rows
# ----------------------------------------------------------------------------

def find_blobs(text: np.ndarray, h: float, cfg: Config = CFG) -> list[Blob]:
    """Join characters of one cell into a blob; drop specks."""
    kx = max(3, int(round(cfg.blob_join_x * h)))
    joined = cv2.dilate(text, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, cfg.blob_join_y)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    blobs: list[Blob] = []
    for i in range(1, n):
        x, y, w, hh, _ = stats[i]
        sub = text[y:y + hh, x:x + w] & (lab[y:y + hh, x:x + w] == i).astype(np.uint8) * 255
        ink = int(np.count_nonzero(sub))
        if ink < cfg.blob_min_ink:
            continue
        ys, xs = np.nonzero(sub)
        bx0, bx1 = x + int(xs.min()), x + int(xs.max()) + 1
        by0, by1 = y + int(ys.min()), y + int(ys.max()) + 1
        bw, bh = bx1 - bx0, by1 - by0
        if bw < cfg.blob_min_width * h and bh < 0.6 * h:
            continue                       # speck cluster, not a dash or a character
        if bh > 2.5 * h:                   # leftover rule fragment or picture junk
            continue
        blob = Blob(bx0, by0, bx1, by1, ink)
        blob.lead_dash = _has_leading_minus(sub[by0 - y:by1 - y, bx0 - x:bx1 - x], h)
        blobs.append(blob)
    return blobs


def _has_leading_minus(sub: np.ndarray, h: float) -> bool:
    """Left-most glyph is a short flat bar at mid height, followed by more glyphs."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    if n < 3:
        return False
    comps = sorted((stats[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 3),
                   key=lambda s: s[cv2.CC_STAT_LEFT])
    if len(comps) < 2:
        return False
    x, y, w, hh, _ = comps[0]
    nxt = comps[1]
    bh = sub.shape[0]
    cy = (y + hh / 2) / max(bh, 1)
    return (hh <= 0.35 * h and 0.3 * h <= w <= 0.9 * h and 0.3 <= cy <= 0.8
            and nxt[cv2.CC_STAT_HEIGHT] >= 0.6 * h and nxt[cv2.CC_STAT_LEFT] > x + w - 1)


def cluster_rows(blobs: list[Blob], h: float, cfg: Config = CFG) -> list[list[int]]:
    """Group blob indices into text rows by y-centre gaps."""
    if not blobs:
        return []
    order = sorted(range(len(blobs)), key=lambda i: blobs[i].cy)
    rows: list[list[int]] = [[order[0]]]
    last = blobs[order[0]].cy
    for i in order[1:]:
        cy = blobs[i].cy
        if cy - last > cfg.row_split * h:
            rows.append([i])
        else:
            rows[-1].append(i)
        last = cy
    return [sorted(r, key=lambda i: blobs[i].x0) for r in rows]


def row_center(blobs: list[Blob], row: list[int]) -> float:
    return float(np.median([blobs[i].cy for i in row]))


# ----------------------------------------------------------------------------
# rs=2 stub detection
# ----------------------------------------------------------------------------

def find_stub_row(blobs: list[Blob], rows: list[list[int]], cfg: Config = CFG) -> int | None:
    """A lone left-most blob sitting half-way between the first two header rows."""
    if len(rows) < 4:
        return None
    centers = [row_center(blobs, r) for r in rows]
    for i in (1, 2):
        if i + 1 >= len(rows) or len(rows[i]) != 1:
            continue
        b = blobs[rows[i][0]]
        left_ok = b.x1 <= min(blobs[j].x0 for j in rows[i + 1]) + 0.5 * b.w
        mid = (centers[i - 1] + centers[i + 1]) / 2
        pitch = centers[i + 1] - centers[i - 1]
        if left_ok and pitch > 0 and abs(b.cy - mid) < cfg.stub_mid_tol * pitch:
            return i
    return None


# ----------------------------------------------------------------------------
# columns
# ----------------------------------------------------------------------------

def column_gaps(blobs: list[Blob], rows: list[list[int]], h: float,
                cfg: Config = CFG) -> list[tuple[int, int]]:
    """Empty vertical corridors in the given rows -> list of (x_start, x_end) gaps."""
    idx = [i for r in rows for i in r]
    if not idx:
        return []
    xmin = min(blobs[i].x0 for i in idx)
    xmax = max(blobs[i].x1 for i in idx)
    cover = np.zeros(xmax - xmin + 1, np.int32)
    for r in rows:
        line = np.zeros_like(cover, dtype=bool)
        for i in r:
            line[blobs[i].x0 - xmin:blobs[i].x1 - xmin] = True
        cover += line
    best: list[tuple[int, int]] = []
    for frac, min_gap in ((cfg.col_cover_frac, cfg.col_min_gap),
                          (cfg.col_cover_frac, 0.6 * cfg.col_min_gap),
                          (2 * cfg.col_cover_frac, 0.6 * cfg.col_min_gap)):
        thr = int(frac * len(rows))
        low = cover <= thr
        gaps = []
        x = 0
        while x < len(low):
            if low[x]:
                s = x
                while x < len(low) and low[x]:
                    x += 1
                if s > 0 and x < len(low) and (x - s) >= min_gap * h:
                    gaps.append((s + xmin, x + xmin))
            else:
                x += 1
        if len(gaps) > cfg.max_cols - 1:
            gaps = sorted(sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)
                          [:cfg.max_cols - 1])
        best = gaps
        if len(gaps) >= cfg.min_cols - 1:
            break
    return best


def right_edge_columns(blobs: list[Blob], rows: list[list[int]], h: float,
                       cfg: Config = CFG) -> list[float] | None:
    """Numbers are right-aligned: every data column has many blobs sharing one right edge.
    Merged cells, dashes and n/a are centred and do not disturb this.  -> sorted edges."""
    pts = []                                    # (x1, row, width)
    for k, r in enumerate(rows):
        for i in r[1:]:                         # r[0] is the left-aligned row label
            pts.append((blobs[i].x1, k, blobs[i].w))
    if not pts:
        return None
    pts.sort()
    tol = cfg.edge_tol * h
    clusters: list[list[tuple[int, int, int]]] = [[pts[0]]]
    for p in pts[1:]:
        if p[0] - clusters[-1][-1][0] <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    need = max(2, int(np.ceil(cfg.edge_support * len(rows))))
    edges = []
    for c in clusters:
        support = len({p[1] for p in c})
        if support >= need and np.median([p[2] for p in c]) >= cfg.edge_min_width * h:
            edges.append(float(np.median([p[0] for p in c])))
    merged: list[float] = []
    for e in edges:
        if merged and e - merged[-1] < 2 * h:
            merged[-1] = max(merged[-1], e)
        else:
            merged.append(e)
    if not (cfg.min_cols - 1 <= len(merged) <= cfg.max_cols - 1):
        return None
    return merged


def bounds_from_edges(blobs: list[Blob], rows: list[list[int]], edges: list[float],
                      h: float) -> list[float]:
    """Cell borders from right edges: border = right edge + padding (~0.9 h)."""
    pad = 0.9 * h
    labels = [blobs[r[0]] for r in rows if r]
    first_x0 = [blobs[i].x0 for r in rows for i in r[1:] if abs(blobs[i].x1 - edges[0]) < h]
    left_of_c1 = (min(first_x0) - pad) if first_x0 else edges[0] - 5 * h
    lab_right = max((b.x1 for b in labels if b.x1 < left_of_c1 + pad), default=left_of_c1 - pad)
    b01 = (lab_right + pad + left_of_c1) / 2 if lab_right + pad < left_of_c1 else left_of_c1
    left = min(b.x0 for b in labels) - pad if labels else b01 - 6 * h
    return [float(left), float(b01)] + [float(e + pad) for e in edges]


def column_bounds(blobs: list[Blob], rows: list[list[int]], gaps: list[tuple[int, int]],
                  h: float) -> list[float]:
    """Cell borders: half-way between the text extents of neighbouring columns."""
    seps = [(a + b) / 2 for a, b in gaps]
    n = len(seps) + 1
    lo = [np.inf] * n
    hi = [-np.inf] * n
    for r in rows:
        for i in r:
            b = blobs[i]
            j = int(np.searchsorted(seps, b.cx))
            # only blobs that stay inside one column describe its extent
            left = seps[j - 1] if j > 0 else -np.inf
            right = seps[j] if j < len(seps) else np.inf
            if b.x0 >= left and b.x1 <= right:
                lo[j] = min(lo[j], b.x0)
                hi[j] = max(hi[j], b.x1)
    bounds = []
    pad = h
    for j in range(n):
        if j == 0:
            bounds.append((lo[0] if np.isfinite(lo[0]) else seps[0] - 4 * h) - pad)
        else:
            a = hi[j - 1] if np.isfinite(hi[j - 1]) else seps[j - 1]
            b = lo[j] if np.isfinite(lo[j]) else seps[j - 1]
            bounds.append((a + b) / 2 if a < b else seps[j - 1])
    bounds.append((hi[-1] if np.isfinite(hi[-1]) else seps[-1] + 4 * h) + pad)
    return [float(x) for x in bounds]


# ----------------------------------------------------------------------------
# full layout
# ----------------------------------------------------------------------------

def analyze(binary: np.ndarray, cfg: Config = CFG) -> Layout:
    """Rectified binary image -> blobs, rows, grid."""
    h = text_height(binary)
    hmask, vmask = line_masks(binary, h, cfg)
    text = remove_lines(binary, hmask, vmask)
    blobs = find_blobs(text, h, cfg)
    rows = cluster_rows(blobs, h, cfg)
    rows = _drop_noise_rows(blobs, rows, h)
    stub = find_stub_row(blobs, rows, cfg)

    real_rows = [r for k, r in enumerate(rows) if k != stub]
    body = real_rows[2:] if len(real_rows) >= 5 else real_rows

    idx = [i for r in real_rows for i in r]
    box = (min(blobs[i].x0 for i in idx), min(blobs[i].y0 for i in idx),
           max(blobs[i].x1 for i in idx), max(blobs[i].y1 for i in idx))
    rules = style_mod.detect(hmask, vmask, box, cfg)

    col_bounds, col_src = None, ""
    if rules.style == "grid":
        col_bounds = grid_bounds(rules.v_xs, box, cfg)
        col_src = "rules"
    if col_bounds is None:
        edges = right_edge_columns(blobs, body, h, cfg)
        if edges is not None:
            col_bounds = bounds_from_edges(blobs, body, edges, h)
            col_src = "edges"
    if col_bounds is None:
        gaps = column_gaps(blobs, body, h, cfg)
        if len(gaps) < cfg.min_cols - 1 and body is not real_rows:
            gaps = column_gaps(blobs, real_rows, h, cfg)
        if not gaps:
            raise ValueError("no column gaps found")
        extent_rows = real_rows[1:] if len(real_rows) > 3 else real_rows
        col_bounds = column_bounds(blobs, extent_rows, gaps, h)
        col_src = "gaps"

    centers = [row_center(blobs, r) for r in real_rows]
    pitch = float(np.median(np.diff(centers))) if len(centers) > 1 else 3 * h
    row_bounds = [centers[0] - pitch / 2]
    row_bounds += [(a + b) / 2 for a, b in zip(centers, centers[1:])]
    row_bounds.append(centers[-1] + pitch / 2)
    grid = Grid(row_bounds=row_bounds, col_bounds=col_bounds, row_centers=centers)
    return Layout(h=h, blobs=blobs, rows=rows, grid=grid, hline_mask=hmask,
                  vline_mask=vmask, stub_row=stub,
                  extra=dict(pitch=pitch, style=rules.style, rules=rules, col_src=col_src))


def grid_bounds(v_xs: list[float], box: tuple, cfg: Config = CFG) -> list[float] | None:
    """Vertical rules of a grid table are the column borders."""
    x0, _, x1, _ = box
    inner = [x for x in v_xs if x0 < x < x1]
    left = [x for x in v_xs if x <= x0]
    right = [x for x in v_xs if x >= x1]
    if not (cfg.min_cols - 1 <= len(inner) <= cfg.max_cols - 1):
        return None
    lb = left[-1] if left else x0 - 10
    rb = right[0] if right else x1 + 10
    return [float(lb)] + [float(x) for x in inner] + [float(rb)]


def _drop_noise_rows(blobs: list[Blob], rows: list[list[int]], h: float) -> list[list[int]]:
    """Drop rows that are only specks (tiny ink) - e.g. ghost text remnants above the table."""
    out = []
    for r in rows:
        ink = sum(blobs[i].ink for i in r)
        if ink < 2.0 * h * h * 0.15 and all(blobs[i].h < 0.5 * h for i in r):
            continue
        out.append(r)
    return out
