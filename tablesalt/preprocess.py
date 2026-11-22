"""Step 1 - preprocess: illumination, binarization, deskew, perspective.

1. Divide by an estimated background (morphological closing) -> shadows (T2) and the
   grey header / zebra fills disappear, bleed-through (T3) gets lighter.
2. Otsu threshold (clamped) -> ink mask; tiny specks removed.
3. Deskew: projection-profile search (rows of text give sharp peaks at the right angle).
4. Perspective: the generator left-aligns row labels and right-aligns numbers, so the left
   and right text edges are straight lines in the original table, as are the first and the
   last text row.  Their four intersections give a homography to a rectangle - for all three
   styles, with or without rules.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import structure as st
from .config import CFG, Config
from .types import Prepared

WHITE = 255


def normalize_illumination(gray: np.ndarray, cfg: Config = CFG) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.bg_kernel, cfg.bg_kernel))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    bg = cv2.blur(bg, (cfg.bg_kernel, cfg.bg_kernel))
    bg = np.maximum(bg.astype(np.float32), 1.0)
    norm = gray.astype(np.float32) / bg * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def binarize(norm: np.ndarray, cfg: Config = CFG) -> np.ndarray:
    blur = cv2.GaussianBlur(norm, (3, 3), 0)
    t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    t = float(np.clip(t, cfg.bin_thresh_min, cfg.bin_thresh_max))
    binary = np.where(blur < t, 255, 0).astype(np.uint8)
    return despeckle(binary, cfg.speck_area)


def despeckle(binary: np.ndarray, max_area: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] <= max_area
    small[0] = False
    out = binary.copy()
    out[small[lab]] = 0
    return out


# ----------------------------------------------------------------------------
# deskew
# ----------------------------------------------------------------------------


def _profile_score(small: np.ndarray, angle: float) -> float:
    hh, ww = small.shape
    m = cv2.getRotationMatrix2D((ww / 2, hh / 2), angle, 1.0)
    rot = cv2.warpAffine(small, m, (ww, hh), flags=cv2.INTER_NEAREST, borderValue=0)
    prof = rot.sum(axis=1, dtype=np.float64)
    return float(np.var(prof))


def estimate_skew(binary: np.ndarray, cfg: Config = CFG) -> float:
    """Angle (deg, OpenCV convention) that makes text rows horizontal."""
    scale = 0.5
    small = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = (small > 64).astype(np.uint8)
    best, best_s = 0.0, -1.0
    a = -cfg.deskew_max_deg
    while a <= cfg.deskew_max_deg + 1e-9:
        s = _profile_score(small, a)
        if s > best_s:
            best, best_s = a, s
        a += cfg.deskew_coarse_step
    lo, hi = best - cfg.deskew_coarse_step, best + cfg.deskew_coarse_step
    a = lo
    while a <= hi + 1e-9:
        s = _profile_score(small, a)
        if s > best_s:
            best, best_s = a, s
        a += cfg.deskew_fine_step
    return round(best, 3)


def rotation_matrix(shape: tuple[int, int], angle: float) -> tuple[np.ndarray, tuple[int, int]]:
    """3x3 rotation about the centre with canvas expansion (no clipping)."""
    hh, ww = shape
    m = cv2.getRotationMatrix2D((ww / 2, hh / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(hh * sin + ww * cos + 0.5), int(hh * cos + ww * sin + 0.5)
    m[0, 2] += nw / 2 - ww / 2
    m[1, 2] += nh / 2 - hh / 2
    return np.vstack([m, [0, 0, 1]]), (nw, nh)


# ----------------------------------------------------------------------------
# perspective from text geometry
# ----------------------------------------------------------------------------


def _robust_line(u: np.ndarray, v: np.ndarray, tol: float, min_inliers: int):
    """Fit v = a*u + b robustly (all point pairs as RANSAC hypotheses). None if weak."""
    n = len(u)
    if n < 2:
        return None
    best_in = None
    for i in range(n):
        for j in range(i + 1, n):
            if abs(u[j] - u[i]) < 1e-6:
                continue
            a = (v[j] - v[i]) / (u[j] - u[i])
            b = v[i] - a * u[i]
            inl = np.abs(v - (a * u + b)) <= tol
            if best_in is None or inl.sum() > best_in.sum():
                best_in = inl
    if best_in is None or best_in.sum() < min_inliers:
        return None
    a, b = np.polyfit(u[best_in], v[best_in], 1)
    return float(a), float(b)


def _aligned_right_edge(blobs, rows, tol: float, need: int):
    """Right-most straight line x = a*y + b that many rows' blobs end on (right-aligned
    numbers of one column).  Any column will do for the homography; the right-most one
    conditions it best.  None if no edge has `need` supporting rows."""
    pts = [(st.row_center(blobs, r), blobs[i].x1, k) for k, r in enumerate(rows) for i in r[1:]]
    if len(pts) < 2:
        return None
    y = np.array([p[0] for p in pts])
    x = np.array([p[1] for p in pts], float)
    rid = np.array([p[2] for p in pts])
    n_rows = len(rows)
    best = None  # ((support, mean x), inlier mask)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if abs(rid[j] - rid[i]) < max(2, n_rows // 3):
                continue
            a = (x[j] - x[i]) / (y[j] - y[i])
            if abs(a) > 0.15:  # > ~8.5 deg off vertical: not an edge
                continue
            b = x[i] - a * y[i]
            inl = np.abs(x - (a * y + b)) <= tol
            support = len(set(rid[inl].tolist()))
            if support < need:
                continue
            key = (support, float(x[inl].mean()))
            if best is None or key > best[0]:
                best = (key, inl)
    if best is None:
        return None
    inl = best[1]
    a, b = np.polyfit(y[inl], x[inl], 1)
    return float(a), float(b)


def text_quad(binary: np.ndarray, cfg: Config = CFG):
    """Source quadrilateral (TL, TR, BR, BL) of the table text, or None."""
    h = st.text_height(binary)
    hm, vm = st.line_masks(binary, h, cfg)
    text = st.remove_lines(binary, hm, vm)
    blobs = st.find_blobs(text, h, cfg)
    rows = st.cluster_rows(blobs, h, cfg)
    rows = [r for r in rows if len(r) >= 2]
    if len(rows) < cfg.rectify_min_rows:
        return None
    tol = cfg.rectify_inlier_px * h
    # header rows are centred, never left/right-aligned: keep them out of the edge fits
    body = rows[2:] if len(rows) >= cfg.rectify_min_rows + 2 else rows
    ys = np.array([st.row_center(blobs, r) for r in body])
    lx = np.array([blobs[r[0]].x0 for r in body], float)
    need = max(4, len(body) // 2)
    left = _robust_line(ys, lx, tol, need)
    right = _aligned_right_edge(blobs, body, tol, need)
    full = [r for r in rows if len(r) >= 3] or rows
    top = _robust_line(
        np.array([blobs[i].cx for i in full[0]]), np.array([blobs[i].cy for i in full[0]]), tol, 2
    )
    bot = _robust_line(
        np.array([blobs[i].cx for i in full[-1]]), np.array([blobs[i].cy for i in full[-1]]), tol, 2
    )
    if None in (left, right, top, bot):
        return None

    def cross(hline, vline):  # y = a x + b  and  x = c y + d
        a, b = hline
        c, d = vline
        y = (a * d + b) / (1 - a * c)
        return np.array([c * y + d, y])

    quad = np.array(
        [cross(top, left), cross(top, right), cross(bot, right), cross(bot, left)], np.float32
    )
    return quad, h


def rectify_homography(binary: np.ndarray, cfg: Config = CFG):
    """3x3 homography (and output size) that maps the text quad to a rectangle, or None."""
    res = text_quad(binary, cfg)
    if res is None:
        return None
    quad, h = res
    tl, tr, br, bl = quad
    wd = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    hd = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if wd < 5 * h or hd < 3 * h:
        return None
    # sanity: a mild distortion only (generator: <= 6 % corner jitter)
    for p, q in ((tr - tl, br - bl), (bl - tl, br - tr)):
        ang = np.degrees(np.arctan2(p[1], p[0]) - np.arctan2(q[1], q[0]))
        if abs((ang + 180) % 360 - 180) > 12:
            return None
    dst = np.array([tl, tl + [wd, 0], tl + [wd, hd], tl + [0, hd]], np.float32)
    H = cv2.getPerspectiveTransform(quad, dst)
    return H


def _fit_canvas(M: np.ndarray, shape: tuple[int, int]):
    """Shift M so the whole warped image lands on a canvas starting at (0, 0)."""
    hh, ww = shape
    corners = np.array([[0, 0], [ww, 0], [ww, hh], [0, hh]], np.float32).reshape(-1, 1, 2)
    wc = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
    x0, y0 = wc.min(axis=0)
    x1, y1 = wc.max(axis=0)
    if x1 - x0 > 3 * ww or y1 - y0 > 3 * hh:
        return None
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    return T @ M, (int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0)))


def prepare(gray: np.ndarray, cfg: Config = CFG) -> Prepared:
    """Raw grayscale -> rectified, normalized gray + binary."""
    norm = normalize_illumination(gray, cfg)
    binary = binarize(norm, cfg)
    angle = estimate_skew(binary, cfg)
    R, size = rotation_matrix(gray.shape, angle)
    rot = cv2.warpPerspective(norm, R, size, flags=cv2.INTER_CUBIC, borderValue=WHITE)
    rot_bin = binarize(rot, cfg)
    H = rectify_homography(rot_bin, cfg)
    fit = None if H is None else _fit_canvas(H @ R, gray.shape)
    if fit is None:
        return Prepared(gray=rot, binary=rot_bin, angle=angle, homography=False)
    M, size2 = fit
    out = cv2.warpPerspective(norm, M, size2, flags=cv2.INTER_CUBIC, borderValue=WHITE)
    return Prepared(gray=out, binary=binarize(out, cfg), angle=angle, homography=True)
