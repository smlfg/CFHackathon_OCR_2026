"""Classic (no AI) image enhancement for table scans — runs before any OCR.

    from src.enhance import enhance
    out, info = enhance(bgr_or_gray_image)     # grayscale uint8, about 2x larger

Command line:
    python -m src.enhance input.jpg output.png
    python -m src.enhance data/dev out_dir/          (whole dataset, prints timing)

Steps (parameters come from tools/image_stats.py measurements, see docs/ENHANCEMENT.md):
  1. diagnose   decide the damage type from the image itself:
                  fax    > 50 % pure black/white pixels   (T3-like)
                  photo  shadow ratio < 0.92 or contrast < 160  (T2-like)
                  scan   JPEG blockiness > 1.10            (T1-like)
                  clean  otherwise                         (T0-like)
  2. despeckle  fax only: remove isolated dots far from text (keeps , and .)
  3. light      divide out uneven lighting / shadow (photo, scan)
  4. denoise    photo only: light non-local-means (JPEG blocks + sensor noise)
  5. contrast   stretch so paper -> 255 and ink -> ~0 (photo, scan)
  6. deskew     rotation search up to +-8 deg
  7. keystone   perspective correction from the table's edges and row lines
  8. crop       cut to the table plus a margin
  9. upscale    2x (text 15 px -> 30 px), then unsharp mask (photo strong, scan mild)
Option no_lines=True erases table rules for text-only OCR (Tesseract).
Output stays grayscale (no hard black/white): vision models read grey better.
Typical cost: well under 1 s per table on one CPU core.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .preprocess import estimate_skew, rotate, remove_specks

SCALE = 2.0


# ----------------------------------------------------------------- diagnosis
def _paper_ink(g: np.ndarray):
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    near = cv2.dilate(m, np.ones((9, 9), np.uint8)) > 0
    core = cv2.erode(m, np.ones((2, 2), np.uint8)) > 0
    paper = g[~near] if (~near).any() else g.ravel()
    ink = g[core] if core.any() else g[m > 0]
    return near, float(np.median(paper)), float(np.median(ink)) if ink.size else 0.0


def diagnose(g: np.ndarray) -> dict:
    h, w = g.shape
    binarized = float(((g <= 3) | (g >= 252)).mean())
    near, paper, ink = _paper_ink(g)
    third = w // 3
    pl = g[:, :third][~near[:, :third]]
    pr = g[:, -third:][~near[:, -third:]]
    shade = float(min(np.median(pl), np.median(pr)) / max(1.0, max(np.median(pl), np.median(pr)))) \
        if pl.size and pr.size else 1.0
    d = np.abs(np.diff(g.astype(np.float32), axis=1))
    block = float(d[:, 7::8].mean() / max(np.delete(d, np.s_[7::8], axis=1).mean(), 1e-6))
    contrast = paper - ink
    if binarized > 0.5:
        kind = "fax"
    elif shade < 0.92 or contrast < 160:
        kind = "photo"
    elif block > 1.10:
        kind = "scan"
    else:
        kind = "clean"
    return dict(kind=kind, binarized=round(binarized, 3), shade=round(shade, 3),
                contrast=round(contrast, 1), blockiness=round(block, 3))


# ----------------------------------------------------------------- steps
def flatten_light(g: np.ndarray) -> np.ndarray:
    k = max(15, (min(g.shape) // 30) | 1)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    out = cv2.divide(g.astype(np.float32), bg.astype(np.float32) + 1, scale=255)
    return np.clip(out, 0, 255).astype(np.uint8)


def stretch_contrast(g: np.ndarray) -> np.ndarray:
    _, paper, ink = _paper_ink(g)
    lo = max(0.0, ink - 10)
    hi = max(lo + 30, paper - 4)
    out = (g.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def ink_mask(g: np.ndarray) -> np.ndarray:
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m


def _row_bands(mask: np.ndarray):
    prof = (mask > 0).sum(axis=1)
    on = prof > max(3, 0.02 * mask.shape[1])
    bands, start = [], None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        elif not v and start is not None:
            if y - start >= 6:
                bands.append((start, y))
            start = None
    if start is not None and len(on) - start >= 6:
        bands.append((start, len(on)))
    return bands


def _robust_fit(x, y):
    """Least-squares line y = a*x + b with one round of outlier removal."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return None
    a, b = np.polyfit(x, y, 1)
    r = np.abs(y - (a * x + b))
    keep = r <= max(2.0, 2.5 * np.median(r))
    if keep.sum() >= 2:
        a, b = np.polyfit(x[keep], y[keep], 1)
    return a, b


def correct_keystone(g: np.ndarray) -> tuple[np.ndarray, bool]:
    """Perspective fix: fit the table's left/right edges (from each text row's
    first and last ink) and the top/bottom rows' slopes, then warp that
    quadrilateral back to a rectangle. Skipped when the fit is unreliable."""
    m = ink_mask(g)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    bands = _row_bands(m)
    if len(bands) < 4:
        return g, False
    ys, lefts, rights, slopes = [], [], [], []
    for y0, y1 in bands:
        yy, xx = np.nonzero(m[y0:y1])
        if xx.size < 30:
            continue
        ys.append((y0 + y1) / 2)
        lefts.append(np.percentile(xx, 0.5))
        rights.append(np.percentile(xx, 99.5))
        slopes.append(np.polyfit(xx, yy + y0, 1)[0])
    if len(ys) < 4:
        return g, False
    left = _robust_fit(ys, lefts)       # x = a*y + b
    right = _robust_fit(ys, rights)
    top_slope, bot_slope = np.median(slopes[:2]), np.median(slopes[-2:])
    if left is None or right is None:
        return g, False
    # ignore tiny distortions; refuse implausible ones
    spread = max(abs(left[0]), abs(right[0]), abs(top_slope), abs(bot_slope))
    if spread < 0.008 or spread > 0.15 or abs(top_slope - bot_slope) > 0.1:
        return g, False
    yt, yb = ys[0], ys[-1]
    xl_t, xl_b = left[0] * yt + left[1], left[0] * yb + left[1]
    xr_t, xr_b = right[0] * yt + right[1], right[0] * yb + right[1]
    xm_t, xm_b = (xl_t + xr_t) / 2, (xl_b + xr_b) / 2
    src = np.float32([
        [xl_t, yt + top_slope * (xl_t - xm_t)], [xr_t, yt + top_slope * (xr_t - xm_t)],
        [xr_b, yb + bot_slope * (xr_b - xm_b)], [xl_b, yb + bot_slope * (xl_b - xm_b)]])
    width = ((xr_t - xl_t) + (xr_b - xl_b)) / 2
    height = yb - yt
    if width < 50 or height < 50:
        return g, False
    x0, y0 = float(np.mean([xl_t, xl_b])), float(yt)
    dst = np.float32([[x0, y0], [x0 + width, y0], [x0 + width, y0 + height], [x0, y0 + height]])
    H = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(g, H, (g.shape[1], g.shape[0]), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return out, True


def crop_to_content(g: np.ndarray, margin: int = 20) -> np.ndarray:
    m = ink_mask(g)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = st[1:, cv2.CC_STAT_AREA] >= 8
    if not keep.any():
        return g
    xs0 = st[1:, 0][keep]; ys0 = st[1:, 1][keep]
    xs1 = xs0 + st[1:, 2][keep]; ys1 = ys0 + st[1:, 3][keep]
    x0, y0 = max(0, int(np.percentile(xs0, 0.5)) - margin), max(0, int(np.percentile(ys0, 0.5)) - margin)
    x1 = min(g.shape[1], int(np.percentile(xs1, 99.5)) + margin)
    y1 = min(g.shape[0], int(np.percentile(ys1, 99.5)) + margin)
    if x1 - x0 < 50 or y1 - y0 < 50:
        return g
    return g[y0:y1, x0:x1]


def _thin_lines(mask, kernel, horizontal, max_thick):
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n, lab, st, _ = cv2.connectedComponentsWithStats(opened, 8)
    thick = st[1:, cv2.CC_STAT_HEIGHT] if horizontal else st[1:, cv2.CC_STAT_WIDTH]
    keep = np.zeros(n, bool)
    keep[1:] = thick <= max_thick          # filled bands (header shading, zebra) are thick
    return (keep[lab] * 255).astype(np.uint8), int(keep.sum())


def remove_lines(g: np.ndarray, only_if_grid: bool = True) -> np.ndarray:
    """Erase thin table rules so text-only OCR engines (Tesseract) don't read
    them as | or _. By default only on grid tables (>= 2 vertical rules):
    tests showed erasing on other styles costs more than it gains.
    Use this on the copy you OCR; keep the lines for structure detection."""
    _, m = cv2.threshold(g, 160, 255, cv2.THRESH_BINARY_INV)   # dark strokes only, not grey fills
    h, w = g.shape
    t = max(4, int(round(3 * h / 600)))
    hl, _ = _thin_lines(m, cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 20), 1)), True, t * 2)
    vl, nv = _thin_lines(m, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 12))), False, t * 2)
    if only_if_grid and nv < 2:
        return g
    lines = cv2.dilate(hl | vl, np.ones((3, 3), np.uint8))
    out = g.copy()
    out[lines > 0] = 255
    return out


def unsharp(g: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    if amount <= 0:
        return g
    blur = cv2.GaussianBlur(g, (0, 0), sigma)
    return cv2.addWeighted(g, 1 + amount, blur, -amount, 0)


# ----------------------------------------------------------------- main entry
SETTINGS = {  # per damage type: (light, denoise_h, contrast, sharpen_amount)
    "clean": (False, 0, False, 0.0),
    "scan":  (True, 0, True, 0.5),
    "photo": (True, 6, True, 1.0),
    "fax":   (False, 0, False, 0.0),
}


def enhance(img: np.ndarray, scale: float = SCALE, no_lines: bool = False) -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    info = diagnose(g)
    light, den_h, contrast, sharp = SETTINGS[info["kind"]]

    if info["kind"] == "fax":
        g = remove_specks(g, max_area=6)
    if light:
        g = flatten_light(g)
    if den_h:
        g = cv2.fastNlMeansDenoising(g, None, h=den_h, templateWindowSize=7, searchWindowSize=15)
    if contrast:
        g = stretch_contrast(g)

    angle = estimate_skew(ink_mask(g))
    if abs(angle) >= 0.05:
        g = rotate(g, angle)
    keyed = False
    if info["kind"] in ("photo", "fax"):   # only camera / fax pages have perspective
        g, keyed = correct_keystone(g)
    g = crop_to_content(g)

    if scale != 1:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    g = unsharp(g, sigma=1.0 * scale, amount=sharp)
    if no_lines:
        g = remove_lines(g)

    info.update(skew_deg=angle, keystone=keyed, size=list(g.shape[::-1]),
                ms=round((time.perf_counter() - t0) * 1000))
    return g, info


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
        times = []
        for e in manifest:
            out, info = enhance(cv2.imread(str(src / e["id"] / Path(e["image"]).name)))
            cv2.imwrite(str(dst / f"{e['id']}.png"), out)
            times.append(info["ms"])
            print(e["id"], e["tier"], info)
        print(f"\n{len(times)} images, mean {np.mean(times):.0f} ms, max {max(times)} ms per image")
    else:
        out, info = enhance(cv2.imread(str(src)))
        cv2.imwrite(str(dst), out)
        print(info)
