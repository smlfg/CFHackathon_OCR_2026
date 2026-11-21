"""Step 1 of the pipeline: clean a table image before reading it.

clean(img) -> (cleaned grayscale uint8 image, info dict)

1. Shadow / uneven light: estimate the page background with a large
   blur of the "ink-free" image and divide it out, so the page becomes
   evenly white (handles the T2 left-to-right shadow and T3 grey tones).
2. Tilt: try angles in [-5, 5] degrees and keep the one where the rows of
   ink line up best (sharpest horizontal projection profile).
3. Specks (T3 fax): remove tiny isolated dots that are not part of text.
"""
from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide out the page background so paper is ~255 everywhere."""
    h, w = gray.shape
    k = max(15, (min(h, w) // 40) | 1)
    # closing removes dark text/lines, leaving the paper brightness
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    out = cv2.divide(gray.astype(np.float32), bg.astype(np.float32) + 1, scale=255)
    return np.clip(out, 0, 255).astype(np.uint8)


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask (255 = ink) using Otsu on the flattened image."""
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m


def _profile_score(mask: np.ndarray, angle: float) -> float:
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    prof = rot.sum(axis=1, dtype=np.float64)
    return float(np.square(np.diff(prof)).sum())


def estimate_skew(mask: np.ndarray, max_angle: float = 8.0) -> float:
    """Angle (degrees, opencv convention) that best straightens the rows."""
    scale = 800 / max(mask.shape)
    small = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1 else mask
    best = 0.0
    # coarse then fine search; on ties prefer the angle closest to 0
    for step, span in ((0.5, max_angle), (0.1, 0.6), (0.02, 0.12)):
        cands = np.arange(best - span, best + span + 1e-9, step)
        cands = cands[np.abs(cands) <= max_angle]
        scores = np.array([_profile_score(small, a) for a in cands])
        top = cands[scores >= scores.max() * 0.999]
        best = float(top[np.argmin(np.abs(top))])
    return round(best, 2)


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderValue=255)


def remove_specks(gray: np.ndarray, max_area: int = 6) -> np.ndarray:
    """Whiten tiny connected ink blobs (fax noise) that are far from text."""
    mask = ink_mask(gray)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] <= max_area
    if not small.any():
        return gray
    # keep small blobs that sit next to real text (dots, commas, decimal points)
    big = np.isin(labels, np.flatnonzero(~small)[1:]).astype(np.uint8) * 255
    near_text = cv2.dilate(big, np.ones((9, 9), np.uint8)) > 0
    kill = small[labels] & ~near_text
    out = gray.copy()
    out[kill] = 255
    return out


def clean(img: np.ndarray) -> tuple[np.ndarray, dict]:
    gray = remove_specks(flatten_illumination(to_gray(img)))
    angle = estimate_skew(ink_mask(gray))
    gray = rotate(gray, angle)
    return gray, {"skew_deg": angle}


if __name__ == "__main__":
    import sys
    src, dst = sys.argv[1], sys.argv[2]
    out, info = clean(cv2.imread(src))
    cv2.imwrite(dst, out)
    print(info)
