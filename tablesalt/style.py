"""Step 2 - style: grid / hlines / borderless, decided from the image's rules only.

grid       inner vertical rules exist (the generator draws every cell border)
hlines     long horizontal rules but no vertical ones (top, under the header, bottom)
borderless no rules at all
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import CFG, Config


@dataclass
class Rules:
    style: str
    v_xs: list[float] = field(default_factory=list)  # x of vertical rules, left to right
    h_ys: list[float] = field(default_factory=list)  # y of horizontal rules, top to bottom


def _peaks(profile: np.ndarray, thr: float) -> list[float]:
    """Centres of runs where profile >= thr."""
    out, x, n = [], 0, len(profile)
    while x < n:
        if profile[x] >= thr:
            s = x
            while x < n and profile[x] >= thr:
                x += 1
            seg = profile[s:x]
            out.append(float(s + np.average(np.arange(len(seg)), weights=seg)))
        else:
            x += 1
    merged: list[float] = []
    for p in out:  # double-drawn / thick rules -> one
        if merged and p - merged[-1] < 6:
            merged[-1] = (merged[-1] + p) / 2
        else:
            merged.append(p)
    return merged


def detect(
    hmask: np.ndarray,
    vmask: np.ndarray,
    text_box: tuple[float, float, float, float],
    cfg: Config = CFG,
) -> Rules:
    """text_box = (x0, y0, x1, y1) of all text; rules are measured against its size."""
    x0, y0, x1, y1 = text_box
    tw, th = max(1.0, x1 - x0), max(1.0, y1 - y0)
    hprof = (hmask > 0).sum(axis=1).astype(float)
    vprof = (vmask > 0).sum(axis=0).astype(float)
    h_ys = _peaks(hprof, 0.5 * tw)
    v_xs = _peaks(vprof, cfg.vline_min_cover * th)
    inner = [x for x in v_xs if x0 < x < x1]  # outer borders lie outside the text
    if len(v_xs) >= cfg.vline_min_count + 1 and len(inner) >= cfg.vline_min_count:
        style = "grid"
    elif len(h_ys) >= 2:
        style = "hlines"
    else:
        style = "borderless"
    return Rules(style=style, v_xs=v_xs, h_ys=h_ys)
