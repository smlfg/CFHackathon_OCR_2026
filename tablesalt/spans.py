"""Step 4 - spans: header rows, the rs=2 stub, grouped header spans, merged body cells.

Generator conventions (docs/NOTES.md):
* 1 or 2 header rows.  With 2, cell (0,0) has rs=2 and row 0 holds group labels that
  partition columns 1..n-1 into consecutive runs of 1-3 columns, each label centred.
* Body merges are always horizontal, cs=2, starting at column >= 1, text one of
  "n/a", "—" (em dash), "see note 3", centred over both columns.
* "–" (en dash), "n/a" and "" also occur as single cells.
"""

from __future__ import annotations

import numpy as np

from . import lexicon as lx
from .config import CFG, Config
from .types import Blob, Cell, Layout


def col_range(b: Blob, bounds: list[float], cross: float) -> tuple[int, int]:
    """Columns a blob occupies: the one holding its centre, plus neighbours it clearly
    reaches into."""
    n = len(bounds) - 1
    j = int(np.clip(np.searchsorted(bounds, b.cx) - 1, 0, n - 1))
    lo = hi = j
    while lo > 0 and b.x0 < bounds[lo] - cross:
        lo -= 1
    while hi < n - 1 and b.x1 > bounds[hi + 1] + cross:
        hi += 1
    return lo, hi


def classify_dash(b: Blob, h: float, cfg: Config = CFG) -> str | None:
    """A flat short blob is a dash: wide = em dash (merged cell), narrow = en dash."""
    if b.h <= 0.4 * h and b.w <= 2.2 * h:
        return lx.EM_DASH if b.w >= cfg.em_dash_min_w * h else lx.EN_DASH
    return None


def _merge_kind(text: str) -> str | None:
    t = lx.norm(text)
    if not t:
        return None
    if lx.score(t, "see note 3") >= 70 or ("note" in t and len(t) <= 14):
        return "see note 3"
    # body text without digits that is not a dash can only be "n/a" (OCR: "Wa", "nia", ...)
    if not any(ch.isdigit() for ch in t) and len(t) <= 5 and not all(ch in "-–—_~=" for ch in t):
        return "n/a"
    return None


def group_partition(
    centers: list[float],
    extents: list[tuple[float, float]],
    bounds: list[float],
    first: int,
    last: int,
    max_span: int = 3,
) -> list[tuple[int, int]] | None:
    """Split columns first..last into len(centers) consecutive runs (1..max_span wide) so
    every label sits centred over its run.  Dynamic programming; None if impossible."""
    k = len(centers)
    cols = list(range(first, last + 1))
    m = len(cols)
    if k == 0 or k > m:
        return None
    INF = float("inf")
    # cost[g][e] = best cost placing labels 0..g-1 over the first e columns
    cost = [[INF] * (m + 1) for _ in range(k + 1)]
    back = [[-1] * (m + 1) for _ in range(k + 1)]
    cost[0][0] = 0.0
    for g in range(1, k + 1):
        for e in range(1, m + 1):
            for s in range(1, max_span + 1):
                b = e - s
                if b < 0 or cost[g - 1][b] == INF:
                    continue
                x0, x1 = bounds[cols[b]], bounds[cols[e - 1] + 1]
                c = abs(centers[g - 1] - (x0 + x1) / 2)
                ex0, ex1 = extents[g - 1]
                c += 3 * max(0.0, x0 - ex0) + 3 * max(0.0, ex1 - x1)  # must fit inside
                if cost[g - 1][b] + c < cost[g][e]:
                    cost[g][e] = cost[g - 1][b] + c
                    back[g][e] = b
    if cost[k][m] == INF:
        return None if max_span >= m else group_partition(centers, extents, bounds, first, last, m)
    runs, e = [], m
    for g in range(k, 0, -1):
        b = back[g][e]
        runs.append((cols[b], cols[e - 1]))
        e = b
    return runs[::-1]


def header_rows(
    lay: Layout, real_rows: list[list[int]], ranges: dict[int, tuple[int, int]], texts: list[str]
) -> int:
    """Decide whether the table has one or two header rows."""
    if lay.stub_row is not None:
        return 2
    if len(real_rows) < 3:
        return 1
    top = [i for i in real_rows[0] if ranges[i][1] >= 1]
    if any(ranges[i][1] > ranges[i][0] and ranges[i][0] >= 1 for i in top):
        return 2
    rules = lay.extra.get("rules")
    if rules is not None and rules.style == "hlines" and len(rules.h_ys) >= 3:
        above = sum(1 for c in lay.grid.row_centers if c < rules.h_ys[1])
        if above in (1, 2):
            return above
    tt = [texts[i] for i in top if lx.norm(texts[i])]
    if tt:
        g = np.mean([lx.best(t, lx.GROUP_NAMES)[1] for t in tt])
        c = np.mean([lx.best(t, [x for p in lx.COL_POOLS for x in p])[1] for t in tt])
        if g > c and g >= 60:
            return 2
    return 1


def build_cells(lay: Layout, texts: list[str], cfg: Config = CFG) -> tuple[list[Cell], int]:
    """Blobs + OCR texts -> logical cells covering the whole grid, and the header row count."""
    h = lay.h
    bounds = lay.grid.col_bounds
    n_cols = len(bounds) - 1
    cross = cfg.cross_px * h
    blobs = lay.blobs
    texts = list(texts)
    for i, b in enumerate(blobs):  # dashes: shape beats OCR
        d = classify_dash(b, h, cfg)
        if d is not None:
            texts[i] = d
    ranges = {i: col_range(b, bounds, cross) for i, b in enumerate(blobs)}
    real_rows = [r for k, r in enumerate(lay.rows) if k != lay.stub_row]
    n_rows = len(real_rows)
    hr = header_rows(lay, real_rows, ranges, texts)

    taken = [[False] * n_cols for _ in range(n_rows)]
    cells: list[Cell] = []

    def add(r, c, rs, cs, ids, header=False):
        for dr in range(rs):
            for dc in range(cs):
                taken[r + dr][c + dc] = True
        txt = " ".join(texts[i] for i in sorted(ids, key=lambda i: blobs[i].x0) if texts[i]).strip()
        cells.append(Cell(text=txt, r=r, c=c, rs=rs, cs=cs, blobs=list(ids), header=header))

    start = 0
    if hr == 2 and n_rows >= 3:
        stub_ids = list(lay.rows[lay.stub_row]) if lay.stub_row is not None else []
        stub_ids += [i for i in real_rows[0] + real_rows[1] if ranges[i][1] == 0]
        add(0, 0, 2, 1, stub_ids, header=True)
        top = sorted((i for i in real_rows[0] if ranges[i][1] >= 1), key=lambda i: blobs[i].cx)
        runs = (
            group_partition(
                [blobs[i].cx for i in top],
                [(blobs[i].x0, blobs[i].x1) for i in top],
                bounds,
                1,
                n_cols - 1,
            )
            if top
            else None
        )
        if runs is not None:
            for i, (a, b) in zip(top, runs, strict=False):
                add(0, a, 1, b - a + 1, [i], header=True)
        for c in range(1, n_cols):
            if not taken[0][c]:
                add(
                    0,
                    c,
                    1,
                    1,
                    [
                        i
                        for i in top
                        if ranges[i][0] <= c <= ranges[i][1] and runs is None and ranges[i][0] == c
                    ],
                    header=True,
                )
        for c in range(1, n_cols):
            add(
                1,
                c,
                1,
                1,
                [i for i in real_rows[1] if _home(ranges[i], blobs[i], bounds) == c],
                header=True,
            )
        start = 2
    elif n_rows >= 1:
        for c in range(n_cols):
            add(
                0,
                c,
                1,
                1,
                [i for i in real_rows[0] if _home(ranges[i], blobs[i], bounds) == c],
                header=True,
            )
        start = 1

    for r in range(start, n_rows):
        row = real_rows[r]
        per_col: dict[int, list[int]] = {}
        merges: list[tuple[int, int, int]] = []  # (blob, c, cs)
        for i in row:
            lo, hi = ranges[i]
            if lo == 0:
                per_col.setdefault(0, []).append(i)
                continue
            kind = None
            if hi > lo:
                kind = "geom"
            elif texts[i] == lx.EM_DASH or _merge_kind(texts[i]) == "see note 3":
                kind = "text"
            elif _merge_kind(texts[i]) == "n/a":
                kind = "maybe"
            if kind is not None:
                pair = _best_pair(blobs[i], lo, hi, bounds, n_cols, kind)
                if pair is not None:
                    merges.append((i, pair, 2 if kind != "geom" else max(2, hi - lo + 1)))
                    continue
            per_col.setdefault(_home((lo, hi), blobs[i], bounds), []).append(i)
        for i, c, cs in merges:
            cs = min(cs, n_cols - c)
            if any(taken[r][c + d] for d in range(cs)) or any(
                per_col.get(c + d) for d in range(cs)
            ):
                per_col.setdefault(_home(ranges[i], blobs[i], bounds), []).append(i)
                continue
            if texts[i] == lx.EN_DASH:
                texts[i] = lx.EM_DASH
            add(r, c, 1, cs, [i])
        for c in range(n_cols):
            if not taken[r][c]:
                add(r, c, 1, 1, per_col.get(c, []))
    return cells, hr


def _home(rng: tuple[int, int], b: Blob, bounds: list[float]) -> int:
    lo, hi = rng
    if lo == hi:
        return lo
    return int(np.clip(np.searchsorted(bounds, b.cx) - 1, lo, hi))


def _best_pair(
    b: Blob, lo: int, hi: int, bounds: list[float], n_cols: int, kind: str
) -> int | None:
    """Left column of the 2-wide merge that centres this blob best (None = single cell)."""
    if kind == "geom" and hi > lo:
        return lo if lo >= 1 else None
    cands = [c for c in (lo - 1, lo) if c >= 1 and c + 1 <= n_cols - 1]
    if not cands:
        return None
    dist = {c: abs(b.cx - (bounds[c] + bounds[c + 2]) / 2) for c in cands}
    c = min(dist, key=dist.get)
    if kind == "maybe":
        single = abs(b.cx - (bounds[lo] + bounds[lo + 1]) / 2)
        if dist[c] >= single:
            return None
    return c
