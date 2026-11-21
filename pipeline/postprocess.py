"""Principled post-processing passes for HTML table predictions.

All passes are general structural/format normalizations or pixel-geometry
cross-checks (no hardcoded dataset vocabulary).
"""
from __future__ import annotations

import html as htmllib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lxml import html as lxml_html
from PIL import Image, ImageFilter

_DASH_RE = re.compile(r"^[-–—―−]+$")
_PLACEHOLDER_RE = re.compile(r"^(?:[-–—―−]+|n/a|see note \d+)$", re.I)
_UNIT_RE = re.compile(r"^(.*?)\s*(%|€|kg|ms)$")
_STRICT_NUM_RE = re.compile(r"^\(?-?\d+(?:[., ]\d+)*\)?(?:\s*(?:%|€|kg|ms))?$")

_INT_SPACE_RE = re.compile(r"^\d{1,3}(?: \d{3})+$")
_INT_COMMA_RE = re.compile(r"^\d{1,3}(?:,\d{3})+$")
_DE_DEC_RE = re.compile(r"^(?:\d{1,3}(?:\.\d{3})+|\d{1,3}),\d{2}$")
_EN_DEC_RE = re.compile(r"^(?:\d{1,3}(?:,\d{3})+|\d{1,3})\.\d{2}$")
_BARE_INT_4_6_RE = re.compile(r"^\d{4,6}$")
_THREE_GROUP_DOT_COMMA_RE = re.compile(r"^\d{1,3}[.,]\d{3}$")
_THREE_GROUP_DOT_SPACE_RE = re.compile(r"^\d{1,3}[. ]\d{3}$")
_DE_MISSING_DOT_RE = re.compile(r"^\d{4,6},\d{2}$")
_EN_MISSING_COMMA_RE = re.compile(r"^\d{4,6}\.\d{2}$")


@dataclass
class Cell:
    r: int
    c: int
    rs: int
    cs: int
    text: str
    tag: str = "td"


def _span_val(v) -> int:
    m = re.match(r"\s*(\d+)", str(v if v is not None else ""))
    return min(max(int(m.group(1)), 1), 100) if m else 1


def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").replace("\u00a0", " ")).strip()


def parse_table(html_str: str) -> tuple[list[Cell], int, int]:
    root = lxml_html.fromstring(html_str)
    tables = root.xpath("//table")
    if not tables:
        return [], 0, 0
    occ: set[tuple[int, int]] = set()
    cells: list[Cell] = []
    for r, tr in enumerate(tables[0].xpath(".//tr")):
        c = 0
        for td in tr.xpath("./td|./th"):
            while (r, c) in occ:
                c += 1
            rs = _span_val(td.get("rowspan", 1))
            cs = _span_val(td.get("colspan", 1))
            txt = _norm_text("".join(td.itertext()))
            tag = td.tag.lower() if isinstance(td.tag, str) else "td"
            cells.append(Cell(r=r, c=c, rs=rs, cs=cs, text=txt, tag=tag))
            for dr in range(rs):
                for dc in range(cs):
                    occ.add((r + dr, c + dc))
            c += cs
    if not cells:
        return [], 0, 0
    nrows = max(cell.r + cell.rs for cell in cells)
    ncols = max(cell.c + cell.cs for cell in cells)
    return cells, nrows, ncols


def render_table(cells: list[Cell]) -> str:
    if not cells:
        return "<table></table>"
    rows: dict[int, list[Cell]] = {}
    for cell in cells:
        rows.setdefault(cell.r, []).append(cell)
    out = ["<table>"]
    for r in sorted(rows):
        out.append("<tr>")
        for cell in sorted(rows[r], key=lambda x: x.c):
            attrs = ""
            if cell.rs > 1:
                attrs += f' rowspan="{cell.rs}"'
            if cell.cs > 1:
                attrs += f' colspan="{cell.cs}"'
            tag = cell.tag if cell.tag in ("td", "th") else "td"
            out.append(f"<{tag}{attrs}>{htmllib.escape(cell.text)}</{tag}>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def _header_rows(cells: list[Cell], nrows: int) -> int:
    if nrows <= 1:
        return nrows
    has_th = any(cell.tag == "th" for cell in cells)
    if has_th:
        th_rows = [cell.r + cell.rs for cell in cells if cell.tag == "th"]
        if th_rows:
            return max(th_rows)
    r0 = [cell for cell in cells if cell.r == 0]
    r1 = [cell for cell in cells if cell.r == 1]
    if any(cell.c == 0 and cell.rs >= 2 for cell in r0):
        return 2
    if any(cell.c >= 1 and cell.cs > 1 for cell in r0):
        return 2
    if any(cell.c == 0 and cell.text == "" for cell in r1):
        return 2
    # Years like 2021..2026 in row 1
    r1_data = [cell.text for cell in r1 if cell.c >= 1 and cell.text]
    if r1_data and all(re.fullmatch(r"(?:19|20)\d{2}", t) for t in r1_data):
        return 2
    return 1


def _is_strict_num(text: str) -> bool:
    t = text.strip()
    if not t or not any(ch.isdigit() for ch in t):
        return False
    return bool(_STRICT_NUM_RE.fullmatch(t))


# ----------------------------------------------------------------------------
# P0: collapse_phantom_columns
# ----------------------------------------------------------------------------

def collapse_phantom_columns(cells: list[Cell], nrows: int, ncols: int) -> tuple[list[Cell], int, int]:
    if not cells or ncols <= 1:
        return cells, nrows, ncols
    cells = [Cell(c.r, c.c, c.rs, c.cs, c.text, c.tag) for c in cells]
    changed = True
    while changed and ncols > 1:
        changed = False
        starts = {cell.c for cell in cells}
        for col in range(ncols - 1, -1, -1):
            if col not in starts:
                for cell in cells:
                    if cell.c < col < cell.c + cell.cs:
                        cell.cs -= 1
                    elif cell.c > col:
                        cell.c -= 1
                ncols -= 1
                changed = True
                break
    return cells, nrows, ncols


# ----------------------------------------------------------------------------
# P1: normalize_dashes
# ----------------------------------------------------------------------------

def normalize_dashes(cells: list[Cell]) -> list[Cell]:
    out = []
    for cell in cells:
        txt = cell.text.strip()
        if _DASH_RE.fullmatch(txt):
            txt = "—" if cell.cs > 1 else "–"
        else:
            txt = cell.text
        out.append(Cell(cell.r, cell.c, cell.rs, cell.cs, txt, cell.tag))
    return out


# ----------------------------------------------------------------------------
# P2: normalize_number_format
# ----------------------------------------------------------------------------

def _split_num_decor(text: str) -> tuple[str, str, str, str] | None:
    """Return (prefix_sign, core_digits_seps, suffix_sign, unit) or None."""
    t = text.strip()
    if not _is_strict_num(t):
        return None
    unit = ""
    m_u = _UNIT_RE.match(t)
    if m_u:
        t, unit = m_u.group(1).strip(), " " + m_u.group(2)
    pre, suf = "", ""
    if t.startswith("(") and t.endswith(")"):
        pre, suf = "(", ")"
        t = t[1:-1].strip()
    elif t.startswith("-"):
        pre = "-"
        t = t[1:].strip()
    if not re.fullmatch(r"\d+(?:[., ]\d+)*", t):
        return None
    return pre, t, suf, unit


def normalize_number_format(cells: list[Cell], nrows: int) -> list[Cell]:
    hr = _header_rows(cells, nrows)
    votes = {"int_space": 0, "int_comma": 0, "de": 0, "en": 0}
    parsed: dict[int, tuple[str, str, str, str]] = {}
    for idx, cell in enumerate(cells):
        if cell.r < hr or cell.c == 0:
            continue
        sp = _split_num_decor(cell.text)
        if sp is None:
            continue
        parsed[idx] = sp
        core = sp[1]
        if _INT_SPACE_RE.fullmatch(core):
            votes["int_space"] += 1
        elif _INT_COMMA_RE.fullmatch(core):
            votes["int_comma"] += 1
        elif _DE_DEC_RE.fullmatch(core):
            votes["de"] += 1
        elif _EN_DEC_RE.fullmatch(core):
            votes["en"] += 1

    total = sum(votes.values())
    if total < 3:
        return cells

    out = [Cell(c.r, c.c, c.rs, c.cs, c.text, c.tag) for c in cells]

    def _grp(digits: str, sep: str) -> str:
        return f"{int(digits):,}".replace(",", sep)

    if votes["int_space"] >= 3 and votes["int_space"] / total >= 0.80 and votes["de"] == 0 and votes["en"] == 0 and votes["int_comma"] == 0:
        for idx, (pre, core, suf, unit) in parsed.items():
            if _THREE_GROUP_DOT_COMMA_RE.fullmatch(core):
                digits = core.replace(".", "").replace(",", "")
                out[idx].text = f"{pre}{_grp(digits, ' ')}{suf}{unit}"
            elif _BARE_INT_4_6_RE.fullmatch(core):
                out[idx].text = f"{pre}{_grp(core, ' ')}{suf}{unit}"
    elif votes["int_comma"] >= 3 and votes["int_comma"] / total >= 0.80 and votes["de"] == 0 and votes["en"] == 0 and votes["int_space"] == 0:
        for idx, (pre, core, suf, unit) in parsed.items():
            if _THREE_GROUP_DOT_SPACE_RE.fullmatch(core):
                digits = core.replace(".", "").replace(" ", "")
                out[idx].text = f"{pre}{_grp(digits, ',')}{suf}{unit}"
            elif _BARE_INT_4_6_RE.fullmatch(core):
                out[idx].text = f"{pre}{_grp(core, ',')}{suf}{unit}"
    elif votes["de"] >= 3 and votes["de"] / total >= 0.80 and votes["en"] == 0 and votes["int_space"] == 0 and votes["int_comma"] == 0:
        for idx, (pre, core, suf, unit) in parsed.items():
            if _DE_MISSING_DOT_RE.fullmatch(core):
                whole, dec = core.split(",")
                out[idx].text = f"{pre}{_grp(whole, '.')},{dec}{suf}{unit}"
    elif votes["en"] >= 3 and votes["en"] / total >= 0.80 and votes["de"] == 0 and votes["int_space"] == 0 and votes["int_comma"] == 0:
        for idx, (pre, core, suf, unit) in parsed.items():
            if _EN_MISSING_COMMA_RE.fullmatch(core):
                whole, dec = core.split(".")
                out[idx].text = f"{pre}{_grp(whole, ',')}.{dec}{suf}{unit}"

    return out


# ----------------------------------------------------------------------------
# Pixel-ink grid extractor (pure NumPy + PIL, shared by P3 & P5)
# ----------------------------------------------------------------------------

def _remove_long_lines(mask: np.ndarray, min_run_x: int, min_run_y: int) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for y in range(h):
        row = mask[y]
        if row.sum() >= min_run_x:
            padded = np.concatenate(([False], row, [False]))
            starts = np.where(~padded[:-1] & padded[1:])[0]
            ends = np.where(padded[:-1] & ~padded[1:])[0]
            for s, e in zip(starts, ends):
                if e - s >= min_run_x:
                    out[y, s:e] = False
    for x in range(w):
        col = mask[:, x]
        if col.sum() >= min_run_y:
            padded = np.concatenate(([False], col, [False]))
            starts = np.where(~padded[:-1] & padded[1:])[0]
            ends = np.where(padded[:-1] & ~padded[1:])[0]
            for s, e in zip(starts, ends):
                if e - s >= min_run_y:
                    out[s:e, x] = False
    return out


def _deskew_ink(ink_img: Image.Image) -> np.ndarray:
    small = ink_img.resize((max(1, ink_img.width // 2), max(1, ink_img.height // 2)), Image.BILINEAR)
    best_ang, best_var = 0.0, -1.0
    for ang in np.linspace(-4.5, 4.5, 37):
        rot = np.asarray(small.rotate(ang, resample=Image.BILINEAR, fillcolor=0))
        proj = rot.sum(axis=1, dtype=np.float64)
        v = float(np.var(proj))
        if v > best_var:
            best_var = v
            best_ang = float(ang)
    return np.asarray(ink_img.rotate(best_ang, resample=Image.BILINEAR, fillcolor=0)) > 128


def extract_table_ink_grid(
    image_path: Path, cells: list[Cell], nrows: int, ncols: int
) -> tuple[np.ndarray, list[int], list[int]] | None:
    if nrows < 2 or ncols < 2:
        return None
    img = Image.open(image_path).convert("L")
    clean = img.filter(ImageFilter.MedianFilter(size=3))
    bg = clean.filter(ImageFilter.GaussianBlur(radius=15))
    arr = np.asarray(clean, dtype=np.float32)
    bg_arr = np.asarray(bg, dtype=np.float32)
    raw_ink = ((bg_arr - arr) > 25.0).astype(np.uint8) * 255
    ink = _deskew_ink(Image.fromarray(raw_ink))
    h, w = ink.shape
    text_ink = _remove_long_lines(ink, max(30, w // 8), max(30, h // 8))

    row_sum = text_ink.sum(axis=1)
    col_sum = text_ink.sum(axis=0)
    ys = np.where(row_sum > max(6, np.percentile(row_sum, 90) * 0.05))[0]
    xs = np.where(col_sum > max(6, np.percentile(col_sum, 90) * 0.05))[0]
    if len(ys) < nrows or len(xs) < ncols:
        return None
    y_top, y_bot = int(ys[0]), int(ys[-1]) + 1
    x_left, x_right = int(xs[0]), int(xs[-1]) + 1

    row_edges = [round(y_top + i * (y_bot - y_top) / nrows) for i in range(nrows + 1)]
    rh = (y_bot - y_top) / nrows
    for i in range(1, nrows):
        lo = max(y_top, int(row_edges[i] - 0.3 * rh))
        hi = min(y_bot - 1, int(row_edges[i] + 0.3 * rh))
        if hi > lo:
            row_edges[i] = lo + int(np.argmin(row_sum[lo:hi + 1]))

    clean_rows = [
        r for r in range(1, nrows)
        if all(cell.cs == 1 and cell.rs == 1 for cell in cells if cell.r == r)
    ]
    use_rows = clean_rows if clean_rows else list(range(1, nrows))
    row_has_ink = np.zeros((len(use_rows), w), dtype=bool)
    for idx, r in enumerate(use_rows):
        y0, y1 = row_edges[r], row_edges[r + 1]
        cs_row = text_ink[y0:y1, :].sum(axis=0) >= 2
        dil = np.zeros(w, dtype=bool)
        for dx in range(-6, 7):
            if dx < 0:
                dil[:w + dx] |= cs_row[-dx:]
            elif dx > 0:
                dil[dx:] |= cs_row[:w - dx]
            else:
                dil |= cs_row
        ero = np.ones(w, dtype=bool)
        for dx in range(-6, 7):
            if dx < 0:
                ero[:w + dx] &= dil[-dx:]
            elif dx > 0:
                ero[dx:] &= dil[:w - dx]
            else:
                ero &= dil
        row_has_ink[idx] = ero

    col_occ = row_has_ink.sum(axis=0).astype(float)
    raw_col_sum = np.zeros(w, dtype=float)
    for r in use_rows:
        raw_col_sum += text_ink[row_edges[r]:row_edges[r + 1], :].sum(axis=0)
    prefix_ink = np.concatenate(([0.0], np.cumsum(raw_col_sum)))

    total_ink = float(prefix_ink[x_right] - prefix_ink[x_left])
    min_col_ink = max(20.0, total_ink / (ncols * 5.0))
    min_col_w = max(35, (x_right - x_left) // (ncols * 2 + 2))

    smooth = np.convolve(col_occ, np.ones(5) / 5.0, mode="same")
    k = ncols - 1
    cands = list(range(x_left + min_col_w, x_right - min_col_w + 1, 2))
    m = len(cands)
    if m < k:
        return None
    inf = 1e18
    dp = np.full((k + 1, m), inf)
    prev = np.full((k + 1, m), -1, dtype=int)
    for j, xj in enumerate(cands):
        if prefix_ink[xj] - prefix_ink[x_left] >= min_col_ink:
            dp[1, j] = smooth[xj]
    for i in range(2, k + 1):
        for j, xj in enumerate(cands):
            if i == k and (prefix_ink[x_right] - prefix_ink[xj] < min_col_ink):
                continue
            best_v = inf
            best_p = -1
            for p in range(j):
                xp = cands[p]
                if xj - xp < min_col_w:
                    break
                if dp[i - 1, p] < inf and (prefix_ink[xj] - prefix_ink[xp] >= min_col_ink):
                    if dp[i - 1, p] < best_v:
                        best_v = dp[i - 1, p]
                        best_p = p
            if best_p >= 0:
                dp[i, j] = best_v + smooth[xj]
                prev[i, j] = best_p

    last_j = int(np.argmin(dp[k]))
    if dp[k, last_j] >= inf / 2:
        col_edges = [round(x_left + i * (x_right - x_left) / ncols) for i in range(ncols + 1)]
        return text_ink, row_edges, col_edges

    cuts = []
    cur = last_j
    for i in range(k, 0, -1):
        cuts.append(cands[cur])
        cur = int(prev[i, cur])
    cuts.reverse()
    refined = []
    for x in cuts:
        lo = max(x_left + 10, x - 18)
        hi = min(x_right - 10, x + 18)
        sub = col_occ[lo:hi + 1]
        mn = sub.min()
        mins = np.where(sub == mn)[0]
        refined.append(lo + int(np.median(mins)))
    return text_ink, row_edges, [x_left] + refined + [x_right]


# ----------------------------------------------------------------------------
# P3: split_numeric_colspans
# ----------------------------------------------------------------------------

def split_numeric_colspans(
    cells: list[Cell], nrows: int, ncols: int, image_path: Path | None,
    ink_grid: tuple[np.ndarray, list[int], list[int]] | None = None,
) -> tuple[list[Cell], tuple[np.ndarray, list[int], list[int]] | None]:
    if image_path is None:
        return cells, ink_grid
    hr = _header_rows(cells, nrows)
    by_row: dict[int, list[Cell]] = {}
    for cell in cells:
        by_row.setdefault(cell.r, []).append(cell)

    target_rows = [
        r for r in range(hr, nrows)
        if any(cell.c >= 1 and cell.cs == 2 and _is_strict_num(cell.text)
               for cell in by_row.get(r, []))
    ]
    if not target_rows:
        return cells, ink_grid

    if ink_grid is None:
        ink_grid = extract_table_ink_grid(image_path, cells, nrows, ncols)
    if ink_grid is None:
        return cells, None
    text_ink, row_edges, col_edges = ink_grid

    new_cells: list[Cell] = []
    for r in range(nrows):
        row = sorted(by_row.get(r, []), key=lambda x: x.c)
        if r not in target_rows:
            new_cells.extend(row)
            continue

        data_cells = [cell for cell in row if cell.c >= 1]
        y0, y1 = row_edges[r], row_edges[r + 1]
        inks = [int(text_ink[y0:y1, col_edges[j]:col_edges[j + 1]].sum())
                for j in range(1, ncols)]

        has_complex = (
            any(cell.rs > 1 for cell in row)
            or any(cell.cs > 2 for cell in data_cells)
            or any(cell.cs == 2 and not _is_strict_num(cell.text) for cell in data_cells)
        )
        non_empty = [cell for cell in data_cells if cell.text != ""]
        m = len(non_empty)

        if not has_complex and 0 < m < ncols - 1:
            ranked = sorted(range(1, ncols), key=lambda j: inks[j - 1], reverse=True)
            top_cols = sorted(ranked[:m])
            rest_cols = [j for j in range(1, ncols) if j not in top_cols]
            min_top = min(inks[j - 1] for j in top_cols)
            max_rest = max((inks[j - 1] for j in rest_cols), default=0)
            if min_top >= 2.5 * max(max_rest, 1) or (max_rest <= 15 and min_top >= 40):
                for cell in row:
                    if cell.c == 0:
                        new_cells.append(cell)
                it = iter(non_empty)
                for j in range(1, ncols):
                    if j in top_cols:
                        src = next(it)
                        new_cells.append(Cell(r, j, 1, 1, src.text, src.tag))
                    else:
                        new_cells.append(Cell(r, j, 1, 1, "", "td"))
                continue

        # Fallback: split individual cs==2 numeric cells locally
        for cell in row:
            if cell.c >= 1 and cell.cs == 2 and _is_strict_num(cell.text) and cell.c + 1 < ncols:
                li = inks[cell.c - 1]
                ri = inks[cell.c]
                if li >= ri:
                    new_cells.append(Cell(r, cell.c, cell.rs, 1, cell.text, cell.tag))
                    new_cells.append(Cell(r, cell.c + 1, cell.rs, 1, "", cell.tag))
                else:
                    new_cells.append(Cell(r, cell.c, cell.rs, 1, "", cell.tag))
                    new_cells.append(Cell(r, cell.c + 1, cell.rs, 1, cell.text, cell.tag))
            else:
                new_cells.append(cell)

    return new_cells, ink_grid


# ----------------------------------------------------------------------------
# P4: repair_header_corner_rowspan
# ----------------------------------------------------------------------------

def repair_header_corner_rowspan(cells: list[Cell], nrows: int) -> list[Cell]:
    if nrows < 2:
        return cells
    r0_has_group = any(cell.r == 0 and cell.c >= 1 and cell.cs > 1 for cell in cells)
    if not r0_has_group:
        return cells
    c00 = next((i for i, c in enumerate(cells) if c.r == 0 and c.c == 0), None)
    c10 = next((i for i, c in enumerate(cells) if c.r == 1 and c.c == 0), None)
    if c00 is None or c10 is None:
        return cells
    a, b = cells[c00], cells[c10]
    if a.text == "" and a.rs == 1 and a.cs == 1 and b.text == "" and b.rs == 1 and b.cs == 1:
        out = [Cell(c.r, c.c, c.rs, c.cs, c.text, c.tag)
               for i, c in enumerate(cells) if i != c10]
        new_c00 = next(c for c in out if c.r == 0 and c.c == 0)
        new_c00.rs = 2
        return out
    return cells


# ----------------------------------------------------------------------------
# P5: merge_split_placeholders
# ----------------------------------------------------------------------------

def merge_split_placeholders(
    cells: list[Cell], nrows: int, ncols: int, image_path: Path | None,
    ink_grid: tuple[np.ndarray, list[int], list[int]] | None = None,
) -> tuple[list[Cell], tuple[np.ndarray, list[int], list[int]] | None]:
    if image_path is None:
        return cells, ink_grid
    hr = _header_rows(cells, nrows)
    grid: dict[tuple[int, int], Cell] = {(c.r, c.c): c for c in cells}

    candidates: list[tuple[int, int, str]] = []
    for r in range(hr, nrows):
        c = 1
        while c < ncols - 1:
            a = grid.get((r, c))
            b = grid.get((r, c + 1))
            if a is not None and b is not None and a.rs == 1 and a.cs == 1 and b.rs == 1 and b.cs == 1:
                ta, tb = a.text.strip(), b.text.strip()
                if _PLACEHOLDER_RE.fullmatch(ta) and tb == "":
                    candidates.append((r, c, a.text))
                    c += 2
                    continue
                if ta == "" and _PLACEHOLDER_RE.fullmatch(tb):
                    candidates.append((r, c, b.text))
                    c += 2
                    continue
                if _DASH_RE.fullmatch(ta) and _DASH_RE.fullmatch(tb):
                    candidates.append((r, c, a.text))
                    c += 2
                    continue
            c += 1

    if not candidates:
        return cells, ink_grid
    if ink_grid is None:
        ink_grid = extract_table_ink_grid(image_path, cells, nrows, ncols)
    if ink_grid is None:
        return cells, None
    text_ink, row_edges, col_edges = ink_grid

    merged_starts: dict[tuple[int, int], str] = {}
    skip_coords: set[tuple[int, int]] = set()
    for r, c, label in candidates:
        y0, y1 = row_edges[r], row_edges[r + 1]
        x0, xb, x2 = col_edges[c], col_edges[c + 1], col_edges[c + 2]
        patch = text_ink[y0:y1, x0:x2].copy()
        span_w = x2 - x0
        # Drop residual tilted horizontal border lines spanning >= 45% of the two-column patch
        rule_rows = patch.sum(axis=1) >= 0.45 * span_w
        if rule_rows.any():
            patch[rule_rows, :] = False
        total = int(patch.sum())
        if total < 12:
            continue
        inner = int(patch[:, span_w // 4 : 3 * span_w // 4].sum())
        if inner / total < 0.72:
            continue
        col_w = span_w / 2.0
        col_weights = patch.sum(axis=0).astype(float)
        xs = np.arange(x0, x2, dtype=float)
        cx = float(np.sum(xs * col_weights) / total)
        center_l = (x0 + xb) / 2.0
        center_r = (xb + x2) / 2.0
        if abs(cx - xb) < 0.22 * col_w and abs(cx - xb) < 0.6 * min(abs(cx - center_l), abs(cx - center_r)):
            # Ensure actual ink straddles the boundary (distinguishes 2 separate cells from 1 centered span)
            b_lo = max(0, (xb - x0) - 8)
            b_hi = min(span_w, (xb - x0) + 8)
            if patch[:, b_lo:b_hi].sum() >= 3:
                merged_starts[(r, c)] = label
                skip_coords.add((r, c + 1))

    if not merged_starts:
        return cells, ink_grid

    out: list[Cell] = []
    for cell in cells:
        key = (cell.r, cell.c)
        if key in skip_coords:
            continue
        if key in merged_starts:
            out.append(Cell(cell.r, cell.c, cell.rs, 2, merged_starts[key], cell.tag))
        else:
            out.append(cell)
    return out, ink_grid


# ----------------------------------------------------------------------------
# Combined entrypoint
# ----------------------------------------------------------------------------

def postprocess_html(
    html_str: str,
    image_path: Path | str | None = None,
    *,
    p0: bool = True,
    p1: bool = True,
    p2: bool = True,
    p3: bool = True,
    p4: bool = True,
    p5: bool = True,
) -> str:
    cells, nrows, ncols = parse_table(html_str)
    if not cells:
        return html_str
    img_p = Path(image_path) if image_path is not None else None
    ink_grid = None

    if p0:
        cells, nrows, ncols = collapse_phantom_columns(cells, nrows, ncols)
    if p3:
        cells, ink_grid = split_numeric_colspans(cells, nrows, ncols, img_p, ink_grid)
    if p4:
        cells = repair_header_corner_rowspan(cells, nrows)
    if p5:
        cells, ink_grid = merge_split_placeholders(cells, nrows, ncols, img_p, ink_grid)
    if p2:
        cells = normalize_number_format(cells, nrows)
    if p1:
        cells = normalize_dashes(cells)
    return render_table(cells)
