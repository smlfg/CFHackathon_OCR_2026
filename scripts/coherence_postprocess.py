#!/usr/bin/env python3
"""Post-OCR Table Coherence Classifier & Repair (Layer 1: deterministic, <1 ms; Layer 2: optional SLM).

Works on any TableSalt cell-JSON directory (e.g. Enes's tableparser output).
Never modifies numeric body cells — only repairs structural anomalies and
incoherent header/label OCR typos.

Usage:
  .venv/bin/python scripts/coherence_postprocess.py --in out/eval_matrix/s3_enes_full --out out/eval_matrix/s3_enes_dev_coh
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers for grid representation
# ---------------------------------------------------------------------------

def _dims(cells: list[dict]) -> tuple[int, int]:
    if not cells:
        return 0, 0
    R = max(int(c["r"]) + int(c.get("rs", 1)) for c in cells)
    C = max(int(c["c"]) + int(c.get("cs", 1)) for c in cells)
    return R, C


def _norm(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _is_numeric_like(s: str) -> bool:
    s = _norm(s)
    if not s:
        return False
    digits = sum(ch.isdigit() for ch in s)
    letters = sum(ch.isalpha() for ch in s)
    return digits >= 2 and digits > letters


# ---------------------------------------------------------------------------
# Rule 1: Drop completely blank or 1-2 char junk leading row 0
# ---------------------------------------------------------------------------

def drop_phantom_row0(cells: list[dict]) -> tuple[list[dict], bool]:
    R, C = _dims(cells)
    if R < 3:
        return cells, False

    row0 = [c for c in cells if c["r"] == 0]
    row1 = [c for c in cells if c["r"] == 1]

    # Case A: Row 0 is 100% empty (and all rs==1)
    if all(not _norm(c.get("text", "")) and int(c.get("rs", 1)) == 1 for c in row0):
        out = []
        for c in cells:
            if c["r"] == 0:
                continue
            nc = dict(c)
            nc["r"] = int(c["r"]) - 1
            out.append(nc)
        return out, True

    # Case B: Row 0 has no real multi-char header word — only <=1 short junk cell (<=2 chars,
    # excluding valid headers like Q1..Q4, FY) on non-spanning cells, while Row 1 is a full
    # non-numeric header row (>=3 non-empty text cells, none numeric-like).
    valid_short = {"q1", "q2", "q3", "q4", "fy", "id", "no", "yr"}
    r0_nonspan_texts = [
        _norm(c.get("text", ""))
        for c in row0
        if int(c.get("rs", 1)) == 1 and _norm(c.get("text", ""))
    ]
    r1_texts = [_norm(c.get("text", "")) for c in row1 if _norm(c.get("text", ""))]
    r1_has_nums = any(_is_numeric_like(t) for t in r1_texts)

    if (
        len(r0_nonspan_texts) <= 1
        and all(len(t) <= 2 and t.lower() not in valid_short for t in r0_nonspan_texts)
        and len(r1_texts) >= 3
        and not r1_has_nums
    ):
        # Also verify Row 2 looks like body (contains at least one number or placeholder)
        row2 = [c for c in cells if c["r"] == 2]
        r2_texts = [_norm(c.get("text", "")) for c in row2 if _norm(c.get("text", ""))]
        if any(_is_numeric_like(t) or t.lower() in ("n/a", "—", "–", "-") for t in r2_texts):
            out = []
            for c in cells:
                r = int(c["r"])
                rs = int(c.get("rs", 1))
                if r == 0:
                    if rs > 1:
                        # Promote spanning row-0 cell (e.g. "Item", "Std") down to new row 0
                        nc = dict(c)
                        nc["r"] = 0
                        nc["rs"] = rs - 1
                        out.append(nc)
                    # else drop junk row-0 cell
                else:
                    nc = dict(c)
                    nc["r"] = r - 1
                    out.append(nc)
            return out, True

    return cells, False


# ---------------------------------------------------------------------------
# Rule 2: Merge complementary split columns (e.g. header-only col + body-only col)
# ---------------------------------------------------------------------------

def merge_split_columns(cells: list[dict]) -> tuple[list[dict], bool]:
    changed = False
    while True:
        R, C = _dims(cells)
        if C < 3 or R < 3:
            break

        # Build per-(r, c) cell lookup for simple 1-col cells; avoid touching cols involved in cs>1 spans
        span_cols: set[int] = set()
        grid: dict[tuple[int, int], dict] = {}
        for c in cells:
            cs = int(c.get("cs", 1))
            rs = int(c.get("rs", 1))
            if cs > 1:
                for k in range(int(c["c"]), int(c["c"]) + cs):
                    span_cols.add(k)
            for dr in range(rs):
                for dc in range(cs):
                    grid[(int(c["r"]) + dr, int(c["c"]) + dc)] = c

        merged_pair = None
        for col in range(1, C - 1):  # never merge label col 0 with col 1
            if col in span_cols or (col + 1) in span_cols:
                continue
            # Check if col and col+1 are strictly disjoint (never both non-empty in the same row)
            overlap = 0
            n_left = 0
            n_right = 0
            for r in range(R):
                tl = _norm(grid.get((r, col), {}).get("text", ""))
                tr = _norm(grid.get((r, col + 1), {}).get("text", ""))
                if tl:
                    n_left += 1
                if tr:
                    n_right += 1
                if tl and tr:
                    overlap += 1
            if overlap == 0 and min(n_left, n_right) <= 2 and max(n_left, n_right) >= max(3, R // 2):
                merged_pair = col
                break

        if merged_pair is None:
            break

        col = merged_pair
        # Merge col and col+1 row by row; shift columns > col+1 left by 1
        new_cells: list[dict] = []
        handled: set[tuple[int, int]] = set()
        for c in sorted(cells, key=lambda x: (int(x["r"]), int(x["c"]))):
            r0, c0 = int(c["r"]), int(c["c"])
            if (r0, c0) in handled:
                continue
            if c0 == col:
                partner = next((x for x in cells if int(x["r"]) == r0 and int(x["c"]) == col + 1), None)
                nc = dict(c)
                if partner is not None:
                    handled.add((r0, col + 1))
                    t1, t2 = _norm(c.get("text", "")), _norm(partner.get("text", ""))
                    nc["text"] = t1 or t2
                new_cells.append(nc)
            elif c0 == col + 1:
                # Only happens if col had no cell at r0
                nc = dict(c)
                nc["c"] = col
                new_cells.append(nc)
            elif c0 > col + 1:
                nc = dict(c)
                nc["c"] = c0 - 1
                new_cells.append(nc)
            else:
                new_cells.append(dict(c))

        cells = new_cells
        changed = True

    return cells, changed


# ---------------------------------------------------------------------------
# Rule 3: Sequence coherence on header rows (Q1..Q4, 20xx years, Train/Dev/Test/OOD)
# ---------------------------------------------------------------------------

_Q_RE = re.compile(r"^Q([1-4])$", re.I)
_YEAR_RE = re.compile(r"^(19|20)[., ]?(\d{2})$")


def _extract_year(t: str) -> int | None:
    m = _YEAR_RE.match(_norm(t))
    if not m:
        return None
    return int(m.group(1) + m.group(2))


def repair_header_coherence(cells: list[dict]) -> tuple[list[dict], bool]:
    R, C = _dims(cells)
    if R < 2 or C < 3:
        return cells, False

    changed = False
    cells = [dict(c) for c in cells]

    for hr in (0, 1):
        row = sorted([c for c in cells if int(c["r"]) == hr and int(c.get("cs", 1)) == 1], key=lambda x: int(x["c"]))
        if len(row) < 3:
            continue
        texts = [_norm(c.get("text", "")) for c in row]

        # A) Quarter sequence coherence: if >= 2 cells match Q[1-4] in order, fix broken neighbors
        q_pos: list[tuple[int, int]] = []  # (idx_in_row, quarter_num)
        for i, t in enumerate(texts):
            m = _Q_RE.match(t)
            if m:
                q_pos.append((i, int(m.group(1))))
        if len(q_pos) >= 2:
            # Check consistent step +1: q_num - idx should be constant for valid anchors
            offsets = [qn - i for i, qn in q_pos]
            # Majority offset
            best_off = max(set(offsets), key=offsets.count)
            if offsets.count(best_off) >= 2:
                for i, c in enumerate(row):
                    if int(c["c"]) == 0:
                        continue  # don't overwrite col-0 label header
                    expected_q = i + best_off
                    if 1 <= expected_q <= 4:
                        cur = texts[i]
                        if not _Q_RE.match(cur):
                            # Only replace if cur looks corrupted (short, digits, or junk — not "FY"/"YoY")
                            if cur.lower() not in ("fy", "yoy", "total", "avg", "sum", "delta", "plan", "actual"):
                                c["text"] = f"Q{expected_q}"
                                changed = True

        # B) Consecutive year sequence coherence (e.g., 20,21 | EZ | 20,23 | 20,24 -> 2021..2024)
        y_pos: list[tuple[int, int]] = []
        for i, t in enumerate(texts):
            y = _extract_year(t)
            if y is not None:
                y_pos.append((i, y))
        if len(y_pos) >= 2:
            offsets = [y - i for i, y in y_pos]
            best_off = max(set(offsets), key=offsets.count)
            if offsets.count(best_off) >= 2:
                for i, c in enumerate(row):
                    if int(c["c"]) == 0:
                        continue
                    exp_y = i + best_off
                    cur = texts[i]
                    if cur != str(exp_y) and cur.lower() not in ("total", "avg", "delta", "yoy", "fy", "cagr"):
                        c["text"] = str(exp_y)
                        changed = True

        # C) ML benchmark header coherence: Train / Dev / Test / <corrupted OOD>
        lower_set = {t.lower() for t in texts}
        if {"train", "dev", "test"}.issubset(lower_set):
            for c in row:
                t = _norm(c.get("text", ""))
                tl = t.lower().replace(" ", "")
                if tl in ("00d", "0d", "cod", "o0d", "0od", "ood.", "od") and t != "OOD":
                    c["text"] = "OOD"
                    changed = True

    return cells, changed


def apply_coherence(cells: list[dict]) -> tuple[list[dict], list[str]]:
    applied: list[str] = []
    cells, ok = drop_phantom_row0(cells)
    if ok:
        applied.append("drop_phantom_row0")
    cells, ok = merge_split_columns(cells)
    if ok:
        applied.append("merge_split_columns")
    cells, ok = repair_header_coherence(cells)
    if ok:
        applied.append("repair_header_coherence")
    return cells, applied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inp, out = Path(args.inp), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_changed = 0
    rule_counts: dict[str, int] = {}
    for p in sorted(inp.glob("*.json")):
        if p.name.startswith("_"):
            continue
        cells = json.loads(p.read_text(encoding="utf-8"))
        new_cells, rules = apply_coherence(cells)
        if rules:
            n_changed += 1
            for r in rules:
                rule_counts[r] = rule_counts.get(r, 0) + 1
        (out / p.name).write_text(json.dumps(new_cells, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Coherence Layer 1: touched {n_changed} tables  rules={rule_counts}")


if __name__ == "__main__":
    main()
