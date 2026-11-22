"""Self-check: does a finished table look right?  (No ground truth needed.)

Checks a cell list against the rules every table of the generator follows
(docs/NOTES.md) and returns a list of problems -- empty means "looks right".
Used after reading to decide which tables deserve a second opinion.

Rules checked
  header     data-column headers are exactly COL_POOLS[k][:n-1] for some k
  groups     two-row header: group labels from GROUP_NAMES, corner label from STUB_NAMES or ""
  labels     row labels all come from ONE row pool, without repeats
  numbers    every data cell is a special ("", "–", "n/a", merged "n/a"/"—"/"see note 3")
             or a number in ONE table-wide format (en / de / int with , or space;
             negatives all "-" or all "( )"; at most one unit, " %", " €", " kg" or " ms")
  empties    not more empty data cells than the generator plausibly makes
  grid       every position covered exactly once (validate.check)
"""

from __future__ import annotations

import re

from . import lexicon, validate

NUM = {
    "en": r"\d{1,3}(?:,\d{3})*\.\d{2}",
    "de": r"\d{1,3}(?:\.\d{3})*,\d{2}",
    "int,": r"\d{1,3}(?:,\d{3})*",
    "int ": r"\d{1,3}(?: \d{3})*",
}
UNIT_RE = r"(?: (%|€|kg|ms))?"


def _num_style(t: str):
    """-> (style, neg_kind, unit) if t is a number in some generator format, else None."""
    for style, core in NUM.items():
        for neg, pat in (("none", rf"{core}"), ("minus", rf"-{core}"), ("paren", rf"\({core}\)")):
            m = re.fullmatch(pat + UNIT_RE, t)
            if m:
                if style.startswith("int") and not re.search(r"\d[ ,]\d", t):
                    style = "int"  # no separator visible: fits both integer styles
                return style, neg, m.group(1) or ""
    return None


def check(cells: list[dict], max_empty_share: float = 0.15) -> list[str]:
    problems: list[str] = []
    if not cells:
        return ["no cells"]
    try:
        problems += [f"grid: {p}" for p in validate.check(cells)][:3]
    except Exception as ex:  # never let the checker crash the run
        problems.append(f"grid: {ex!r}")
    at = {(c["r"], c["c"]): c for c in cells}
    n_cols = max(c["c"] + c.get("cs", 1) for c in cells)
    hr = 2 if at.get((0, 0), {}).get("rs", 1) == 2 else 1

    # header sequence
    heads = [at[(hr - 1, j)]["text"] if (hr - 1, j) in at else None for j in range(1, n_cols)]
    if not any(heads == pool[: n_cols - 1] for pool in lexicon.COL_POOLS):
        problems.append(f"header: {heads} is no known sequence")
    if hr == 2:
        stub = at.get((0, 0), {}).get("text", "")
        if stub not in lexicon.STUB_NAMES + [""]:
            problems.append(f"groups: corner label {stub!r} unknown")
        for c in cells:
            if c["r"] == 0 and c["c"] > 0 and c["text"] not in lexicon.GROUP_NAMES:
                problems.append(f"groups: {c['text']!r} unknown")
    elif at.get((0, 0), {}).get("text", "") not in lexicon.STUB_NAMES + [""]:
        problems.append(f"groups: corner label {at[(0, 0)]['text']!r} unknown")

    # row labels
    labels = [
        c["text"] for c in sorted(cells, key=lambda c: c["r"]) if c["c"] == 0 and c["r"] >= hr
    ]
    if not any(set(labels) <= set(pool) for pool in lexicon.ROW_POOLS) or len(set(labels)) != len(
        labels
    ):
        problems.append("labels: not one topic list, or repeated")

    # numbers / specials
    styles, negs, units, bad, empty, total = set(), set(), set(), [], 0, 0
    for c in cells:
        if c["r"] < hr or c["c"] == 0:
            continue
        total += 1
        t = c["text"]
        if c.get("cs", 1) == 2:
            if t not in lexicon.MERGE_TEXTS:
                bad.append(t)
            continue
        if t == "":
            empty += 1
            continue
        if t in lexicon.SINGLE_SPECIALS:
            continue
        s = _num_style(t)
        if s is None:
            bad.append(t)
            continue
        styles.add(s[0])
        units.add(s[2])
        if s[1] != "none":
            negs.add(s[1])
    kinds = styles - {"int"}
    if len(kinds) > 1 or (kinds & {"en", "de"} and "int" in styles):
        problems.append(f"numbers: mixed formats {sorted(styles)}")
    if len(negs) > 1:
        problems.append("numbers: minus and brackets mixed")
    if len(units - {""}) > 1:
        problems.append(f"numbers: several units {sorted(units - {''})}")
    if bad:
        problems.append(f"numbers: {len(bad)} cells unreadable, e.g. {bad[:3]}")
    if total and empty / total > max_empty_share:
        problems.append(f"empties: {empty}/{total} data cells empty")
    return problems
