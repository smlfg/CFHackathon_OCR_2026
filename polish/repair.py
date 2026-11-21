"""Fix cell text using what tools/learn_from_gt.py learned from ground truth.

repair_cells(cells) -> cells   (cells as produced by src/structure.py)

1. Labels and headers: snap to the learned vocabulary.
   * column headers: pick the learned header sequence (e.g. Q1 Q2 Q3 Q4 FY)
     that best matches the whole OCR'd header row; fixes "04"->"Q4", "00D"->"OOD"
   * row labels: pick the learned label group (topic) that fits the table,
     then snap each label to its closest member; fixes "Receivabies"
   * group labels and the corner label: closest learned name
2. Numbers: a table uses ONE number format. Vote on it from all cells
   (1,234.56 / 1.234,56 / 12,345 / 12 345; minus or brackets), then rebuild
   every number in that format from its digits; fixes "73:840:33", "42.521"
   in an integer table, "(5.872" missing bracket, "6,621%" spacing.
3. Special cells: n/a, —, –, see note 3.
Only changes text when the match is confident; otherwise leaves the cell alone.
"""
from __future__ import annotations

import collections
import difflib
import json
import re
from pathlib import Path

VOCAB = json.loads((Path(__file__).parent / "learned" / "vocab.json").read_text(encoding="utf-8"))
UNITS = ("%", "€", "kg", "ms")
STRIP = " |‚'`\"_:;,.!~^"


def canon(s: str) -> str:
    s = s.lower().replace("0", "o").replace("|", "l").replace("1", "l").replace("5", "s")
    return re.sub(r"[^a-z0-9äöüß() ]", "", s)


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, canon(a), canon(b)).ratio()


def snap(text, options, cutoff):
    t = text.strip(STRIP)
    if not t:
        return text
    best = max(options, key=lambda o: sim(t, o), default=None)
    return best if best is not None and sim(t, best) >= cutoff else t


# ------------------------------------------------------------------ numbers
DEC = re.compile(r"[.,](\d{2})(?!\d)")


def classify(t: str):
    """-> ('en'|'de'|'int', int_sep) or None for a single number string."""
    core = re.sub(r"[^\d.,\s]", "", t).strip()
    if not re.search(r"\d", core):
        return None
    seps = [(m.start(), m.group()) for m in re.finditer(r"[.,]|(?<=\d) (?=\d{3}\b)", core)]
    if not seps:
        return ("int", None)
    pos, ch = seps[-1]
    tail = re.sub(r"\D", "", core[pos + 1:])
    if len(tail) == 2:
        return ("en" if ch == "." else "de", None)
    if len(tail) == 3:
        return ("int", ch)
    return None


def table_format(values):
    styles, seps, paren, minus = collections.Counter(), collections.Counter(), 0, 0
    for t in values:
        c = classify(t)
        if c:
            styles[c[0]] += 1
            if c[0] == "int" and c[1]:
                seps[c[1]] += 1
        if "(" in t or ")" in t:
            paren += 1
        if re.match(r"^\s*[-–]\s*\d", t):
            minus += 1
    if not styles:
        return None
    style = styles.most_common(1)[0][0]
    return dict(style=style, int_sep=(seps.most_common(1)[0][0] if seps else ","),
                parens=paren > minus)


def group3(digits: str, sep: str) -> str:
    digits = digits.lstrip("0") or "0"
    out = []
    while len(digits) > 3:
        out.insert(0, digits[-3:]); digits = digits[:-3]
    out.insert(0, digits)
    return sep.join(out)


def rebuild(t: str, fmt) -> str:
    unit = ""
    m = re.search(r"\s*(%|€|kg|ms|k9|kq|ka)\s*\.?$", t)
    if m:
        unit = {"k9": "kg", "kq": "kg", "ka": "kg"}.get(m.group(1), m.group(1))
        t = t[: m.start()]
    if re.search(r"[A-Za-z]{2,}", t):         # words inside: not a plain number
        return None
    digits = re.sub(r"\D", "", t)
    if not digits:
        return None
    neg = bool(re.search(r"[()]", t)) or bool(re.match(r"^\s*[-–~]", t))
    if fmt["style"] in ("en", "de"):
        if len(DEC.findall(t)) > 1 or len(digits) < 3 or len(digits) > 7:
            return None                        # two numbers glued together, or garbage
        ip, dp = digits[:-2], digits[-2:]
        th, dc = (",", ".") if fmt["style"] == "en" else (".", ",")
        body = f"{group3(ip, th)}{dc}{dp}"
    else:
        if len(digits) > 5:
            return None
        body = group3(digits, fmt["int_sep"])
    if neg:
        body = f"({body})" if fmt["parens"] else f"-{body}"
    return body + (f" {unit}" if unit else "")


# ------------------------------------------------------------------ main
def repair_cells(cells):
    if not cells:
        return cells
    hr = 2 if any(c["r"] == 0 and c["c"] == 0 and c["rs"] == 2 for c in cells) else 1
    ncols = max(c["c"] + c["cs"] for c in cells)

    # column header row: best whole learned sequence
    label_row = sorted([c for c in cells if c["r"] == hr - 1 and c["c"] >= 1], key=lambda c: c["c"])
    seqs = [s for s in VOCAB["header_sequences"] if len(s) == ncols - 1]
    if label_row and seqs and len(label_row) == ncols - 1:
        best = max(seqs, key=lambda s: sum(sim(c["text"], x) for c, x in zip(label_row, s)))
        score = sum(sim(c["text"], x) for c, x in zip(label_row, best)) / len(best)
        if score >= 0.4:
            for c, x in zip(label_row, best):
                c["text"] = x
    else:
        for c in label_row:
            c["text"] = snap(c["text"], VOCAB["header"], 0.6)

    for c in cells:
        if c["r"] == 0 and c["c"] == 0:
            c["text"] = snap(c["text"], [s for s in VOCAB["stub_names"] if s], 0.5) if c["text"].strip(STRIP) else ""
        elif hr == 2 and c["r"] == 0:
            c["text"] = snap(c["text"], VOCAB["group_names"], 0.5)

    # row labels: choose the topic, then snap
    labels = [c for c in cells if c["r"] >= hr and c["c"] == 0]
    if labels:
        group = max(VOCAB["label_groups"],
                    key=lambda g: sum(max(sim(c["text"], x) for x in g) for c in labels))
        for c in labels:
            c["text"] = snap(c["text"], group, 0.5)

    # values
    values = [c for c in cells if c["r"] >= hr and c["c"] >= 1]
    fmt = table_format([c["text"] for c in values])
    for c in values:
        t = c["text"].strip(" |‚'`\"_")
        low = t.lower().strip(".")
        if not t:
            c["text"] = ""
        elif low in ("n/a", "na", "n/a.", "nia", "n/e") or sim(t, "n/a") >= 0.8:
            c["text"] = "n/a"
        elif re.fullmatch(r"[-–—~_=]+", t):
            c["text"] = "—" if c["cs"] >= 2 else "–"
        elif sim(t, "see note 3") >= 0.6:
            c["text"] = "see note 3"
        elif fmt and re.search(r"\d", t):
            r = rebuild(t, fmt)
            c["text"] = r if r is not None else t
        else:
            c["text"] = t
    return cells
