"""Step 6 - postprocess: one number format per table, labels snapped to the lexicon.

Numbers are rebuilt from their digits: the generator prints en/de with exactly 2 decimals
and ``int`` without, so once the table's format is known (majority vote over the cells read
well) every value is re-printed canonically - this repairs lost / swapped separators.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from . import lexicon as lx
from .config import CFG, Config
from .types import Cell

DIGIT_FIX = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "D": "0",
        "Q": "0",
        "l": "1",
        "I": "1",
        "|": "1",
        "i": "1",
        "!": "1",
        "S": "5",
        "s": "5",
        "B": "8",
        "Z": "2",
        "z": "2",
        "G": "6",
        "g": "9",
        "q": "9",
        "A": "4",
        "T": "7",
    }
)
DASHES = "-–—−~_="
UNIT_ALIASES = {
    "%": "%",
    "°/o": "%",
    "o/o": "%",
    "€": "€",
    "e": "€",
    "c": "€",
    "ε": "€",
    "kg": "kg",
    "kq": "kg",
    "k9": "kg",
    "kgs": "kg",
    "ka": "kg",
    "ko": "kg",
    "ms": "ms",
    "rns": "ms",
    "mss": "ms",
    "m5": "ms",
    "ns": "ms",
    "rn5": "ms",
}


@dataclass
class Num:
    neg: bool
    digits: str
    dec_sep: str | None  # "." / "," if the text ends in <sep><2 digits>
    group_seps: str  # separators seen between digit groups
    unit: str | None  # raw unit token
    parens: bool
    minus: bool


NUM_CORE = re.compile(r"[\d][\d\s.,'`]*\d|\d")


def parse(text: str) -> Num | None:
    """Loose reading of an OCR'd number cell. None if it has fewer than 2 digits."""
    t = text.strip()
    if sum(ch.isdigit() for ch in t) < 1:
        # no digit at all: only accept if (almost) every character is a look-alike digit
        t2 = t.translate(DIGIT_FIX)
        core = [ch for ch in t2 if not ch.isspace()]
        if sum(ch.isdigit() for ch in t2) < 3 or sum(
            ch.isdigit() or ch in ".,()-" for ch in core
        ) < 0.8 * len(core):
            return None
        t = t2
    # split off a trailing unit: letters / symbols after the last digit or ')'
    m = re.search(r"[\d)\]}]([^\d)\]}]*)$", t)
    tail = m.group(1) if m else ""
    head = t[: len(t) - len(tail)] if tail else t
    unit = tail.strip() or None
    # inside the number, letters are misread digits
    head = re.sub(
        r"(?<=[\d.,\s])[OoDQlIi|!SBZzG](?=[\d.,\s]|$)",
        lambda mm: mm.group(0).translate(DIGIT_FIX),
        head,
    )
    head = re.sub(r"^[OoDQlIi|!SBZzG](?=[\d.,])", lambda mm: mm.group(0).translate(DIGIT_FIX), head)
    digits = "".join(ch for ch in head if ch.isdigit())
    if len(digits) < 2:
        return None
    lead = head.lstrip()
    parens = lead[:1] in "([{" or head.rstrip()[-1:] in ")]}"
    minus = (lead[:1] in DASHES) and not parens
    core = NUM_CORE.search(head)
    core_s = core.group(0) if core else digits
    dm = re.search(r"([.,])\s?(\d{2})$", core_s)
    dec_sep = dm.group(1) if dm else None
    body = core_s[: dm.start()] if dm else core_s
    group_seps = "".join(ch for ch in body if ch in ".,' " and True)
    return Num(
        neg=parens or minus,
        digits=digits,
        dec_sep=dec_sep,
        group_seps=group_seps,
        unit=unit,
        parens=parens,
        minus=minus,
    )


@dataclass
class Fmt:
    style: str = "en"  # en / de / int
    int_sep: str = ","
    parens: bool = False
    unit: str = ""  # table unit without leading space ("" = none)


def detect_format(nums: list[Num]) -> Fmt:
    style = Counter()
    seps = Counter()
    neg = Counter()
    units = Counter()
    for n in nums:
        if n.dec_sep is not None and len(n.digits) >= 3:
            gs = n.group_seps.replace(" ", "")
            if n.dec_sep == "." and "." not in gs:
                style["en"] += 1
            elif n.dec_sep == "," and "," not in gs:
                style["de"] += 1
            # "12,345" style ints look like ",dd" only by a missing digit -> ignore
        elif len(n.digits) >= 4 and n.group_seps:
            style["int"] += 1
            if " " in n.group_seps or "'" in n.group_seps:
                seps[" "] += 1
            elif "," in n.group_seps or "." in n.group_seps:
                seps[","] += 1
        elif len(n.digits) <= 5 and n.dec_sep is None:
            style["int"] += 0.5
        if n.parens:
            neg["parens"] += 1
        elif n.minus:
            neg["minus"] += 1
        if n.unit:
            u = unit_of(n.unit)
            if u:
                units[u] += 1
    f = Fmt()
    if style:
        f.style = style.most_common(1)[0][0]
    if seps:
        f.int_sep = seps.most_common(1)[0][0]
    f.parens = neg["parens"] > neg["minus"]
    if units:
        f.unit = units.most_common(1)[0][0]
    return f


def unit_of(tok: str) -> str | None:
    t = tok.strip().lower().replace(" ", "")
    if not t:
        return None
    if t in UNIT_ALIASES:
        return UNIT_ALIASES[t]
    if "%" in t:
        return "%"
    if "€" in t:
        return "€"
    if t.startswith("k"):
        return "kg"
    if t.endswith("s") and len(t) <= 3:
        return "ms"
    return None


def group(digits: str, sep: str) -> str:
    digits = digits.lstrip("0") or "0"
    out = []
    while len(digits) > 3:
        out.append(digits[-3:])
        digits = digits[:-3]
    out.append(digits)
    return sep.join(reversed(out))


def render(n: Num, f: Fmt) -> str:
    d = n.digits
    if f.style in ("en", "de"):
        if len(d) < 3:
            d = d.rjust(3, "0")
        ip, dp = d[:-2], d[-2:]
        if f.style == "en":
            s = f"{group(ip, ',')}.{dp}"
        else:
            s = f"{group(ip, '.')},{dp}"
    else:
        s = group(d, f.int_sep)
    if n.neg and s.strip("0.,") != "":
        s = f"({s})" if f.parens else f"-{s}"
    if n.unit and f.unit and unit_of(n.unit):
        s += f" {f.unit}"
    return s


# ----------------------------------------------------------------------------
# whole table
# ----------------------------------------------------------------------------

ALL_COL_LABELS = [x for p in lx.COL_POOLS for x in p]
ALL_ROW_LABELS = [x for p in lx.ROW_POOLS for x in p]


def role_vocab(c: Cell, hr: int) -> list[str] | None:
    if c.r == 0 and c.c == 0:
        return lx.STUB_NAMES
    if c.r < hr:
        return lx.GROUP_NAMES if (hr == 2 and c.r == 0) else ALL_COL_LABELS
    if c.c == 0:
        return ALL_ROW_LABELS
    return None


def plausible(n: Num, f: Fmt) -> bool:
    """Digit count / decimal separator consistent with the table format (generator range
    -9999 .. 99999, en/de always 2 decimals)."""
    if f.style in ("en", "de"):
        want = "." if f.style == "en" else ","
        return (
            3 <= len(n.digits) <= 7 and n.dec_sep in (want, None) and len(n.digits.lstrip("0")) <= 7
        )
    return 1 <= len(n.digits) <= 5 and n.dec_sep is None


def choose_reads(
    cells: list[Cell], hr: int, cands: list[list[tuple[str, float]]], blob_x: list[int]
) -> None:
    """Pick one OCR read per cell.  Labels: the read closest to the vocabulary of the
    cell's role.  Body values: the read that parses as a plausible number in the table's
    format, highest confidence first."""

    def joined(c: Cell, k: int) -> tuple[str, float]:
        ids = sorted(c.blobs, key=lambda i: blob_x[i])
        parts = [cands[i][k] for i in ids]
        text = " ".join(t for t, _ in parts if t).strip()
        conf = sum(cf for _, cf in parts) / len(parts) if parts else -1.0
        return text, conf

    n_var = len(cands[0]) if cands else 0
    body = []
    for c in cells:
        if not c.blobs or n_var == 0:
            continue
        if c.text in (lx.EN_DASH, lx.EM_DASH):  # decided by blob shape in spans
            continue
        opts = [joined(c, k) for k in range(n_var)]
        vocab = role_vocab(c, hr)
        if vocab is not None:
            c.text = max(opts, key=lambda o: (lx.best(o[0], vocab)[1], o[1]))[0]
        else:
            c.text = max(opts, key=lambda o: o[1])[0]
            body.append((c, opts))
    fmt = detect_format([n for c, _ in body if (n := parse(c.text)) is not None])
    for c, opts in body:
        parsed = [(o, parse(o[0])) for o in opts]
        good = [(o, n) for o, n in parsed if n is not None and plausible(n, fmt)]
        pool = good or [(o, n) for o, n in parsed if n is not None]
        if pool:
            # vote over the digit string; ties -> summed confidence
            votes: dict[str, list] = {}
            for o, n in pool:
                v = votes.setdefault(n.digits, [0, 0.0, o])
                v[0] += 1
                v[1] += max(o[1], 0.0)
                if o[1] > v[2][1]:
                    v[2] = o
            c.text = max(votes.values(), key=lambda v: (v[0], v[1]))[2][0]


def fix_table(
    cells: list[Cell],
    hr: int,
    n_cols: int,
    cfg: Config = CFG,
    cands: list[list[tuple[str, float]]] | None = None,
    blob_x: list[int] | None = None,
) -> list[Cell]:
    if cands is not None and blob_x is not None:
        choose_reads(cells, hr, cands, blob_x)
    by_pos = {(c.r, c.c): c for c in cells}

    # column headers: the generator always prints pool[:n_cols-1] in order
    hdr = [by_pos.get((hr - 1, c)) for c in range(1, n_cols)]
    hdr_texts = [h.text if h is not None and h.cs == 1 else "" for h in hdr]
    k, s = lx.choose_pool(hdr_texts, lx.COL_POOLS)
    if k >= 0 and s >= cfg.label_min_score:
        pool = lx.COL_POOLS[k]
        agree = sum(
            1
            for j, t in enumerate(hdr_texts)
            if lx.norm(t) and j < len(pool) and lx.best(t, pool)[0] == pool[j]
        )
        nonempty = sum(1 for t in hdr_texts if lx.norm(t))
        if n_cols - 1 <= len(pool) and nonempty and agree >= cfg.pool_min_share * nonempty:
            for j, h in enumerate(hdr):
                if h is not None and h.cs == 1:
                    h.text = pool[j]
        else:
            for h in hdr:
                if h is not None and lx.norm(h.text):
                    c, sc = lx.best(h.text, pool)
                    if sc >= cfg.label_min_score:
                        h.text = c

    # grouped header row
    if hr == 2:
        for c in range(1, n_cols):
            g = by_pos.get((0, c))
            if g is not None and lx.norm(g.text):
                name, sc = lx.best(g.text, lx.GROUP_NAMES)
                if sc >= cfg.group_min_score:
                    g.text = name
    stub = by_pos.get((0, 0))
    if stub is not None:
        if lx.norm(stub.text):
            name, sc = lx.best(stub.text, lx.STUB_NAMES)
            stub.text = name if sc >= 50 else stub.text
        else:
            stub.text = ""

    # row labels: one pool per table, sampled without replacement
    labels = [c for c in cells if c.c == 0 and c.r >= hr]
    k, s = lx.choose_pool([c.text for c in labels], lx.ROW_POOLS)
    if k >= 0 and s >= cfg.label_min_score:
        # pool is certain: every label is one of its 15 entries, so snap even weak reads
        snapped = lx.assign_unique(
            [c.text for c in labels], lx.ROW_POOLS[k], cfg.row_label_min_score
        )
        for c, sn in zip(labels, snapped):
            if sn is not None:
                c.text = sn

    # body values
    body = [c for c in cells if c.r >= hr and c.c >= 1]
    parsed = {}
    for c in body:
        if c.cs > 1:
            c.text = _merged_text(c.text)
            continue
        sp = _special(c.text)
        if sp is not None:
            c.text = sp
            continue
        n = parse(c.text)
        if n is not None:
            parsed[id(c)] = (c, n)
    fmt = detect_format([n for _, n in parsed.values()])
    for c, n in parsed.values():
        if c.neg_hint and not fmt.parens:
            n.neg = True  # geometry saw a minus Tesseract dropped
        c.text = render(n, fmt)
    return cells


def _merged_text(t: str) -> str:
    """A merged body cell holds one of three texts (generator): pick the closest."""
    if t in (lx.EM_DASH, lx.EN_DASH) or (t and all(ch in DASHES for ch in t)):
        return lx.EM_DASH
    n = lx.norm(t)
    if not n:
        return ""
    if "note" in n or len(n) >= 6:
        return "see note 3"
    return "n/a"


def _special(t: str) -> str | None:
    """Single body cells are numbers, "–", "n/a" or empty (generator) - and "see note 3"
    when a merge was missed.  Short text without digits can only be "n/a"."""
    s = t.strip()
    if not s:
        return ""
    if s in (lx.EN_DASH, lx.EM_DASH) or all(ch in DASHES for ch in s):
        return lx.EN_DASH
    n = lx.norm(s)
    n_digits = sum(ch.isdigit() for ch in n)
    if "note" in n or (len(n) >= 6 and lx.score(n, "see note 3") >= 50 and n_digits <= 1):
        return "see note 3"
    if n_digits == 0 and len(n) <= 5:
        return "n/a"
    return None
