"""Vocabulary of generate_dataset.py (see docs/NOTES.md) + fuzzy matching.

The lists are copied, not imported: the container must not depend on the generator file.
"""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

ROW_POOLS = [
    [
        "Revenue",
        "Cost of sales",
        "Gross profit",
        "Operating expenses",
        "EBITDA",
        "Depreciation",
        "EBIT",
        "Interest expense",
        "Taxes",
        "Net income",
        "Cash flow",
        "Capex",
        "Working capital",
        "Inventory",
        "Receivables",
    ],
    [
        "Umsatzerlöse",
        "Materialaufwand",
        "Rohertrag",
        "Personalaufwand",
        "Abschreibungen",
        "Sonstige Kosten",
        "Betriebsergebnis",
        "Zinsaufwand",
        "Steuern",
        "Jahresüberschuss",
        "Rückstellungen",
        "Verbindlichkeiten",
        "Forderungen",
        "Vorräte",
        "Eigenkapital",
    ],
    [
        "Hydrogen",
        "Helium",
        "Lithium",
        "Carbon",
        "Nitrogen",
        "Oxygen",
        "Sodium",
        "Magnesium",
        "Aluminium",
        "Silicon",
        "Phosphorus",
        "Sulfur",
        "Chlorine",
        "Potassium",
        "Calcium",
    ],
    [
        "Baseline",
        "Variant A",
        "Variant B",
        "Variant C",
        "Ablation 1",
        "Ablation 2",
        "Ablation 3",
        "Ours (small)",
        "Ours (base)",
        "Ours (large)",
        "Prior SOTA",
        "Human",
        "Random",
        "Majority",
        "Oracle",
    ],
]

# column headers are always pool[:n_cols - 1] in this order (never repeated: <= 6 data cols)
COL_POOLS = [
    ["Q1", "Q2", "Q3", "Q4", "FY", "YoY"],
    ["2021", "2022", "2023", "2024", "2025", "2026"],
    ["Min", "Max", "Mean", "Median", "Std", "N"],
    ["Train", "Dev", "Test", "OOD", "Avg", "Rank"],
    ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun"],
]

GROUP_NAMES = ["Actual", "Plan", "Delta", "Reported", "Adjusted"]
STUB_NAMES = ["Position", "Item", "Metric"]  # or "" (empty)
UNITS = ["%", "€", "kg", "ms"]  # printed as " <unit>" after the number
EN_DASH = "–"  # single empty-ish body cell
EM_DASH = "—"  # merged body cell (cs=2)
MERGE_TEXTS = ["n/a", EM_DASH, "see note 3"]  # merged body cells
SINGLE_SPECIALS = [EN_DASH, "n/a", ""]


def norm(s: str) -> str:
    """NFKC-normalize, casefold and collapse whitespace for fuzzy matching."""
    s = unicodedata.normalize("NFKC", s or "").casefold()
    return re.sub(r"\s+", " ", s).strip()


def score(a: str, b: str) -> float:
    """Fuzzy similarity (0..100) between two strings after normalization."""
    return float(fuzz.ratio(norm(a), norm(b)))


def best(text: str, choices: list[str]) -> tuple[str, float]:
    """Closest choice and its similarity (0..100)."""
    if not choices:
        return "", 0.0
    scored = [(score(text, c), c) for c in choices]
    s, c = max(scored, key=lambda t: t[0])
    return c, s


def choose_pool(texts: list[str], pools: list[list[str]]) -> tuple[int, float]:
    """Pool that explains the texts best -> (pool index, mean best score)."""
    texts = [t for t in texts if norm(t)]
    if not texts:
        return -1, 0.0
    res = []
    for k, pool in enumerate(pools):
        res.append((sum(best(t, pool)[1] for t in texts) / len(texts), k))
    s, k = max(res)
    return k, s


def assign_unique(texts: list[str], pool: list[str], min_score: float) -> list[str | None]:
    """Greedy one-to-one snapping of texts to pool entries (labels are sampled without
    replacement).  None where nothing is similar enough."""
    pairs = sorted(((score(t, p), i, p) for i, t in enumerate(texts) for p in pool), reverse=True)
    out: list[str | None] = [None] * len(texts)
    used: set[str] = set()
    for s, i, p in pairs:
        if s < min_score:
            break
        if out[i] is None and p not in used:
            out[i] = p
            used.add(p)
    return out
