#!/usr/bin/env python3
"""LightOnOCR-3 output -> the TableSalt cells JSON (README §5).

The model answers with a page transcription: either markdown pipe tables, sometimes
wrapped in a stray `<table>` tag, or a real HTML table with `<thead>`/`<tbody>`. This
module turns either shape into the schema the challenge asks for:

    [{"text": "Position", "r": 0, "c": 0, "rs": 2, "cs": 1},
     {"text": "Delta",    "r": 0, "c": 1, "rs": 1, "cs": 3}, ...]

`r`/`c` are the top-left grid position of a cell, `rs`/`cs` its span, and every cell
of the table is written - an empty cell is still a cell. Text is normalised exactly
like the official scorer normalises it (`evaluate.norm_text`), so a JSON file scores
identically to the equivalent HTML.

Usage
    python experiments/vlm_scripts/lighton_cells.py --raw raw/lighton --out pred_lighton_json
    python experiments/vlm_scripts/lighton_cells.py --raw raw/lighton --out pred_lighton_json --check
    python experiments/vlm_scripts/lighton_cells.py --text raw/lighton/table_000.txt      # one table to stdout

`--check` reports schema violations, overlaps, interior grid holes and ragged edges;
the dev run is clean except `table_014`, where the model writes a six-label header over
a five-column body. `--fill` closes interior holes only. Measured on dev, completing
that phantom column cost final score (0.9184 -> 0.9149) instead of gaining: TEDS prices
each extra cell as an insertion, and the missing text is still missing - see the notes
in `fill_holes`.
"""

from __future__ import annotations

import argparse
import html as htmllib
import json
import re
import unicodedata
from pathlib import Path

CELL_TAG = re.compile(r"<t[dh]\b", re.I)
PIPE_SEPARATOR = re.compile(r"[\s|:\-]+")
MAX_SPAN = 100


def norm_text(t: str) -> str:
    """Same normalisation the official scorer applies (evaluate.norm_text)."""
    t = unicodedata.normalize("NFKC", t or "")
    t = t.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", t).strip()


def span(value) -> int:
    """rowspan/colspan value -> int in [1, MAX_SPAN] (evaluate.span)."""
    m = re.match(r"\s*(\d+)", str(value if value is not None else ""))
    return min(max(int(m.group(1)), 1), MAX_SPAN) if m else 1


def table_region(text: str) -> str | None:
    """The substring from the first `<table` to the last `</table>`, if any."""
    low = text.lower()
    start, end = low.find("<table"), low.rfind("</table>")
    if start >= 0 and end > start:
        return text[start : end + len("</table>")]
    return None


def html_table_to_cells(html_str: str) -> list[dict]:
    """Expand `<table>` into cells; spans footprinted so r/c stay true grid positions."""
    from lxml import html as lxml_html

    try:
        tables = lxml_html.fromstring(html_str).xpath("//table")
    except Exception:
        return []
    if not tables:
        return []

    occupied: dict[tuple[int, int], dict] = {}
    cells: list[dict] = []
    for r, tr in enumerate(tables[0].xpath(".//tr")):
        c = 0
        for node in tr.xpath("./td|./th"):
            while (r, c) in occupied:  # skip slots taken by a span above
                c += 1
            rs, cs = span(node.get("rowspan", 1)), span(node.get("colspan", 1))
            while (
                any((r + dr, c + dc) in occupied for dr in range(rs) for dc in range(cs)) and cs > 1
            ):
                cs -= 1  # a span may not overlap another cell
            cell = {"text": norm_text("".join(node.itertext())), "r": r, "c": c, "rs": rs, "cs": cs}
            for dr in range(rs):
                for dc in range(cs):
                    occupied[(r + dr, c + dc)] = cell
            cells.append(cell)
            c += cs
    return cells


def pipe_to_cells(text: str) -> list[dict]:
    """Markdown pipe table -> cells. Markdown cannot express spans, so rs/cs stay 1."""
    cells: list[dict] = []
    r = 0
    for line in text.splitlines():
        if line.count("|") < 2 or PIPE_SEPARATOR.fullmatch(line):
            continue
        fields = line.strip().strip("|").split("|")
        for c, field in enumerate(fields):
            cells.append(
                {"text": norm_text(htmllib.unescape(field)), "r": r, "c": c, "rs": 1, "cs": 1}
            )
        r += 1
    return cells


def text_to_cells(text: str) -> list[dict]:
    """Model text -> cells, whatever shape the model chose."""
    region = table_region(text)
    candidate = region if region and CELL_TAG.search(region) else text
    if CELL_TAG.search(candidate):
        cells = html_table_to_cells(candidate)
        if cells:
            return cells
    return pipe_to_cells(text)


def check_cells(cells: list[dict], max_span: int = 100) -> list[str]:
    """Schema and grid problems. Empty list = the file is a valid submission."""
    problems: list[str] = []
    if not isinstance(cells, list) or not cells:
        return ["no cells"]
    grid: dict[tuple[int, int], tuple] = {}
    rows, cols = 0, 0
    for i, cell in enumerate(cells):
        if not isinstance(cell, dict):
            problems.append(f"cell {i}: not an object")
            continue
        for key in ("text", "r", "c"):
            if key not in cell:
                problems.append(f"cell {i}: missing {key!r}")
        r, c = cell.get("r"), cell.get("c")
        rs, cs = cell.get("rs", 1), cell.get("cs", 1)
        if not all(isinstance(v, int) for v in (r, c, rs, cs)):
            problems.append(f"cell {i}: r/c/rs/cs must be integers")
            continue
        if r < 0 or c < 0:
            problems.append(f"cell {i}: negative position")
        if not 1 <= rs <= max_span or not 1 <= cs <= max_span:
            problems.append(f"cell {i}: span out of range")
        if not isinstance(cell.get("text", ""), str):
            problems.append(f"cell {i}: text must be a string")
        for dr in range(rs):
            for dc in range(cs):
                slot = (r + dr, c + dc)
                if slot in grid:
                    problems.append(f"cell {i}: overlaps cell at {slot}")
                grid[slot] = (i, dr, dc)
        rows, cols = max(rows, r + rs), max(cols, c + cs)

    for r in range(rows):
        rightmost = max((c for (rr, c) in grid if rr == r), default=None)
        for c in range(cols):
            if (r, c) in grid:
                continue
            if rightmost is None or c > rightmost:
                problems.append(f"ragged edge at ({r}, {c})")
            else:
                problems.append(f"grid hole at ({r}, {c})")
    return problems


def fill_holes(cells: list[dict]) -> tuple[list[dict], int]:
    """Close *interior* grid holes with an empty cell (README: write every cell).

    Only slots with another cell to their right in the same row are filled. A slot
    past the last cell of a row ("ragged edge") is not: that is not a dropped cell,
    it is a row that is genuinely narrower - usually because the model invented a
    column in the header, and completing that phantom column only adds nodes the
    ground truth does not have, which TEDS charges as insertions.
    """
    occupied: dict[tuple[int, int], None] = {}
    rows = cols = 0
    for cell in cells:
        r, c = int(cell["r"]), int(cell["c"])
        rs, cs = int(cell.get("rs", 1)), int(cell.get("cs", 1))
        for dr in range(rs):
            for dc in range(cs):
                occupied[(r + dr, c + dc)] = None
        rows, cols = max(rows, r + rs), max(cols, c + cs)

    filled = [dict(cell) for cell in cells]
    added = 0
    for r in range(rows):
        rightmost = max((c for (rr, c) in occupied if rr == r), default=None)
        if rightmost is None:
            continue
        for c in range(cols):
            if c < rightmost and (r, c) not in occupied:
                filled.append({"text": "", "r": r, "c": c, "rs": 1, "cs": 1})
                added += 1
    filled.sort(key=lambda cell: (cell["r"], cell["c"]))
    return filled, added


def write_predictions(raw_dir: Path, out_dir: Path, check: bool = False, fill: bool = False) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written, flagged = 0, 0
    for path in sorted(raw_dir.glob("*.txt")):
        cells = text_to_cells(path.read_text())
        target = out_dir / f"{path.stem}.json"
        if not cells:
            print(f"{path.stem}: EMPTY - no table found, no file written")
            continue
        added = 0
        if fill:
            cells, added = fill_holes(cells)
        target.write_text(json.dumps(cells, ensure_ascii=False, indent=1))
        written += 1
        if check:
            problems = check_cells(cells)
            holes = [p for p in problems if "grid hole" in p]
            ragged = [p for p in problems if "ragged edge" in p]
            hard = [p for p in problems if "grid hole" not in p and "ragged edge" not in p]
            if hard or holes:
                flagged += 1
                bits = []
                if hard:
                    bits.append("; ".join(hard[:3]))
                if holes:
                    bits.append(f"{len(holes)} interior holes")
                print(f"{path.stem}: {len(cells)} cells, " + ", ".join(bits))
            elif ragged:
                print(
                    f"{path.stem}: {len(cells)} cells, ragged edges "
                    f"({len(ragged)} slots, left alone)"
                )
            elif added:
                print(f"{path.stem}: {len(cells)} cells ({added} empty cells added)")
    print(
        f"{written} JSON predictions -> {out_dir}"
        f"{f', {flagged} with warnings' if check and flagged else ''}"
    )
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", help="directory with <table_id>.txt model output")
    ap.add_argument("--out", help="directory for <table_id>.json")
    ap.add_argument("--text", help="convert a single text file to stdout")
    ap.add_argument(
        "--check", action="store_true", help="validate every file against the schema and the grid"
    )
    ap.add_argument(
        "--fill", action="store_true", help="write an empty cell for every uncovered grid slot"
    )
    args = ap.parse_args()

    if args.text:
        cells = text_to_cells(Path(args.text).read_text())
        if args.fill:
            cells, _ = fill_holes(cells)
        print(json.dumps(cells, ensure_ascii=False, indent=1))
        problems = check_cells(cells)
        if problems:
            print(f"\n# {len(problems)} problem(s): " + "; ".join(problems[:5]), flush=True)
        return
    if not (args.raw and args.out):
        ap.error("pass --raw and --out, or --text")
    write_predictions(Path(args.raw), Path(args.out), check=args.check, fill=args.fill)


if __name__ == "__main__":
    main()
