"""Model output -> HTML table, per model format.

Every model writes its own dialect. This module is the only place that knows
about dialects, so a new model needs one registry row (fmt) and nothing else.

Formats
    otsl      PaddleOCR-VL / MinerU / DocTags: <fcel> <lcel> <ucel> <xcel> <ecel> <nl>
    latex     GOT-OCR2, olmOCR-ish: \\begin{tabular}{...} a & b \\\\ ...
    markdown  generic VLMs: | a | b |
    doctags   granite-docling / SmolDocling document tags with <otsl>
    html      dots.ocr, Nanonets, most general VLMs
    auto      try html, then otsl, then markdown
"""
from __future__ import annotations

import html as htmllib
import re

TAG = re.compile(r"<(fcel|ecel|ched|lcel|ucel|xcel|nl)>")
LOC = re.compile(r"<loc_\d+>|<[a-z_]+_\d+>")
CELL = re.compile(r"<t[dh]\b", re.I)


# ---------------------------------------------------------------- cell builders
def otsl_to_cells(text: str) -> list[dict]:
    parts = TAG.split(text)
    cells: list[dict] = []
    cover: dict[tuple[int, int], dict] = {}
    r = c = 0
    for i in range(1, len(parts) - 1, 2):
        tag, body = parts[i], parts[i + 1]
        if tag == "nl":
            r, c = r + 1, 0
        elif tag in ("fcel", "ecel", "ched"):
            cell = {"text": "" if tag == "ecel" else body.strip(), "r": r, "c": c,
                    "rs": 1, "cs": 1}
            cells.append(cell)
            cover[(r, c)] = cell
            c += 1
        elif tag == "lcel":
            left = cover.get((r, c - 1))
            if left is not None and left["r"] == r:
                left["cs"] += 1
                cover[(r, c)] = left
            c += 1
        elif tag == "ucel":
            up = cover.get((r - 1, c))
            if up is not None and up["c"] == c:
                up["rs"] += 1
                cover[(r, c)] = up
            c += 1
        elif tag == "xcel":
            up = cover.get((r - 1, c - 1))
            if up is not None:
                up["rs"] += 1
                up["cs"] += 1
                cover[(r, c)] = up
            c += 1
    return cells


def pipe_to_cells(text: str) -> list[dict]:
    cells: list[dict] = []
    r = 0
    for line in text.splitlines():
        if line.count("|") < 2:
            continue
        if re.fullmatch(r"[\s|:\-]+", line):
            continue
        for c, field in enumerate(line.strip().strip("|").split("|")):
            cells.append({"text": field.strip(), "r": r, "c": c, "rs": 1, "cs": 1})
        r += 1
    return cells


RULE = re.compile(r"\\(hline|toprule|midrule|bottomrule|cline\{[^}]*\}|cmidrule(\([^)]*\))?\{[^}]*\})")


def _clean_latex(text: str) -> str:
    text = RULE.sub("", text)
    text = text.replace("\\$", "$").replace("~", " ")
    text = re.sub(r"\\(?:textbf|textit|emph|mathrm|text)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[,;:! ]", " ", text)
    text = text.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
    return text.strip()


def latex_to_cells(text: str) -> list[dict]:
    body = text
    start = body.find("\\begin{tabular}")
    if start >= 0:
        brace = body.find("{", start + len("\\begin{tabular}"))
        depth, i = 0, brace
        while i < len(body):
            if body[i] == "{":
                depth += 1
            elif body[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        stop = body.find("\\end{tabular}", i)
        body = body[i + 1:stop if stop > 0 else len(body)]
    rows = re.split(r"\\\\", body)
    cells: list[dict] = []
    spans: dict[tuple[int, int], dict] = {}
    for r, row in enumerate(rows):
        row = row.strip()
        if not row or row in ("\\hline",):
            continue
        c = 0
        for field in row.split("&"):
            field = field.strip()
            cs = rs = 1
            multi_col = re.search(r"\\multicolumn\{(\d+)\}", field)
            if multi_col:
                cs = int(multi_col.group(1))
                field = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", "", field)
            multi_row = re.search(r"\\multirow\{(\d+)\}", field)
            if multi_row:
                rs = int(multi_row.group(1))
                field = re.sub(r"\\multirow\{\d+\}\{[^}]*\}", "", field)
            while (r, c) in spans:
                c += 1
            if cs == 1 and rs == 1 and not _clean_latex(field):
                c += 1
                continue
            cell = {"text": _clean_latex(field), "r": r, "c": c, "rs": rs, "cs": cs}
            cells.append(cell)
            for dr in range(rs):
                for dc in range(cs):
                    spans[(r + dr, c + dc)] = cell
            c += cs
    return cells


def doctags_to_cells(text: str) -> list[dict]:
    cleaned = LOC.sub("", text)
    cleaned = cleaned.replace("<doctag>", "").replace("</doctag>", "")
    if "<otsl>" in cleaned:
        cleaned = cleaned.split("<otsl>", 1)[1]
    return otsl_to_cells(cleaned)


def _has_pipe_table(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.count("|") >= 2]
    return len(lines) >= 2 and any(re.fullmatch(r"[\s|:\-]+", line) for line in lines)


def first_html_table(text: str) -> str | None:
    lower = text.lower()
    start = lower.find("<table")
    end = lower.rfind("</table>")
    if start >= 0 and end > start:
        candidate = text[start:end + len("</table>")]
        if CELL.search(candidate):          # real cells, not a <table> wrapper around markdown
            return candidate
    # a markdown pipe table is more trustworthy than half a <table> tag
    if _has_pipe_table(text):
        return None
    # some VLMs emit bare <tr> rows without the enclosing <table>
    if CELL.search(text):
        row_start = lower.find("<tr")
        row_end = lower.rfind("</tr>")
        if row_start >= 0 and row_end > row_start:
            return "<table>" + text[row_start:row_end + len("</tr>")] + "</table>"
    return None


def cells_to_html(cells: list[dict]) -> str:
    out = ["<table>"]
    rows: dict[int, list[dict]] = {}
    for cell in cells:
        rows.setdefault(cell["r"], []).append(cell)
    for r in sorted(rows):
        out.append("<tr>")
        for cell in sorted(rows[r], key=lambda x: x["c"]):
            attrs = ""
            if cell["rs"] > 1:
                attrs += f' rowspan="{cell["rs"]}"'
            if cell["cs"] > 1:
                attrs += f' colspan="{cell["cs"]}"'
            out.append(f"<td{attrs}>{htmllib.escape(cell['text'])}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


# -------------------------------------------------------------------- dispatch
PARSERS = {
    "otsl": otsl_to_cells,
    "latex": latex_to_cells,
    "markdown": pipe_to_cells,
    "md": pipe_to_cells,
    "doctags": doctags_to_cells,
}


def convert(text: str, fmt: str = "auto") -> tuple[str, dict]:
    """-> (html_table, info). html is '' when nothing table-like was produced."""
    fmt = (fmt or "auto").lower()
    tables = first_html_table(text or "")
    if fmt == "html" or (fmt == "auto" and tables):
        html = tables or ""
        cells = _count_html_cells(html)
        return html, {"fmt": "html", "cells": cells, "empty_rate": 0.0, "parser": "html"}

    parser = PARSERS.get(fmt)
    tried = fmt
    if parser is None:  # auto
        for name in ("otsl", "html", "latex", "markdown"):
            if name == "html":
                continue
            cells = PARSERS[name](text or "")
            if len(cells) >= 2:
                return _finish(cells, name)
        cells = pipe_to_cells(text or "")
        return _finish(cells, "markdown")

    cells = parser(text or "")
    if fmt == "doctags" and len(cells) < 2 and tables:
        return tables, {"fmt": "doctags", "cells": _count_html_cells(tables),
                        "empty_rate": 0.0, "parser": "html-fallback"}
    return _finish(cells, tried)


def _finish(cells: list[dict], parser: str) -> tuple[str, dict]:
    if not cells:
        return "", {"fmt": parser, "cells": 0, "empty_rate": 1.0, "parser": parser}
    empty = sum(1 for c in cells if not c["text"].strip())
    return cells_to_html(cells), {
        "fmt": parser,
        "cells": len(cells),
        "empty_rate": round(empty / len(cells), 3),
        "parser": parser,
    }


def _count_html_cells(html: str) -> int:
    return len(re.findall(r"<t[dh]\b", html, re.I))
