"""Shared conversion + sample writing for real-table evaluation."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image

TD_RE = re.compile(r"^<(td|th)(\s[^>]*)?>$", re.I)
SPAN_RE = re.compile(r'(rowspan|colspan)\s*=\s*"?(\d+)"?', re.I)
SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


def safe_id(raw: str, prefix: str = "", maxlen: int = 80) -> str:
    stem = SAFE_ID.sub("_", raw).strip("._-")
    if prefix:
        stem = f"{prefix}_{stem}" if stem else prefix
    return (stem or "table")[:maxlen]


def norm_ws(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", t).strip()


def tokens_to_text(tokens: Any) -> str:
    """Join cell tokens. PubTabNet/OTSL is character-level plus optional <b> tags."""
    if tokens is None:
        return ""
    if isinstance(tokens, str):
        s = tokens
    elif isinstance(tokens, (list, tuple)):
        if not tokens:
            return ""
        if all(isinstance(t, str) for t in tokens):
            n1 = sum(1 for t in tokens if len(t) <= 1)
            # character-level (possibly mixed with tags like '<b>') vs word chunks
            if n1 >= max(1, int(0.6 * len(tokens))):
                s = "".join(tokens)
            else:
                s = " ".join(tokens)
        else:
            s = " ".join(tokens_to_text(t) for t in tokens)
    else:
        s = str(tokens)
    s = re.sub(r"</?(b|i|u|sup|sub|span)[^>]*>", "", s, flags=re.I)
    return s


def flatten_cell_list(cells: Any) -> list[Any]:
    """OTSL 'cells' is sometimes [[cell, cell, ...]]."""
    if not cells:
        return []
    if isinstance(cells, list) and cells and isinstance(cells[0], list):
        if cells and isinstance(cells[0][0], dict) if cells[0] else False:
            out: list[Any] = []
            for block in cells:
                out.extend(block)
            return out
    return list(cells)


def fill_empty_cells(cells: list[dict]) -> list[dict]:
    if not cells:
        return []
    out = []
    occupied: set[tuple[int, int]] = set()
    for cell in cells:
        r, c = int(cell["r"]), int(cell["c"])
        rs = max(int(cell.get("rs", 1) or 1), 1)
        cs = max(int(cell.get("cs", 1) or 1), 1)
        out.append({"text": cell.get("text", ""), "r": r, "c": c, "rs": rs, "cs": cs})
        for dr in range(rs):
            for dc in range(cs):
                occupied.add((r + dr, c + dc))
    max_r = max(r for r, _ in occupied) + 1
    max_c = max(c for _, c in occupied) + 1
    for r in range(max_r):
        for c in range(max_c):
            if (r, c) not in occupied:
                out.append({"text": "", "r": r, "c": c, "rs": 1, "cs": 1})
    out.sort(key=lambda x: (x["r"], x["c"]))
    return out


def cells_to_html(cells: Iterable[dict]) -> str:
    def esc(t: Any) -> str:
        return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rows: dict[int, list[dict]] = {}
    for cell in cells:
        rows.setdefault(int(cell["r"]), []).append(cell)
    parts = ["<table>"]
    for r in sorted(rows):
        parts.append("<tr>")
        for cell in sorted(rows[r], key=lambda x: int(x["c"])):
            rs = max(int(cell.get("rs", 1) or 1), 1)
            cs = max(int(cell.get("cs", 1) or 1), 1)
            attrs = (f' rowspan="{rs}"' if rs > 1 else "") + (f' colspan="{cs}"' if cs > 1 else "")
            parts.append(f"<td{attrs}>{esc(cell.get('text', ''))}</td>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def normalize_html_tokens(tokens: Iterable[Any]) -> list[str]:
    """Join PubTabNet-style split tags: ['<td', ' colspan=\"2\"', '>']."""
    out: list[str] = []
    buf = ""
    for raw in tokens:
        t = str(raw)
        if buf:
            buf += t
            if ">" in t:
                out.append(buf)
                buf = ""
            continue
        if t.startswith("<") and not t.endswith(">") and "</" not in t:
            buf = t
            continue
        out.append(t)
    if buf:
        out.append(buf)
    return out


def html_tokens_to_cells(html_tokens: Iterable[Any], cell_items: Any) -> list[dict]:
    """Build {text,r,c,rs,cs} from structure tokens + aligned cell texts."""
    texts: list[str] = []
    for item in flatten_cell_list(cell_items):
        if isinstance(item, dict):
            texts.append(tokens_to_text(item.get("tokens", item.get("text", ""))))
        else:
            texts.append(tokens_to_text(item))

    cells: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    r = 0
    c = 0
    idx = 0
    for tok in normalize_html_tokens(html_tokens):
        low = tok.lower().strip()
        if low in {
            "<thead>",
            "</thead>",
            "<tbody>",
            "</tbody>",
            "<tfoot>",
            "</tfoot>",
            "<table>",
            "</table>",
        }:
            continue
        if low == "<tr>":
            c = 0
            continue
        if low == "</tr>":
            r += 1
            c = 0
            continue
        if low in ("</td>", "</th>"):
            continue
        m = TD_RE.match(tok.strip())
        if not m:
            continue
        while (r, c) in occupied:
            c += 1
        rs, cs = 1, 1
        for am in SPAN_RE.finditer(tok):
            if am.group(1).lower() == "rowspan":
                rs = max(int(am.group(2)), 1)
            else:
                cs = max(int(am.group(2)), 1)
        text = texts[idx] if idx < len(texts) else ""
        idx += 1
        cells.append({"text": text, "r": r, "c": c, "rs": rs, "cs": cs})
        for dr in range(rs):
            for dc in range(cs):
                occupied.add((r + dr, c + dc))
        c += cs

    if not cells:
        raise ValueError("no cells parsed from HTML tokens")
    if idx != len(texts):
        # If we consumed too few texts, the structure/text alignment is unsafe.
        if idx < len(texts) * 0.9:
            raise ValueError(f"cell/text mismatch: used {idx} of {len(texts)} texts")
    return fill_empty_cells(cells)


def scitsr_to_cells(raw_cells: list[dict]) -> list[dict]:
    cells = []
    for item in raw_cells:
        r0 = int(item["start_row"])
        r1 = int(item["end_row"])
        c0 = int(item["start_col"])
        c1 = int(item["end_col"])
        text = tokens_to_text(item.get("content", item.get("tex", "")))
        cells.append(
            {
                "text": text,
                "r": r0,
                "c": c0,
                "rs": max(r1 - r0 + 1, 1),
                "cs": max(c1 - c0 + 1, 1),
            }
        )
    return fill_empty_cells(cells)


def _points_bbox(points: str) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for part in points.strip().split():
        x, y = part.split(",")
        xs.append(float(x))
        ys.append(float(y))
    return min(xs), min(ys), max(xs), max(ys)


def _bbox(el) -> tuple[float, float, float, float]:
    return (
        float(el.findtext("xmin")),
        float(el.findtext("ymin")),
        float(el.findtext("xmax")),
        float(el.findtext("ymax")),
    )


def _overlap_1d(a1: float, a2: float, b1: float, b2: float) -> float:
    inter = max(0.0, min(a2, b2) - max(a1, b1))
    denom = min(a2 - a1, b2 - b1)
    return inter / denom if denom > 1e-6 else 0.0


def icdar2013_voc_to_cells(
    xml_path: Path, words: list[dict], iob_thresh: float = 0.5
) -> list[dict]:
    """Convert TATR/Pascal-VOC structure XML + word boxes into cell JSON."""
    from lxml import etree

    root = etree.parse(str(xml_path)).getroot()
    rows, cols, spans = [], [], []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        box = _bbox(obj.find("bndbox"))
        if name == "table row":
            rows.append(box)
        elif name == "table column":
            cols.append(box)
        elif name == "table spanning cell":
            spans.append(box)
    if not rows or not cols:
        raise ValueError(f"missing rows/cols in {xml_path.name}")
    rows.sort(key=lambda b: (b[1] + b[3]) / 2)
    cols.sort(key=lambda b: (b[0] + b[2]) / 2)

    def covered_indices(box, lines, axis: str):
        out = []
        for i, line in enumerate(lines):
            if axis == "row":
                o = _overlap_1d(box[1], box[3], line[1], line[3])
            else:
                o = _overlap_1d(box[0], box[2], line[0], line[2])
            if o >= iob_thresh:
                out.append(i)
        return out

    occupied: set[tuple[int, int]] = set()
    cells: list[dict] = []

    def add_cell(r0, r1, c0, c1):
        bbox = (cols[c0][0], rows[r0][1], cols[c1][2], rows[r1][3])
        cells.append(
            {
                "text": _words_in_bbox(words, bbox),
                "r": r0,
                "c": c0,
                "rs": r1 - r0 + 1,
                "cs": c1 - c0 + 1,
                "_bbox": bbox,
            }
        )
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                occupied.add((r, c))

    for sp in spans:
        rs = covered_indices(sp, rows, "row")
        cs = covered_indices(sp, cols, "col")
        if not rs or not cs:
            continue
        add_cell(min(rs), max(rs), min(cs), max(cs))

    for r in range(len(rows)):
        for c in range(len(cols)):
            if (r, c) not in occupied:
                add_cell(r, r, c, c)

    for cell in cells:
        cell.pop("_bbox", None)
    cells.sort(key=lambda x: (x["r"], x["c"]))
    # footprints should cover the whole grid
    return fill_empty_cells(cells)


def _words_in_bbox(words: list[dict], bbox: tuple[float, float, float, float]) -> str:
    x0, y0, x1, y1 = bbox
    pad = 1.0
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    picked = []
    for i, w in enumerate(words):
        b = w.get("bbox") or w.get("box")
        if not b or len(b) < 4:
            continue
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            picked.append(
                (
                    w.get("block_num", 0),
                    w.get("line_num", 0),
                    w.get("span_num", i),
                    cy,
                    cx,
                    w.get("text") or "",
                )
            )
    picked.sort()
    return norm_ws(" ".join(t[-1] for t in picked if t[-1]))


def grid_stats(cells: list[dict]) -> dict:
    if not cells:
        return {"n_cells": 0, "n_rows": 0, "n_cols": 0, "n_spans": 0}
    n_rows = max(c["r"] + c.get("rs", 1) for c in cells)
    n_cols = max(c["c"] + c.get("cs", 1) for c in cells)
    n_spans = sum(1 for c in cells if c.get("rs", 1) > 1 or c.get("cs", 1) > 1)
    return {
        "n_cells": len(cells),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_spans": n_spans,
    }


def maybe_resize(img: Image.Image, max_side: int = 2000) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / m
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)


def crop_to_bbox(
    img: Image.Image,
    bbox: tuple[float, float, float, float],
    pad: int = 8,
) -> Image.Image:
    w, h = img.size
    x0, y0, x1, y1 = bbox
    x0 = max(0, int(x0) - pad)
    y0 = max(0, int(y0) - pad)
    x1 = min(w, int(x1) + pad)
    y1 = min(h, int(y1) + pad)
    if x1 <= x0 or y1 <= y0:
        return img
    return img.crop((x0, y0, x1, y1))


def write_sample(
    dataset_dir: Path,
    tid: str,
    image: Image.Image,
    cells: list[dict],
    *,
    style: str,
    extra: dict | None = None,
) -> dict:
    cells = fill_empty_cells(cells)
    stats = grid_stats(cells)
    table_dir = dataset_dir / tid
    gt_dir = table_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image = maybe_resize(image, 2000)
    img_name = "image.png"
    image.save(table_dir / img_name)
    image.save(images_dir / f"{tid}.png")

    html = cells_to_html(cells)
    (gt_dir / "table.html").write_text(html, encoding="utf-8")
    (gt_dir / "cells.json").write_text(
        json.dumps(cells, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    entry = {
        "id": tid,
        "image": f"{tid}/{img_name}",
        "style": style,
        "tier": "real",
        "has_spans": stats["n_spans"] > 0,
        "n_rows": stats["n_rows"],
        "n_cols": stats["n_cols"],
        "n_cells": stats["n_cells"],
        "img_w": image.size[0],
        "img_h": image.size[1],
    }
    if extra:
        entry.update(extra)
    return entry


def write_manifest(dataset_dir: Path, entries: list[dict]) -> None:
    (dataset_dir / "manifest.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_source(dataset_dir: Path, info: dict) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "SOURCE.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
