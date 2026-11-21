#!/usr/bin/env python3
"""Table Transformer structure on every dev table — for human inspection.

Same boxes/scores as the one-off table_008 snippet, but over the whole set,
with expected row/column/span counts from gt/cells.json next to the findings.

    .venv-tatr/bin/python scripts/tatr_visualize.py
    open /tmp/tatr_viz/index.html

Output: one annotated PNG per table, a contact sheet (all.png) and a
self-contained index.html with original + overlay side by side, failures first.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import TableTransformerForObjectDetection

COLORS = {
    "table row": "blue",
    "table column": "green",
    "table spanning cell": "red",
    "table column header": "orange",
    "table projected row header": "purple",
}


def data_uri(path: Path) -> str:
    suffix = path.suffix.lstrip(".") or "png"
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def detect(model, img: Image.Image, threshold: float) -> tuple[Image.Image, dict]:
    w, h = img.size
    arr = (np.asarray(img, np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = torch.from_numpy(arr.transpose(2, 0, 1))[None].float()
    with torch.no_grad():
        out = model(x)

    scores, labels = out.logits.softmax(-1)[0].max(-1)
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    counts: dict[str, int] = {}
    confidences: list[float] = []

    for s, label, (cx, cy, bw, bh) in zip(
        scores.tolist(), labels.tolist(), out.pred_boxes[0].tolist()
    ):
        name = model.config.id2label.get(label, "no object")
        if s < threshold or name not in COLORS:
            continue
        counts[name] = counts.get(name, 0) + 1
        confidences.append(s)
        draw.rectangle(
            [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
            outline=COLORS[name],
            width=3,
        )

    info = {
        "counts": counts,
        "conf_min": round(min(confidences), 3) if confidences else 0.0,
        "conf_mean": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
    }
    return overlay, info


def verdict(found: int, expected: int) -> str:
    if found == expected:
        return "OK"
    if abs(found - expected) <= 1:
        return "WARN"
    return "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/dev")
    ap.add_argument("--out", default="/tmp/tatr_viz")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    gt_dir, out_dir = Path(args.gt), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((gt_dir / "manifest.json").read_text())
    entries = manifest[: args.limit or None]

    model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-structure-recognition", local_files_only=True
    ).eval()

    cards, results = [], []
    t0 = time.time()
    for i, entry in enumerate(entries, 1):
        tid = entry["id"]
        img_path = gt_dir / entry["image"]
        try:
            cells = json.loads((gt_dir / tid / "gt" / "cells.json").read_text())
            n_spans = sum(1 for c in cells if c.get("rs", 1) > 1 or c.get("cs", 1) > 1)
            img = Image.open(img_path).convert("RGB")
            overlay, info = detect(model, img, args.threshold)
            overlay_path = out_dir / f"{tid}.png"
            overlay.save(overlay_path)

            counts = info["counts"]
            rows, cols = counts.get("table row", 0), counts.get("table column", 0)
            spans = counts.get("table spanning cell", 0)
            status = verdict(rows, entry["n_rows"])
            status = "FAIL" if status == "FAIL" or verdict(cols, entry["n_cols"]) == "FAIL" else status
            if entry.get("has_spans") and spans == 0 and status == "OK":
                status = "WARN"
            results.append((status, tid, entry["style"], entry["tier"], entry["n_rows"], rows,
                            entry["n_cols"], cols, n_spans, spans, info["conf_mean"]))
            cards.append((status, tid, entry, info, img_path, overlay_path, n_spans))
            print(f"[{i}/{len(entries)}] {tid} {entry['style']:11s} {entry['tier']} "
                  f"rows {rows}/{entry['n_rows']} cols {cols}/{entry['n_cols']} "
                  f"spans {spans}/{n_spans} -> {status}", flush=True)
        except Exception as exc:  # never crash; keep the other tables
            print(f"[{i}/{len(entries)}] {tid} ERROR {type(exc).__name__}: {exc}", flush=True)
            results.append(("ERROR", tid, "?", "?", 0, 0, 0, 0, 0, 0, 0.0))

    # Contact sheet: all overlays at a glance, failures first.
    order = {"FAIL": 0, "ERROR": 0, "WARN": 1, "OK": 2}
    cards.sort(key=lambda c: (order.get(c[0], 3), c[1]))
    thumb_w = 320
    cols_sheet = 4
    thumb_h = 260
    rows_sheet = (len(cards) + cols_sheet - 1) // cols_sheet
    sheet = Image.new("RGB", (cols_sheet * thumb_w, rows_sheet * thumb_h), "white")
    sd = ImageDraw.Draw(sheet)
    for idx, (status, tid, _entry, _info, _img, overlay, _spans) in enumerate(cards):
        thumb = Image.open(overlay).convert("RGB")
        scale = thumb_w / thumb.width
        thumb = thumb.resize((thumb_w, max(1, int(thumb.height * scale))))
        x, y = (idx % cols_sheet) * thumb_w, (idx // cols_sheet) * thumb_h
        sheet.paste(thumb, (x, y + 18))
        sd.text((x + 4, y + 3), f"{tid}  {status}", fill="black")
    sheet_path = out_dir / "all.png"
    sheet.save(sheet_path)

    # Self-contained HTML: original | overlay, expected | found, failures first.
    parts = [
        "<!doctype html><meta charset='utf-8'><title>TATR structure — dev</title>",
        "<style>body{font:14px sans-serif;background:#eee;margin:20px}",
        ".card{background:white;margin:12px 0;padding:12px;border-radius:6px}",
        ".row{display:flex;gap:12px;align-items:flex-start}",
        "img{width:480px;max-height:520px;object-fit:contain;border:1px solid #ccc}",
        "table{border-collapse:collapse}td,th{padding:2px 10px;border:1px solid #ddd}",
        ".FAIL,.ERROR{color:#b00;font-weight:bold}.WARN{color:#a60;font-weight:bold}",
        ".OK{color:#080;font-weight:bold}.legend span{padding:0 12px}</style>",
        "<h1>Table Transformer structure — dev</h1>",
        "<p class='legend'><span style='color:blue'>row</span>",
        "<span style='color:green'>column</span>",
        "<span style='color:red'>spanning cell</span>",
        "<span style='color:orange'>column header</span>",
        "<span style='color:purple'>projected row header</span></p>",
        f"<p><a href='all.png'>contact sheet</a> · {len(cards)} tables · "
        f"{time.time() - t0:.1f}s total</p>",
        f"<img src='{data_uri(sheet_path)}' style='width:1280px'><hr>",
    ]
    for status, tid, entry, info, img_path, overlay_path, n_spans in cards:
        counts = info["counts"]
        rows, cols = counts.get("table row", 0), counts.get("table column", 0)
        spans = counts.get("table spanning cell", 0)
        parts.append(
            f"<div class='card'><h2>{html.escape(tid)} — "
            f"<span class='{status}'>{status}</span> "
            f"<small>{entry['style']} / {entry['tier']}</small></h2>"
            "<div class='row'>"
            f"<div>original<br><img src='{data_uri(img_path)}'></div>"
            f"<div>overlays<br><img src='{data_uri(overlay_path)}'></div>"
            "<div><table><tr><th>class</th><th>expected</th><th>found</th></tr>"
            f"<tr><td>rows</td><td>{entry['n_rows']}</td><td>{rows}</td></tr>"
            f"<tr><td>columns</td><td>{entry['n_cols']}</td><td>{cols}</td></tr>"
            f"<tr><td>spans</td><td>{n_spans}</td><td>{spans}</td></tr>"
            f"<tr><td>column headers</td><td>–</td><td>{counts.get('table column header', 0)}</td></tr>"
            f"<tr><td>row headers</td><td>–</td><td>{counts.get('table projected row header', 0)}</td></tr>"
            "</table>"
            f"<p>mean conf {info['conf_mean']} · min {info['conf_min']}</p></div>"
            "</div></div>"
        )
    (out_dir / "index.html").write_text("".join(parts), encoding="utf-8")

    print(f"\n{len(cards)} tables in {time.time() - t0:.1f}s -> {out_dir}/index.html")
    print("FAIL/ERROR: " + ", ".join(r[1] for r in results if r[0] in ("FAIL", "ERROR")))
    print("WARN:      " + ", ".join(r[1] for r in results if r[0] == "WARN"))
    print("OK:        " + ", ".join(r[1] for r in results if r[0] == "OK"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
