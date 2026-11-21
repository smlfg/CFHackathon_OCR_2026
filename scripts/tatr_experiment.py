#!/usr/bin/env python3
"""TATR experiment (see challenge/TATR_RESEARCH.md §6).

Measures the Table Transformer structure model on TableSalt dev, without any
text model:

  A  as-is            : structure out of the box
  B  deskewed         : same, after projection-profile deskew
  C  oracle text      : TATR cells filled with ground-truth text -> upper bound
                        of the structure path

Text is never predicted here. `--text oracle` only copies GT text into TATR's
grid so TEDS measures cell assignment instead of OCR.

  python scripts/tatr_experiment.py --gt data/dev --out out/tatr_asitis
  python scripts/tatr_experiment.py --gt data/dev --out out/tatr_deskew --deskew
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import TableTransformerForObjectDetection

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import evaluate  # noqa: E402  (repo scorer, reused for TEDS)

SCORE_THRESHOLD = 0.5


# ---------------------------------------------------------------- preprocessing
def max_resize(img: Image.Image, max_size: int) -> Image.Image:
    w, h = img.size
    scale = max_size / max(w, h)
    if scale >= 1:
        return img
    return img.resize((int(round(scale * w)), int(round(scale * h))))


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(arr.transpose(2, 0, 1))[None]


def deskew(img: Image.Image, max_angle: float = 12.0) -> tuple[Image.Image, float]:
    """Rotate so the horizontal projection profile is sharpest (min entropy).

    Axis-aligned row boxes are the core assumption of TATR's cell derivation, so
    removing global skew is the minimum honest preprocessing before measuring.
    """
    gray = np.asarray(img.convert("L").resize((img.width // 2 or 1, img.height // 2 or 1)),
                      dtype=np.float32)
    ink = (255.0 - gray) / 255.0  # ink = 1
    best, best_angle = None, 0.0
    for angle in np.arange(-max_angle, max_angle + 0.5, 1.0):
        rot = np.asarray(Image.fromarray(ink).rotate(
            float(angle), resample=Image.BILINEAR, expand=False, fillcolor=0.0))
        profile = rot.sum(axis=1)
        p = profile / (profile.sum() or 1.0)
        entropy = -(p * np.log(p + 1e-12)).sum()
        if best is None or entropy < best:
            best, best_angle = entropy, float(angle)
    if abs(best_angle) < 0.5:
        return img, 0.0
    return img.rotate(best_angle, resample=Image.BICUBIC, expand=True,
                      fillcolor=(255, 255, 255)), best_angle


# ------------------------------------------------------------------ model -> boxes
def boxes_from_outputs(outputs, size, id2label):
    probs = outputs.logits.softmax(-1)[0]
    scores, labels = probs.max(-1)
    boxes = outputs.pred_boxes[0].detach().cpu().numpy()  # cxcywh, normalized
    w, h = size
    cx, cy, bw, bh = boxes[:, 0] * w, boxes[:, 1] * h, boxes[:, 2] * w, boxes[:, 3] * h
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    out = []
    for (x0, y0, x1, y1), score, label in zip(xyxy, scores.tolist(), labels.tolist()):
        name = id2label.get(int(label), "no object")
        if name == "no object" or score < SCORE_THRESHOLD:
            continue
        out.append({"label": name, "score": float(score),
                    "bbox": [float(x0), float(y0), float(x1), float(y1)]})
    return out


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def overlap_frac(inner, outer):
    """share of `inner` covered by `outer`"""
    if area(inner) <= 0:
        return 0.0
    return area(inter(inner, outer)) / area(inner)


def iou(a, b):
    i = inter(a, b)
    ai, au = area(i), area(a) + area(b) - area(i)
    return ai / au if au > 0 else 0.0


def nms(objects, labels, threshold):
    """Greedy NMS per class. TATR's own postprocess only de-duplicates headers
    and spanning cells; off-distribution (degraded) input makes the detector
    emit 2-4 near-duplicates of the same row/column, which then explodes the
    row x column grid."""
    if threshold <= 0:
        return objects
    out = []
    for label in labels:
        group = sorted([o for o in objects if o["label"] == label],
                       key=lambda o: -o["score"])
        keep = []
        for cand in group:
            if all(iou(cand["bbox"], k["bbox"]) < threshold for k in keep):
                keep.append(cand)
        out.extend(keep)
    out.extend([o for o in objects if o["label"] not in labels])
    return out


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


# ------------------------------------------------------- boxes -> grid -> cells
def structure_to_cells(objects, no_spans=False):
    rows = sorted([o for o in objects if o["label"] == "table row"], key=lambda o: o["bbox"][1])
    cols = sorted([o for o in objects if o["label"] == "table column"], key=lambda o: o["bbox"][0])
    spans = [] if no_spans else [o for o in objects
                                 if o["label"] in ("table spanning cell", "table projected row header")]
    header_objs = [o for o in objects if o["label"] == "table column header"]

    header_rows = set()
    for r, row in enumerate(rows):
        h = max(1e-6, row["bbox"][3] - row["bbox"][1])
        for hd in header_objs:
            i = inter(row["bbox"], hd["bbox"])
            if (i[3] - i[1]) / h >= 0.5:
                header_rows.add(r)

    cells, subcells = [], []
    for r, row in enumerate(rows):
        for c, col in enumerate(cols):
            rect = inter(row["bbox"], col["bbox"])
            if area(rect) <= 0:
                continue
            cell = {"bbox": rect, "rows": {r}, "cols": {c}, "header": r in header_rows}
            if any(overlap_frac(rect, s["bbox"]) > 0.5 for s in spans):
                subcells.append(cell)
            else:
                cells.append(cell)

    for s in spans:
        covered = [sc for sc in subcells if overlap_frac(sc["bbox"], s["bbox"]) > 0.5]
        if not covered:
            continue
        rows_hit = set().union(*(sc["rows"] for sc in covered))
        cols_hit = set().union(*(sc["cols"] for sc in covered))
        cells.append({
            "bbox": union([sc["bbox"] for sc in covered]),
            "rows": rows_hit,
            "cols": cols_hit,
            "header": all(sc["header"] for sc in covered),
        })

    out = []
    for cell in sorted(cells, key=lambda x: (min(x["rows"]), min(x["cols"]))):
        out.append({
            "r": min(cell["rows"]), "c": min(cell["cols"]),
            "rs": len(cell["rows"]), "cs": len(cell["cols"]),
            "header": bool(cell["header"]),
            "bbox": [round(v, 1) for v in cell["bbox"]],
        })
    return out, len(rows), len(cols), len(spans)


def gt_grid(gt_cells):
    grid, text = {}, {}
    for c in gt_cells:
        for dr in range(int(c.get("rs", 1))):
            for dc in range(int(c.get("cs", 1))):
                grid[(int(c["r"]) + dr, int(c["c"]) + dc)] = c
    for k, c in grid.items():
        text[k] = c.get("text", "")
    return grid, text


def ocr_cells(img, cells):
    """Realistic end-to-end: TATR grid + plain Tesseract per cell crop."""
    import pytesseract

    for c in cells:
        x0, y0, x1, y1 = [int(round(v)) for v in c["bbox"]]
        pad = 2
        crop = img.crop((max(0, x0 - pad), max(0, y0 - pad),
                         min(img.width, x1 + pad), min(img.height, y1 + pad)))
        if crop.width < 40:
            crop = crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS)
        try:
            text = pytesseract.image_to_string(crop, config="--psm 7").strip()
        except Exception:
            text = ""
        c["text"] = " ".join(text.split())
    return cells


def fill_text(cells, gt_cells, mode, img=None):
    if mode == "empty":
        for c in cells:
            c["text"] = ""
        return cells
    if mode == "tesseract":
        return ocr_cells(img, cells)
    grid, text = gt_grid(gt_cells)
    for c in cells:
        c["text"] = text.get((c["r"], c["c"]), "")
    return cells


# ---------------------------------------------------------------------- visualize
def draw(img, cells, path):
    from PIL import ImageDraw
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for c in cells:
        color = (220, 40, 120) if c["header"] else (40, 110, 200)
        d.rectangle(c["bbox"], outline=color, width=2)
    im.save(path)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/dev")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="microsoft/table-transformer-structure-recognition")
    ap.add_argument("--text", default="oracle", choices=["oracle", "empty", "tesseract"])
    ap.add_argument("--deskew", action="store_true")
    ap.add_argument("--max-dim", type=int, default=1000)
    ap.add_argument("--nms", type=float, default=0.0,
                    help="NMS IoU threshold for table row/column duplicates (0 = off)")
    ap.add_argument("--denoise", action="store_true", help="3x3 median filter before the model")
    ap.add_argument("--no-spans", action="store_true", help="ignore spanning-cell predictions")
    ap.add_argument("--viz", type=int, default=0, help="save overlays for the N worst tables")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    gt_dir, out_dir = Path(args.gt), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "_viz"

    manifest = json.loads((gt_dir / "manifest.json").read_text(encoding="utf-8"))

    model = TableTransformerForObjectDetection.from_pretrained(args.model).to(args.device).eval()
    id2label = model.config.id2label
    print(f"model   {args.model}  classes={list(id2label.values())}")
    print(f"device  {args.device}  text={args.text}  deskew={args.deskew}\n")

    report, times = [], []
    for entry in manifest:
        tid, tier = entry["id"], entry["tier"]
        img_path = gt_dir / entry["image"]
        img = Image.open(img_path).convert("RGB")
        angle = 0.0
        if args.deskew:
            img, angle = deskew(img)

        if args.denoise:
            from PIL import ImageFilter
            img = img.filter(ImageFilter.MedianFilter(3))
        work = max_resize(img, args.max_dim)
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(to_tensor(work).to(args.device))
        if args.device == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)

        objects = boxes_from_outputs(outputs, work.size, id2label)
        objects = nms(objects, ("table row", "table column"), args.nms)
        cells, n_rows, n_cols, n_spans = structure_to_cells(objects, no_spans=args.no_spans)

        # boxes live in `work` coordinates; rescale to the original image for viz
        sx, sy = img.width / work.width, img.height / work.height
        for c in cells:
            c["bbox"] = [c["bbox"][0] * sx, c["bbox"][1] * sy, c["bbox"][2] * sx, c["bbox"][3] * sy]

        gt_cells = json.loads((gt_dir / tid / "gt" / "cells.json").read_text(encoding="utf-8"))
        gt_rows = max(int(c["r"]) + int(c.get("rs", 1)) for c in gt_cells)
        gt_cols = max(int(c["c"]) + int(c.get("cs", 1)) for c in gt_cells)

        filled = fill_text([dict(c) for c in cells], gt_cells, args.text, img=img)
        (out_dir / f"{tid}.json").write_text(
            json.dumps([{k: c[k] for k in ("text", "r", "c", "rs", "cs", "header")} for c in filled],
                       ensure_ascii=False, indent=1), encoding="utf-8")

        pred_html = evaluate.cells_json_to_html(filled)
        gt_html = (gt_dir / tid / "gt" / "table.html").read_text(encoding="utf-8")
        report.append({
            "id": tid, "tier": tier, "style": entry["style"],
            "angle": round(angle, 1),
            "gt_shape": [gt_rows, gt_cols], "pred_shape": [n_rows, n_cols],
            "shape_ok": [n_rows, n_cols] == [gt_rows, gt_cols],
            "spans_detected": n_spans,
            "cells_gt": len(gt_cells), "cells_pred": len(filled),
            "teds": round(evaluate.teds(gt_html, pred_html), 4),
            "structure_teds": round(evaluate.teds(gt_html, pred_html, structure_only=True), 4),
            "seconds": round(dt, 2),
            "_html": pred_html, "_img": img, "_cells": cells,
        })

    (out_dir / "_report.json").write_text(
        json.dumps([{k: v for k, v in r.items() if not k.startswith("_")} for r in report],
                   indent=1), encoding="utf-8")

    # ---- summary
    by_tier = {}
    for r in report:
        by_tier.setdefault(r["tier"], []).append(r)

    print(f"{'tier':5} {'TEDS':>7} {'struct':>7} {'shape ok':>9} {'s/table':>8}")
    for tier in sorted(by_tier):
        rs = by_tier[tier]
        print(f"{tier:5} {np.mean([r['teds'] for r in rs]):7.4f} "
              f"{np.mean([r['structure_teds'] for r in rs]):7.4f} "
              f"{sum(r['shape_ok'] for r in rs)}/{len(rs):<7} "
              f"{np.mean([r['seconds'] for r in rs]):8.2f}")

    def wavg(key):
        w = {"T0": .1, "T1": .2, "T2": .3, "T3": .4}
        num = sum(w[t] * np.mean([r[key] for r in by_tier[t]]) for t in by_tier)
        return num / sum(w[t] for t in by_tier)

    print(f"\nweighted TEDS (tier-weighted, no QA): {wavg('teds'):.4f}")
    print(f"weighted structure TEDS              : {wavg('structure_teds'):.4f}")
    print(f"shape exactly right                  : {sum(r['shape_ok'] for r in report)}/{len(report)}")
    print(f"runtime mean {np.mean(times):.2f}s  p90 {np.percentile(times, 90):.2f}s")

    if args.viz:
        viz_dir.mkdir(exist_ok=True)
        for r in sorted(report, key=lambda x: x["teds"])[: args.viz]:
            draw(r["_img"], r["_cells"], viz_dir / f"{r['id']}.png")
        print(f"overlays -> {viz_dir}")


if __name__ == "__main__":
    main()
