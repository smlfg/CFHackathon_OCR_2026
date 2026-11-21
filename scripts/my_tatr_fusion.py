#!/usr/bin/env python3
"""Saubere TATR -> Tesseract Pipeline (Phase 3 Prototyp).

Aufbau in 4 klaren Bausteinen:
  1. Geometrie        (intersect, box_area, containment, iou, nms)
  2. TATR -> Zellen   (Median-Denoise, MaxResize 1000, DETR, NMS, Zeilen ∩ Spalten)
  3. Zell-OCR         (Randlinien weg, weißer Rand, Upscale, Tesseract deu+eng)
  4. CLI              (speichert <id>.json im offiziellen TableSalt-Format)

Ausführen:
  .venv-tatr/bin/python scripts/my_tatr_fusion.py --gt data/dev --out out/my_tatr_fusion
  .venv/bin/python score_all.py --gt data/dev --pred out/my_tatr_fusion
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytesseract
import torch
from PIL import Image, ImageFilter, ImageOps
from transformers import TableTransformerForObjectDetection

Box = tuple[float, float, float, float]  # (x0, y0, x1, y1) in Pixeln

MODEL_ID = "microsoft/table-transformer-structure-recognition"
TESS_LANG = "deu+eng"


# ============================================================================
# BAUSTEIN 1: Rechteck-Geometrie
# ============================================================================

def intersect(a: Box, b: Box) -> Box:
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


def box_area(box: Box) -> float:
    x0, y0, x1, y1 = box
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    return width * height


def containment(word_box: Box, cell_box: Box) -> float:
    word_area = box_area(word_box)
    if word_area <= 0.0:
        return 0.0
    return box_area(intersect(word_box, cell_box)) / word_area


def iou(a: Box, b: Box) -> float:
    inter_area = box_area(intersect(a, b))
    union_area = box_area(a) + box_area(b) - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def union_boxes(boxes: list[Box]) -> Box:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def nms(objects: list[dict], labels: tuple[str, ...], threshold: float = 0.3) -> list[dict]:
    """Entfernt doppelte Zeilen-/Spalten-Boxen (z. B. bei Fax-Rauschen T3)."""
    out: list[dict] = []
    for label in labels:
        group = sorted(
            [o for o in objects if o["label"] == label],
            key=lambda o: -o["score"],
        )
        keep: list[dict] = []
        for cand in group:
            if all(iou(cand["bbox"], k["bbox"]) < threshold for k in keep):
                keep.append(cand)
        out.extend(keep)
    out.extend([o for o in objects if o["label"] not in labels])
    return out


# ============================================================================
# BAUSTEIN 2: TATR -> Zellen-Gitter
# ============================================================================

def max_resize(img: Image.Image, max_size: int = 1000) -> Image.Image:
    w, h = img.size
    scale = max_size / max(w, h)
    if scale >= 1.0:
        return img
    return img.resize((int(round(w * scale)), int(round(h * scale))), Image.BICUBIC)


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(arr.transpose(2, 0, 1))[None]


def detect_structure_objects(
    img: Image.Image,
    model: TableTransformerForObjectDetection,
    device: str,
    score_threshold: float = 0.5,
    nms_threshold: float = 0.3,
) -> list[dict]:
    """Bild -> MaxResize(1000) -> TATR -> Pixel-Boxen im Original-Bildformat."""
    work = max_resize(img, 1000)

    with torch.no_grad():
        outputs = model(image_to_tensor(work).to(device))
    if device == "mps":
        torch.mps.synchronize()

    probs = outputs.logits.softmax(-1)[0]
    scores, labels = probs.max(-1)
    boxes = outputs.pred_boxes[0].detach().cpu().numpy()

    # Direkt in Original-Bildkoordinaten umrechnen (cx, cy, bw, bh -> x0, y0, x1, y1)
    orig_w, orig_h = img.size
    id2label = model.config.id2label
    objs: list[dict] = []
    for (cx, cy, bw, bh), s, l in zip(boxes, scores.tolist(), labels.tolist()):
        name = id2label.get(int(l), "no object")
        if name == "no object" or s < score_threshold:
            continue
        bbox: Box = (
            float((cx - bw / 2.0) * orig_w),
            float((cy - bh / 2.0) * orig_h),
            float((cx + bw / 2.0) * orig_w),
            float((cy + bh / 2.0) * orig_h),
        )
        objs.append({"label": name, "score": float(s), "bbox": bbox})

    return nms(objs, ("table row", "table column"), threshold=nms_threshold)


def structure_to_cells(objects: list[dict]) -> list[dict]:
    """Zeilen ∩ Spalten = Basis-Zellen; Spanning-Cells verschmelzen abgedeckte Zellen."""
    rows = sorted([o for o in objects if o["label"] == "table row"], key=lambda o: o["bbox"][1])
    cols = sorted([o for o in objects if o["label"] == "table column"], key=lambda o: o["bbox"][0])
    spans = [
        o for o in objects
        if o["label"] in ("table spanning cell", "table projected row header")
    ]
    headers = [o for o in objects if o["label"] == "table column header"]

    header_rows: set[int] = set()
    for r, row in enumerate(rows):
        row_h = max(1e-6, row["bbox"][3] - row["bbox"][1])
        for hd in headers:
            ov = intersect(row["bbox"], hd["bbox"])
            if max(0.0, ov[3] - ov[1]) / row_h >= 0.5:
                header_rows.add(r)

    cells: list[dict] = []
    subcells: list[dict] = []
    for r, row in enumerate(rows):
        for c, col in enumerate(cols):
            rect = intersect(row["bbox"], col["bbox"])
            if box_area(rect) <= 0.0:
                continue
            cell = {"bbox": rect, "rows": {r}, "cols": {c}, "header": r in header_rows}
            if any(containment(rect, s["bbox"]) > 0.5 for s in spans):
                subcells.append(cell)
            else:
                cells.append(cell)

    for s in spans:
        covered = [sc for sc in subcells if containment(sc["bbox"], s["bbox"]) > 0.5]
        if not covered:
            continue
        cells.append({
            "bbox": union_boxes([sc["bbox"] for sc in covered]),
            "rows": set().union(*(sc["rows"] for sc in covered)),
            "cols": set().union(*(sc["cols"] for sc in covered)),
            "header": all(sc["header"] for sc in covered),
        })

    out: list[dict] = []
    for cell in sorted(cells, key=lambda x: (min(x["rows"]), min(x["cols"]))):
        out.append({
            "r": min(cell["rows"]),
            "c": min(cell["cols"]),
            "rs": len(cell["rows"]),
            "cs": len(cell["cols"]),
            "header": bool(cell["header"]),
            "bbox": cell["bbox"],
        })
    return out


# ============================================================================
# BAUSTEIN 3: Sauberes Zell-OCR (Tesseract)
# ============================================================================

def clean_cell_crop(gray: Image.Image, bbox: Box) -> Image.Image | None:
    """Schneidet eine Zelle sauber aus: Randlinien entfernen + weißer Rand + Upscale."""
    w, h = gray.size
    x0 = max(0, int(round(bbox[0])))
    y0 = max(0, int(round(bbox[1])))
    x1 = min(w, int(round(bbox[2])))
    y1 = min(h, int(round(bbox[3])))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None

    arr = np.asarray(gray.crop((x0, y0, x1, y1)), dtype=np.uint8).copy()
    ch, cw = arr.shape

    # 1) Grauen Header-Hintergrund / Schatten sanft aufhellen (Hintergrund -> ~255)
    bg = float(np.percentile(arr, 85))
    if 120.0 < bg < 248.0:
        arr = np.clip(arr.astype(np.float32) * (255.0 / bg), 0, 255).astype(np.uint8)

    # 2) Dunkle Tabellen-Randlinien an den äußeren 5 px jeder Seite weiß übermalen
    #    (verhindert, dass Tesseract "|" oder "[" aus den Gitterlinien liest!)
    dark = arr < 140
    edge_y = min(6, ch // 4)
    edge_x = min(6, cw // 4)
    for y in list(range(edge_y)) + list(range(ch - edge_y, ch)):
        if dark[y, :].mean() > 0.55:
            arr[y, :] = 255
    for x in list(range(edge_x)) + list(range(cw - edge_x, cw)):
        if dark[:, x].mean() > 0.55:
            arr[:, x] = 255

    # 3) Wenn im Inneren praktisch keine dunkle Tinte ist -> leere Zelle!
    if (arr < 150).sum() < 6:
        return None

    crop = Image.fromarray(arr)

    # 4) Kleine Zellen hochskalieren (Tesseract liest am besten bei ~30-40px Schrifthöhe)
    if crop.height < 64:
        scale = min(3.0, 64.0 / max(1, crop.height))
        crop = crop.resize(
            (int(round(crop.width * scale)), int(round(crop.height * scale))),
            Image.LANCZOS,
        )

    # 5) Sauberer weißer Rand rundherum (Tesseract braucht Weißraum um Buchstaben!)
    return ImageOps.expand(crop, border=16, fill=255)


def ocr_single_cell(gray: Image.Image, cell: dict) -> str:
    crop = clean_cell_crop(gray, cell["bbox"])
    if crop is None:
        return ""

    # Standard: eine Textzeile (--psm 7); bei sehr hohen Zellen erst Block (--psm 6)
    first_psm = 6 if (cell["bbox"][3] - cell["bbox"][1]) > 55 else 7
    fallback_psms = [6, 8] if first_psm == 7 else [7, 8]

    for psm in [first_psm, *fallback_psms]:
        try:
            raw = pytesseract.image_to_string(
                crop, lang=TESS_LANG, config=f"--psm {psm}"
            ).strip()
        except Exception:
            raw = ""
        text = " ".join(raw.split())
        if text:
            return text
    return ""


def fill_cells_with_tesseract(img: Image.Image, cells: list[dict], workers: int = 6) -> list[dict]:
    gray = img.convert("L")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        texts = list(pool.map(lambda c: ocr_single_cell(gray, c), cells))
    out: list[dict] = []
    for c, txt in zip(cells, texts):
        out.append({
            "text": txt,
            "r": c["r"],
            "c": c["c"],
            "rs": c["rs"],
            "cs": c["cs"],
            "header": c["header"],
        })
    return out


# ============================================================================
# TESTS & CLI
# ============================================================================

def test_geometry() -> None:
    assert box_area((10, 20, 30, 50)) == 600.0
    assert box_area((30, 20, 10, 50)) == 0.0
    word, cell = (10, 10, 30, 20), (0, 0, 100, 50)
    assert containment(word, cell) == 1.0
    assert containment((90, 10, 110, 20), cell) == 0.5
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - (50.0 / 150.0)) < 1e-9
    assert abs(iou(word, cell) - 0.04) < 1e-9
    print("OK: alle Geometrie-Tests bestanden")


def has_speckle_noise(img: Image.Image, threshold: float = 0.010) -> bool:
    """Erkennt Salz-und-Pfeffer-Rauschen (Fax/T3) über Abweichung vom 3x3-Median."""
    g = img.convert("L")
    a = np.asarray(g, dtype=np.int16)
    m = np.asarray(g.filter(ImageFilter.MedianFilter(3)), dtype=np.int16)
    return float((np.abs(a - m) > 80).mean()) > threshold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/dev")
    ap.add_argument("--out", default="out/my_tatr_fusion")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    if args.test:
        test_geometry()
        return

    gt_dir, out_dir = Path(args.gt), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((gt_dir / "manifest.json").read_text(encoding="utf-8"))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = TableTransformerForObjectDetection.from_pretrained(
        MODEL_ID, local_files_only=True
    ).to(device).eval()

    t_start = time.perf_counter()
    for entry in manifest:
        tid = entry["id"]
        t0 = time.perf_counter()
        img = Image.open(gt_dir / entry["image"]).convert("RGB")
        clean = img.filter(ImageFilter.MedianFilter(3))
        objs = detect_structure_objects(clean, model, device)
        cells = structure_to_cells(objs)
        # Median-Filter fürs OCR nur bei echtem Salz-und-Pfeffer-Rauschen (schont scharfe Schrift!)
        ocr_img = clean if has_speckle_noise(img) else img
        filled = fill_cells_with_tesseract(ocr_img, cells, workers=args.workers)
        (out_dir / f"{tid}.json").write_text(
            json.dumps(filled, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        dt = time.perf_counter() - t0
        print(f"{tid} ({entry['tier']} {entry['style']:10s})  cells={len(filled):3d}  {dt:.2f}s")

    total = time.perf_counter() - t_start
    print(f"\nFertig: {len(manifest)} Tabellen in {total:.1f}s ({total / len(manifest):.2f}s/table) -> {out_dir}")


if __name__ == "__main__":
    main()
