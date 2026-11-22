"""Step 5 - OCR: read every text blob with Tesseract.

Each blob is cropped from the rectified grey image (rule pixels whitened), scaled so the text
height becomes ``ocr_cap_height`` and padded white.  All crops of one table go to ONE
Tesseract process via a list file (``tesseract list.txt stdout tsv``): this avoids ~100
process starts per table.  ``page_num`` in the TSV maps the words back to their crop.
"""

from __future__ import annotations

import csv
import io
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .config import CFG, Config
from .types import Blob


def crop_blob(
    gray: np.ndarray, b: Blob, h: float, cfg: Config = CFG, rules_mask: np.ndarray | None = None
) -> np.ndarray:
    """Crop one blob, whiten rules, scale to cap height and pad with white."""
    m = cfg.ocr_margin
    y0, y1 = max(0, b.y0 - m), min(gray.shape[0], b.y1 + m)
    x0, x1 = max(0, b.x0 - m), min(gray.shape[1], b.x1 + m)
    crop = gray[y0:y1, x0:x1].copy()
    if rules_mask is not None:
        crop[rules_mask[y0:y1, x0:x1]] = 255
    scale = cfg.ocr_cap_height / max(h, 1.0)
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if cfg.ocr_median:
        crop = cv2.medianBlur(crop, cfg.ocr_median)
    p = cfg.ocr_pad
    return cv2.copyMakeBorder(crop, p, p, p, p, cv2.BORDER_CONSTANT, value=255)


TESS_ERRORS: list[str] = []  # per process; read by run.py


def run_tesseract(
    images: list[np.ndarray], psm: int = 7, whitelist: str | None = None, cfg: Config = CFG
) -> list[tuple[str, float]]:
    """OCR many small images in one Tesseract call -> [(text, mean word conf)]."""
    if not images:
        return []
    with tempfile.TemporaryDirectory(prefix="ts_ocr_") as tmp:
        tmpd = Path(tmp)
        names = []
        for k, im in enumerate(images):
            p = tmpd / f"{k:04d}.png"
            cv2.imwrite(str(p), im)
            names.append(str(p))
        lst = tmpd / "list.txt"
        lst.write_text("\n".join(names) + "\n", encoding="utf-8")
        cmd = [cfg.tess_cmd, str(lst), "stdout", "--psm", str(psm), "-l", cfg.tess_lang]
        if whitelist:
            cmd += ["-c", f"tessedit_char_whitelist={whitelist}"]
        cmd += ["tsv"]
        env = dict(os.environ, OMP_THREAD_LIMIT="1")
        res = subprocess.run(cmd, capture_output=True, env=env, timeout=120)
        out = res.stdout.decode("utf-8", errors="replace")
        if res.returncode != 0 or not out.strip():
            # make failures visible: they used to vanish and turn into empty cells
            TESS_ERRORS.append(
                f"rc={res.returncode} {res.stderr.decode('utf-8', 'replace')[-300:]}"
            )
            if len(TESS_ERRORS) <= 3:
                print(
                    f"[tesseract] {' '.join(cmd[2:])}: {TESS_ERRORS[-1]}",
                    file=sys.stderr,
                    flush=True,
                )
    return parse_tsv(out, len(images))


def parse_tsv(tsv: str, n: int) -> list[tuple[str, float]]:
    """Tesseract TSV output -> per-crop (text, mean word confidence)."""
    words: list[list[tuple[int, int, str, float]]] = [[] for _ in range(n)]
    reader = csv.reader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)
    header = next(reader, None)
    if not header:
        return [("", -1.0)] * n
    col = {name: i for i, name in enumerate(header)}
    for row in reader:
        if len(row) < len(header) or row[col["level"]] != "5":
            continue
        text = row[col["text"]].strip()
        if not text:
            continue
        page = int(row[col["page_num"]]) - 1
        if 0 <= page < n:
            words[page].append(
                (int(row[col["line_num"]]), int(row[col["left"]]), text, float(row[col["conf"]]))
            )
    out = []
    for ws in words:
        ws.sort(key=lambda w: (w[0], w[1]))
        text = " ".join(w[2] for w in ws)
        conf = float(np.mean([w[3] for w in ws])) if ws else -1.0
        out.append((text, conf))
    return out


def read_variants(
    gray: np.ndarray,
    blobs: list[Blob],
    h: float,
    rules_mask: np.ndarray | None = None,
    cfg: Config = CFG,
) -> list[list[tuple[str, float]]]:
    """Read every blob once per (margin, median) variant -> per blob a list of candidates.
    Different crops fail differently (a tight crop clips the flat top of a '5', a loose one
    picks up specks), so the caller chooses per cell role."""
    per_variant = []
    for margin, median, psm in cfg.ocr_variants:
        vcfg = replace(cfg, ocr_margin=margin, ocr_median=median, tess_psm=psm)
        per_variant.append(read_blobs(gray, blobs, h, rules_mask=rules_mask, cfg=vcfg))
    cands = [[pv[i] for pv in per_variant] for i in range(len(blobs))]
    empty = [i for i, c in enumerate(cands) if not any(t for t, _ in c)]
    if empty and cfg.rescue_variant:
        margin, median, psm = cfg.rescue_variant
        vcfg = replace(cfg, ocr_margin=margin, ocr_median=median, tess_psm=psm)
        for i, r in zip(
            empty,
            read_blobs(gray, blobs, h, idx=empty, rules_mask=rules_mask, cfg=vcfg),
            strict=False,
        ):
            cands[i] = [r] * len(cands[i])
    return cands


def read_blobs(
    gray: np.ndarray,
    blobs: list[Blob],
    h: float,
    idx: list[int] | None = None,
    whitelist: str | None = None,
    rules_mask: np.ndarray | None = None,
    cfg: Config = CFG,
) -> list[tuple[str, float]]:
    """OCR the blobs with the given indices (default: all) -> [(text, conf)] in that order."""
    idx = list(range(len(blobs))) if idx is None else idx
    crops = [crop_blob(gray, blobs[i], h, cfg, rules_mask) for i in idx]
    return run_tesseract(crops, psm=cfg.tess_psm, whitelist=whitelist, cfg=cfg)
