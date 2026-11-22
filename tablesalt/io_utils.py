"""Input discovery and decoding (images and PDFs) and atomic JSON writing."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

RASTER_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
PDF_EXTS = (".pdf",)
# The contract promises .png/.jpg; the kick-off said "PDF files or PNG files" may appear, so we
# accept a few more raster formats and PDFs instead of silently skipping them (ADR-009).
IMAGE_EXTS = RASTER_EXTS + PDF_EXTS
PDF_DPI = 200  # first page only; 200 dpi puts 10-12 pt table text near the generator's 20 px font


def load_gray(path: Path) -> np.ndarray:
    """Read an image as uint8 grayscale (works with non-ASCII paths on Windows)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot decode image {path}")
    return img


def is_pdf(path: Path) -> bool:
    return Path(path).suffix.lower() in PDF_EXTS


def render_pdf_page(path: Path, out_png: Path, dpi: int = PDF_DPI) -> Path:
    """Render the first page of a PDF to a PNG file and return its path.

    pypdfium2 is a self-contained wheel (no system libraries), so this works offline in the
    slim container.  Multi-page PDFs: only page 1 is read - the contract is one table per file.
    """
    import pypdfium2 as pdfium  # local import: only needed when a PDF actually shows up

    pdf = pdfium.PdfDocument(str(path))
    try:
        if len(pdf) == 0:
            raise ValueError(f"PDF without pages: {path}")
        bitmap = pdf[0].render(scale=dpi / 72)
        bitmap.to_pil().convert("RGB").save(out_png)
    finally:
        pdf.close()
    return Path(out_png)


def list_images(folder: Path) -> list[Path]:
    """All table inputs in ``folder`` (flat, sorted by name)."""
    return sorted(
        p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def write_json_atomic(path: Path, obj) -> None:
    """Write JSON via a temp file + os.replace so a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
