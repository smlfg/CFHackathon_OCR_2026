"""Image loading and atomic JSON writing."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def load_gray(path: Path) -> np.ndarray:
    """Read an image as uint8 grayscale (works with non-ASCII paths on Windows)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot decode image {path}")
    return img


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


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
