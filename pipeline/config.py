"""Paths and runtime settings, all overridable by environment variables."""

from __future__ import annotations

import os
from pathlib import Path


def find_root() -> Path:
    """Repository root: the directory containing pipeline/ and score_all.py."""
    env = os.environ.get("TABLESALT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parent
    return here.parent


ROOT = find_root()
DATA = Path(os.environ.get("TABLESALT_DATA", ROOT / "data"))
RAW = Path(os.environ.get("TABLESALT_RAW", ROOT / "raw"))
PRED = Path(os.environ.get("TABLESALT_PRED", ROOT / "predictions"))
METRICS = Path(os.environ.get("TABLESALT_METRICS", ROOT / "metrics"))
REGISTRY = Path(os.environ.get("TABLESALT_REGISTRY", ROOT / "registry"))
MODELS_YAML = REGISTRY / "models.json"
MANUAL = REGISTRY / "manual.json"

MLX_BASE_URL = os.environ.get("MLX_BASE_URL", "http://127.0.0.1:8099")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

REQUEST_TIMEOUT = int(os.environ.get("TABLESALT_TIMEOUT", "900"))
# models larger than this are evaluated but logged, never silently dropped
MAX_PARAMS_B = float(os.environ.get("TABLESALT_MAX_PARAMS", "15"))
# adapter sanity gate
MAX_EMPTY_RATE = float(os.environ.get("TABLESALT_MAX_EMPTY_RATE", "0.20"))

DATASETS = {
    "dev": DATA / "dev",  # official yardstick, 24 tables, never trained on
    "val": DATA / "val",  # generated decision set, 240 tables
    "train": DATA / "train",  # only for fine-tuning, not for scoring
}


def dataset_dir(name: str) -> Path:
    if name in DATASETS:
        return DATASETS[name]
    return Path(name).expanduser().resolve()
