#!/usr/bin/env python3
"""
Submission entrypoint contract:
    python predict.py <input_dir> <output_dir>

<input_dir>  contains one image per table, named <table_id>.png or .jpg
<output_dir> receives one <table_id>.json per image, written as soon as it is done

Runs the tablesalt pipeline (tablesalt/run.py); per table it falls back to the Tesseract
floor baseline and at worst to an empty cell list, so every image gets a valid file.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tablesalt.run import main  # noqa: E402


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:          # Windows / macOS
        return os.cpu_count() or 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], workers=available_cpus())
