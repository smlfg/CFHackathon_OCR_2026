"""Datasets: ensure they exist, read the manifest, resolve images.

`dev` is the official 24-table yardstick that ships with the repo. Every other
set comes from generate_dataset.py and is generated once, then never touched.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from . import config


class Dataset:
    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))

    def image(self, entry: dict) -> Path:
        return self.path / entry["image"]

    def gt_html(self, entry: dict) -> str:
        return (self.path / entry["id"] / "gt" / "table.html").read_text(encoding="utf-8")

    def qa(self, entry: dict) -> list[dict]:
        return json.loads((self.path / entry["id"] / "gt" / "qa.json").read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.manifest)

    def tiers(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.manifest:
            counts[entry["tier"]] = counts.get(entry["tier"], 0) + 1
        return dict(sorted(counts.items()))


def load(name: str) -> Dataset:
    path = config.dataset_dir(name)
    if not (path / "manifest.json").exists():
        raise SystemExit(
            f"dataset '{name}' not found at {path}\n"
            f"generate it with: python generate_dataset.py --out {path} --n 240 --seed 2"
        )
    return Dataset(name, path)


def ensure(name: str, n: int = 240, seed: int = 2, generate: bool = False) -> Dataset:
    path = config.dataset_dir(name)
    if not (path / "manifest.json").exists():
        if not generate:
            raise SystemExit(f"dataset '{name}' missing at {path}; pass --generate to build it")
        print(f"[data] generating {n} tables -> {path}", flush=True)
        subprocess.run(
            [sys.executable, str(config.ROOT / "generate_dataset.py"),
             "--out", str(path), "--n", str(n), "--seed", str(seed)],
            check=True,
        )
    return load(name)
