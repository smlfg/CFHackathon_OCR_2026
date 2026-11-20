"""Run one model over one dataset, then convert and inspect the output.

Resumable: a table whose raw text already exists is skipped. The model stays
loaded for the whole dataset and is released by the caller when scoring is done.
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import statistics
import urllib.error
import urllib.request
from pathlib import Path

from . import config, convert, dataset, providers


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


_SIZE_CACHE: dict[str, float] = {}


def remote_size_gb(repo: str) -> float:
    """Download size from the HF API, 0.0 if unknown."""
    if repo in _SIZE_CACHE:
        return _SIZE_CACHE[repo]
    size = 0.0
    try:
        with urllib.request.urlopen(
                f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=30) as response:
            payload = json.loads(response.read().decode())
        size = sum(b.get("size") or 0 for b in payload.get("siblings", [])) / 1e9
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        pass
    _SIZE_CACHE[repo] = size
    return size


def disk_check(model: dict, margin_gb: float = 1.0) -> tuple[bool, float, float]:
    """(ok, need_gb, free_gb). Only MLX models are checked; weights land on this disk."""
    if model.get("runtime") != "mlx":
        return True, 0.0, 0.0
    need = remote_size_gb(model["hf_repo"])
    free = shutil.disk_usage(config.RAW).free / 1e9
    if not need:
        return True, 0.0, free
    return need + margin_gb <= free, need, free


def preprocess(path: Path, max_edge: int) -> bytes:
    """RGB PNG, longest edge capped, never upscaled. Same pixels for every model."""
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if max_edge:
        width, height = image.size
        scale = max_edge / max(width, height)
        if scale < 1:
            image = image.resize((round(width * scale), round(height * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def run(model: dict, ds: dataset.Dataset, *, force: bool = False, limit: int | None = None,
        jobs: int = 1) -> dict:
    raw_dir = config.RAW / model["id"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    provider = providers.get(model["runtime"])
    loaded = provider.load(model["hf_repo"])

    entries = ds.manifest[:limit] if limit else ds.manifest
    done, failed = [], []
    for entry in entries:
        target = raw_dir / f"{entry['id']}.txt"
        if target.exists() and not force:
            done.append(entry["id"])
            continue
        image_png = preprocess(ds.image(entry), int(model.get("max_edge") or 0))
        try:
            text, usage = provider.generate(
                image_png,
                model["prompt"],
                max_tokens=int(model.get("max_tokens") or 4096),
                temperature=float(model.get("temperature") or 0.0),
            )
        except providers.ProviderError as exc:
            print(f"  {entry['id']}: FAILED {exc}", flush=True)
            failed.append({"id": entry["id"], "error": str(exc)[:500]})
            continue
        target.write_text(text)
        _merge_meta(raw_dir / "meta.json", {
            "id": entry["id"], "tier": entry["tier"], "style": entry["style"],
            "seconds": usage.get("seconds"), "prompt_tokens": usage.get("prompt_tokens", 0),
            "generation_tokens": usage.get("completion_tokens", 0),
            "chars": len(text),
        })
        print(f"  {entry['id']} {entry['tier']:<3} {usage.get('seconds', 0):6.2f}s "
              f"{len(text):5d} chars", flush=True)

    if failed:
        (raw_dir / "failures.json").write_text(json.dumps(failed, indent=1))

    pred_dir = config.PRED / model["id"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    converted = _convert_all(model, raw_dir, pred_dir)
    return collect(model, ds, converted=converted, loaded=loaded,
                   n_run=len(done), n_failed=len(failed))


def collect(model: dict, ds: dataset.Dataset, *, converted: dict | None = None,
            loaded: dict | None = None, n_run: int = 0, n_failed: int = 0) -> dict:
    """Build the run row from whatever is already on disk (no inference)."""
    raw_dir = config.RAW / model["id"]
    pred_dir = config.PRED / model["id"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    if converted is None:
        converted = _convert_all(model, raw_dir, pred_dir)
    meta_rows = _read_meta(raw_dir / "meta.json")
    seconds = [m["seconds"] for m in meta_rows if m.get("seconds")]
    tokens = [m.get("generation_tokens") or 0 for m in meta_rows]
    failures = raw_dir / "failures.json"
    n_failed = n_failed or (len(json.loads(failures.read_text())) if failures.exists() else 0)

    empty_tables = converted["missing"]
    adapter_flag = ""
    if converted["tables"] and (empty_tables / converted["tables"] > config.MAX_EMPTY_RATE
                                or converted["median_cells"] < 2):
        adapter_flag = "ADAPTER_SUSPECT"

    return {
        "run_id": store_run_id(model, ds.name),
        "model_id": model["id"],
        "model_name": model.get("name", model["id"]),
        "hf_repo": model["hf_repo"],
        "params_b": model.get("params_b"),
        "active_b": model.get("active_b"),
        "quant": model.get("quant", ""),
        "runtime": model["runtime"],
        "prompt_hash": prompt_hash(model["prompt"]),
        "dataset": ds.name,
        "s_per_table": round(statistics.mean(seconds), 2) if seconds else None,
        "prompt_tokens": sum(m.get("prompt_tokens") or 0 for m in meta_rows),
        "completion_tokens": sum(tokens),
        "empty_rate": converted["empty_rate"],
        "adapter_flag": adapter_flag,
        "status": "ok" if not n_failed else f"{n_failed} failed",
        "load_seconds": (loaded or {}).get("seconds"),
        "meta": {"converted": converted, "tables_run": n_run, "failed": n_failed,
                 "max_edge": model.get("max_edge"), "fmt": model.get("fmt"),
                 "prompt": model["prompt"]},
    }


def store_run_id(model: dict, dataset_name: str) -> str:
    from .store import run_id
    return run_id(model, dataset_name, prompt_hash(model["prompt"]))


def _convert_all(model: dict, raw_dir: Path, pred_dir: Path) -> dict:
    fmt = model.get("fmt", "auto")
    tables = cell_counts = empty_cells = 0
    counts: list[int] = []
    missing = 0
    for path in sorted(raw_dir.glob("*.txt")):
        html, info = convert.convert(path.read_text(), fmt)
        if not html:
            missing += 1
            (pred_dir / f"{path.stem}.txt").write_text(path.read_text())
            continue
        (pred_dir / f"{path.stem}.html").write_text(html)
        tables += 1
        cell_counts += info["cells"]
        empty_cells += round(info["empty_rate"] * info["cells"])
        counts.append(info["cells"])
    return {
        "fmt": fmt,
        "tables": tables,
        "missing": missing,
        "cells": cell_counts,
        "median_cells": statistics.median(counts) if counts else 0,
        "empty_rate": round(empty_cells / cell_counts, 3) if cell_counts else 1.0,
    }


def _merge_meta(path: Path, row: dict) -> None:
    rows = _read_meta(path)
    known = {r["id"]: i for i, r in enumerate(rows)}
    if row["id"] in known:
        rows[known[row["id"]]] = row
    else:
        rows.append(row)
    path.write_text(json.dumps(rows, indent=1))


def _read_meta(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
