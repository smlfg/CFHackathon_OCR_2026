"""Build registry/models.json from the OCRBench v2 data plus manual specialists.

The filter is the point: every open-source model on OCRBench v2 with at most
15B total and active parameters is a candidate. Candidates without an MLX or
GGUF build stay in the registry as `no_runtime` so the board shows what was
skipped and why, instead of silently looking complete.

    python -m pipeline registry build [--offline]
"""
from __future__ import annotations

import csv
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from . import config

CSV = config.ROOT / "ocrbench_v2" / "ocrbench_v2_all.csv"
MEDALS = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
SIZE = re.compile(r"([\d.]+)\s*B")
TOTAL_LIMIT = config.MAX_PARAMS_B


def slug(name: str) -> str:
    name = re.sub(r"\(.*?\)", "", name)
    name = name.replace("🥇", "").replace("🥈", "").replace("🥉", "")
    return re.sub(r"[^a-z0-9.]+", "-", name.strip().lower()).strip("-")


def _num(value: str) -> float | None:
    match = SIZE.search(value or "")
    return float(match.group(1)) if match else None


def hf_exists(repo: str) -> bool:
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}",
                                    timeout=20) as response:
            return response.status == 200
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return False


def candidates() -> dict[str, dict]:
    """Open-source OCRBench rows with total and active parameters <= 15B, deduped."""
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    found: dict[str, dict] = {}
    for row in rows:
        if (row.get("Open-source") or "").strip() != "Yes":
            continue
        total = _num(row.get("LLM Size", ""))
        active = _num(row.get("Active Parameters", "")) or total
        if total is None or total > TOTAL_LIMIT or (active or 0) > TOTAL_LIMIT:
            continue
        name = MEDALS.sub("", row["Method"]).strip()
        key = slug(name)
        entry = found.setdefault(key, {
            "method": name, "params_b": total, "active_b": active,
            "periods": [], "average": None, "venue": row.get("Venue", ""),
        })
        entry["periods"].append(row["period"])
        entry["params_b"] = max(entry["params_b"], total)
        try:
            avg = float(row.get("Average") or "")
            entry["average"] = max(entry["average"] or 0, avg)
        except ValueError:
            pass
    for entry in found.values():
        entry["periods"] = sorted(set(entry["periods"]))
        entry["latest_period"] = entry["periods"][-1]
    return found


def build(offline: bool = False, write: bool = True) -> dict:
    manual = json.loads(config.MANUAL.read_text(encoding="utf-8"))
    defaults = manual["defaults"]
    models: list[dict] = []
    used: set[str] = set()

    def add(entry: dict, source: str) -> None:
        merged = {**defaults, **{k: v for k, v in entry.items() if v is not None}}
        merged["active_b"] = entry.get("active_b") or merged.get("params_b")
        merged["source"] = source
        merged.setdefault("status", "ready")
        merged.setdefault("quant", "")
        models.append(merged)
        used.add(merged["id"])

    for specialist in manual["specialists"]:
        add(specialist, "specialist")

    ocrbench = candidates()
    matched_rules = []
    for rule in manual["ocrbench"]:
        pattern = re.compile(rule["match"])
        hits = [key for key in ocrbench if pattern.search(ocrbench[key]["method"])]
        if not hits:
            print(f"[registry] rule matched nothing: {rule['match']}", flush=True)
            continue
        key = max(hits, key=lambda k: ocrbench[k]["params_b"])
        meta = ocrbench[key]
        add({**rule, "params_b": meta["params_b"], "active_b": meta["active_b"],
             "ocrbench_average": meta["average"], "venue": meta["venue"],
             "periods": meta["periods"]}, "ocrbench")
        matched_rules.append(key)
        del ocrbench[key]

    for key, meta in sorted(ocrbench.items(), key=lambda kv: -(kv[1]["params_b"] or 0)):
        models.append({
            **defaults, "id": key, "name": meta["method"], "hf_repo": "",
            "params_b": meta["params_b"], "active_b": meta["active_b"],
            "quant": "", "source": "ocrbench", "status": "no_runtime",
            "ocrbench_average": meta["average"], "venue": meta["venue"],
            "periods": meta["periods"],
            "note": "open-source and <=15B on OCRBench, but no MLX or GGUF build found",
        })

    if not offline:
        for model in models:
            if model["status"] != "ready" or not model.get("hf_repo"):
                continue
            if not hf_exists(model["hf_repo"]):
                model["status"] = "repo_missing"
                model["note"] = f"HF repo does not resolve: {model['hf_repo']}"

    counts: dict[str, int] = {}
    for model in models:
        counts[model["status"]] = counts.get(model["status"], 0) + 1
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_limit_b": TOTAL_LIMIT,
        "source_csv": str(CSV.relative_to(config.ROOT)),
        "counts": counts,
        "models": models,
    }
    if write:
        config.REGISTRY.mkdir(parents=True, exist_ok=True)
        config.MODELS_YAML.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                                      encoding="utf-8")
    return payload


def load() -> dict:
    if not config.MODELS_YAML.exists():
        raise SystemExit("registry missing: run `python -m pipeline registry build`")
    return json.loads(config.MODELS_YAML.read_text(encoding="utf-8"))


def ready_models(include_status: tuple[str, ...] = ("ready",)) -> list[dict]:
    return [m for m in load()["models"] if m["status"] in include_status]
