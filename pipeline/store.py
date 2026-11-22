"""Metrics storage: sqlite for querying, JSONL for the append-only history.

Idempotent per run key (model, dataset, prompt hash, git sha), so re-running a
model updates its row instead of piling up duplicates.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import time
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    hf_repo TEXT NOT NULL,
    params_b REAL,
    active_b REAL,
    quant TEXT,
    runtime TEXT,
    prompt_hash TEXT,
    dataset TEXT NOT NULL,
    n_tables INTEGER,
    final REAL,
    teds_weighted REAL,
    qa_accuracy REAL,
    cell_f1 REAL,
    t0 REAL, t1 REAL, t2 REAL, t3 REAL,
    s_per_table REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    empty_rate REAL,
    adapter_flag TEXT,
    status TEXT,
    load_seconds REAL,
    git_sha TEXT,
    meta TEXT
);
CREATE TABLE IF NOT EXISTS tables (
    run_id TEXT NOT NULL,
    tid TEXT NOT NULL,
    tier TEXT,
    style TEXT,
    teds REAL,
    cell_f1 REAL,
    qa REAL,
    empty INTEGER,
    PRIMARY KEY (run_id, tid)
);
CREATE INDEX IF NOT EXISTS runs_model ON runs(model_id, dataset);
"""


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def db_path() -> Path:
    config.METRICS.mkdir(parents=True, exist_ok=True)
    return config.METRICS / "metrics.sqlite"


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(db_path())
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def run_id(model: dict, dataset: str, prompt_hash: str) -> str:
    return f"{model['id']}__{dataset}__{prompt_hash[:8]}"


def save(run: dict, tables: list[dict]) -> str:
    rid = run["run_id"]
    run.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    run.setdefault("git_sha", git_sha())
    for tier, value in (run.get("per_tier") or {}).items():
        run[tier.lower()] = value
    run["meta"] = json.dumps(run.get("meta") or {})
    con = connect()
    with con:
        con.execute("DELETE FROM runs WHERE run_id = ?", (rid,))
        con.execute("DELETE FROM tables WHERE run_id = ?", (rid,))
        cols = [c[1] for c in con.execute("PRAGMA table_info(runs)")]
        values = {c: run.get(c) for c in cols}
        con.execute(
            f"INSERT INTO runs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            [values[c] for c in cols],
        )
        con.executemany(
            "INSERT INTO tables (run_id, tid, tier, style, teds, cell_f1, qa, empty)"
            " VALUES (?,?,?,?,?,?,?,?)",
            [
                (
                    rid,
                    t["tid"],
                    t["tier"],
                    t["style"],
                    t["teds"],
                    t["cell_f1"],
                    t["qa"],
                    int(t["empty"]),
                )
                for t in tables
            ],
        )
    con.close()

    config.METRICS.mkdir(parents=True, exist_ok=True)
    with (config.METRICS / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(run, sort_keys=True) + "\n")
    return rid


def runs(dataset: str | None = None, latest_only: bool = True) -> list[dict]:
    con = connect()
    query = "SELECT * FROM runs"
    args: tuple = ()
    if dataset:
        query += " WHERE dataset = ?"
        args = (dataset,)
    query += " ORDER BY final DESC"
    rows = [dict(r) for r in con.execute(query, args)]
    con.close()
    if not latest_only:
        return rows
    best: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["model_id"], row["dataset"])
        if key not in best:
            best[key] = row
    return sorted(best.values(), key=lambda r: r["final"] or 0, reverse=True)


def table_rows(rid: str) -> list[dict]:
    con = connect()
    rows = [
        dict(r)
        for r in con.execute("SELECT * FROM tables WHERE run_id = ? ORDER BY teds ASC", (rid,))
    ]
    con.close()
    return rows


def has(model_id: str, dataset: str, prompt_hash: str) -> bool:
    con = connect()
    hit = con.execute(
        "SELECT 1 FROM runs WHERE run_id = ?", (run_id({"id": model_id}, dataset, prompt_hash),)
    ).fetchone()
    con.close()
    return hit is not None
