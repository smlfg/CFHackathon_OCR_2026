"""CLI.

    python -m pipeline registry build
    python -m pipeline list
    python -m pipeline run --all --dataset dev
    python -m pipeline run --model paddleocr-vl --model lighton-ocr-3-0.8b --dataset dev
    python -m pipeline board
    python -m pipeline status
"""
from __future__ import annotations

import argparse
import sys

from . import board, config, dataset, providers, registry, runner, score, store


def _select(models: list[dict], wanted: list[str] | None, everything: bool) -> list[dict]:
    if everything or not wanted:
        return models
    picked = []
    for needle in wanted:
        needle = needle.lower()
        hits = [m for m in models
                if m["id"].lower() == needle or needle in m["id"].lower()
                or needle in m.get("name", "").lower()]
        if not hits:
            raise SystemExit(f"no model matches '{needle}' (see `python -m pipeline list`)")
        for hit in hits:
            if hit not in picked:
                picked.append(hit)
    return picked


def cmd_registry(args) -> None:
    payload = registry.build(offline=args.offline)
    print(f"[registry] {payload['counts']} -> {config.MODELS_YAML}")
    for model in payload["models"]:
        mark = {"ready": "ok  ", "no_runtime": "skip", "repo_missing": "miss"}.get(
            model["status"], "?")
        print(f"  [{mark}] {model['id']:28s} {model['params_b']:>5.2f}B  "
              f"{model.get('hf_repo') or '—'}")


def cmd_list(args) -> None:
    data = registry.load()
    print(f"registry {data['generated']}  {data['counts']}")
    for model in data["models"]:
        if args.status and model["status"] != args.status:
            continue
        print(f"{model['status']:12s} {model['id']:28s} {model['params_b']:>5.2f}B "
              f"{(model.get('quant') or '–'):5s} {model.get('fmt'):8s} {model.get('hf_repo','')}")


def cmd_run(args) -> None:
    ds = dataset.ensure(args.dataset, n=args.n, seed=args.seed, generate=args.generate)
    print(f"[data] {ds.name}: {len(ds)} tables {ds.tiers()}", flush=True)

    all_models = registry.load()["models"]
    ready = [m for m in all_models if m["status"] == "ready"]
    selected = _select(ready, args.model, args.all)
    if not selected:
        raise SystemExit("nothing to run")

    runtime_needed = {m["runtime"] for m in selected}
    for runtime in runtime_needed:
        provider = providers.get(runtime)
        if not provider.healthy():
            raise SystemExit(
                f"{runtime} endpoint not reachable ({provider.base_url}).\n"
                + ("start scripts/mlx_server.py on the host, e.g.\n"
                   "  uv run --with mlx-vlm --with fastapi --with uvicorn "
                   "python scripts/mlx_server.py --port 8099\n"
                   if runtime == "mlx" else
                   "start Ollama, e.g. `ollama serve`\n"))

    print(f"[run] {len(selected)} model(s) on {ds.name}: "
          f"{', '.join(m['id'] for m in selected)}\n", flush=True)

    for index, model in enumerate(selected):
        print(f"=== {model['id']} ({model.get('name')}, {model['params_b']}B)", flush=True)
        if store.has(model["id"], ds.name, runner.prompt_hash(model["prompt"])) \
                and not args.force and not args.rescore:
            print("  already scored (use --force to re-run)\n", flush=True)
            continue
        if args.rescore:
            if not (config.RAW / model["id"]).glob("*.txt"):
                print("  no raw output to re-score\n", flush=True)
                continue
            run = runner.collect(model, ds)
        else:
            ok, need, free = runner.disk_check(model)
            if not ok:
                print(f"  SKIP: needs {need:.1f} GB, {free:.1f} GB free", flush=True)
                run = runner.collect(model, ds, converted={"tables": 0, "missing": 0,
                                                           "cells": 0, "median_cells": 0,
                                                           "empty_rate": 0.0})
                run["status"] = "no_space"
                run["meta"]["note"] = f"needs {need:.1f} GB, {free:.1f} GB free"
                store.save(run, [])
                continue
            _prefetch_next(selected, index, ds, args)
            try:
                run = runner.run(model, ds, force=args.force, limit=args.limit)
            except providers.ProviderError as exc:
                print(f"  ERROR: {exc}\n", flush=True)
                failed_run = runner.collect(model, ds) if (config.RAW / model["id"]).exists() else None
                if failed_run is not None:
                    failed_run["status"] = "load_error"
                    failed_run["meta"]["note"] = str(exc)[:400]
                    store.save(failed_run, [])
                if not args.keep_model:
                    _prune(model)
                continue

        aggregate, tables = score.score(ds.path, config.PRED / model["id"])
        run.update(aggregate)
        run["status"] = "error" if aggregate["empty_tables"] == aggregate["n_tables"] else run["status"]
        store.save(run, tables)
        board.render(ds.name)
        print(f"  FINAL {run['final']:.4f}  TEDS {run['teds_weighted']:.4f}  "
              f"QA {run['qa_accuracy']:.4f}  CellF1 {run['cell_f1']:.4f}  "
              f"{run['s_per_table'] or 0:.1f}s/table"
              + (f"  [{run['adapter_flag']}]" if run["adapter_flag"] else ""), flush=True)

        if not args.keep_model:
            _prune(model)

    for name in {ds.name}:
        paths = board.render(name)
        print(f"\n[board] {paths[name]['md']}\n[board] {paths[name]['html']}", flush=True)


def _prefetch_next(selected: list[dict], index: int, ds, args) -> None:
    """Warm the next model while this one infers — the GPU is idle otherwise.

    Small enough to co-reside: load it into the server's second slot, so its 2–6 minute
    startup is hidden entirely. Bigger: only warm the download.
    """
    for nxt in selected[index + 1:]:
        if nxt["runtime"] != "mlx":
            continue
        if not (args.force or args.rescore) and store.has(
                nxt["id"], ds.name, runner.prompt_hash(nxt["prompt"])):
            continue
        provider = providers.get("mlx")
        big = (nxt.get("params_b") or 0) > config.MAX_PRELOAD_B
        try:
            if big:
                out = provider.prefetch(nxt["hf_repo"])
                if not out.get("cached"):
                    print(f"  prefetch {nxt['hf_repo']} (next: download only, "
                          f">{config.MAX_PRELOAD_B:.0f}B)", flush=True)
            else:
                out = provider.preload(nxt["hf_repo"])
                if not out.get("cached"):
                    print(f"  preload {nxt['hf_repo']} (next)", flush=True)
        except Exception as exc:  # best effort, the run must not depend on it
            print(f"  warm-up skipped: {exc}", flush=True)
        return


def _prune(model: dict) -> None:
    if model["runtime"] != "mlx":
        return
    provider = providers.get("mlx")
    try:
        provider.prune(model["hf_repo"])
        print(f"  pruned weights {model['hf_repo']}", flush=True)
    except Exception as exc:  # pruning is best effort, never fails a run
        print(f"  prune skipped: {exc}", flush=True)


def cmd_board(args) -> None:
    for name, paths in board.render(args.dataset).items():
        print(f"[board] {name}: {paths['md']} · {paths['html']}")


def cmd_status(args) -> None:
    data = registry.load()
    print(f"registry: {data['counts']} (built {data['generated']})")
    for runtime in ("mlx", "ollama"):
        provider = providers.get(runtime)
        print(f"{runtime:7s} {provider.base_url:32s} "
              f"{'up' if provider.healthy() else 'DOWN'}")
    for name in ("dev", "val"):
        path = config.dataset_dir(name)
        if path.exists():
            ds = dataset.load(name)
            print(f"dataset {name}: {len(ds)} tables {ds.tiers()}")
    rows = store.runs()
    if rows:
        print(f"\n{'model':28s} {'set':4s} {'FINAL':>6s} {'TEDS':>6s} {'QA':>6s} {'s/tab':>6s}")
        for row in rows:
            print(f"{row['model_id']:28s} {row['dataset']:4s} {row['final'] or 0:6.4f} "
                  f"{row['teds_weighted'] or 0:6.4f} {row['qa_accuracy'] or 0:6.4f} "
                  f"{(row['s_per_table'] or 0):6.1f}")
    else:
        print("\nno runs yet")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pipeline", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("registry", help="build or inspect the model registry")
    reg_sub = reg.add_subparsers(dest="action", required=True)
    build = reg_sub.add_parser("build", help="rebuild registry/models.json from OCRBench v2")
    build.add_argument("--offline", action="store_true", help="skip HF repo checks")
    build.set_defaults(func=cmd_registry)

    lst = sub.add_parser("list", help="list registry entries")
    lst.add_argument("--status", default=None)
    lst.set_defaults(func=cmd_list)

    run = sub.add_parser("run", help="run, convert and score models")
    run.add_argument("--model", action="append", default=[], help="id or substring, repeatable")
    run.add_argument("--all", action="store_true", help="every ready model")
    run.add_argument("--dataset", default="dev")
    run.add_argument("--limit", type=int, default=None, help="first N tables only")
    run.add_argument("--force", action="store_true", help="re-run and re-score")
    run.add_argument("--keep-model", action="store_true", help="do not delete weights after scoring")
    run.add_argument("--rescore", action="store_true",
                     help="re-convert and re-score existing raw output, no inference")
    run.add_argument("--generate", action="store_true", help="generate the dataset if missing")
    run.add_argument("--n", type=int, default=240)
    run.add_argument("--seed", type=int, default=2)
    run.set_defaults(func=cmd_run)

    b = sub.add_parser("board", help="render LEADERBOARD.md and leaderboard.html")
    b.add_argument("--dataset", default=None)
    b.set_defaults(func=cmd_board)

    s = sub.add_parser("status", help="endpoints, datasets, scored models")
    s.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    if getattr(args, "model", None) and not args.all and not args.model:
        pass
    args.func(args)


if __name__ == "__main__":
    main()
