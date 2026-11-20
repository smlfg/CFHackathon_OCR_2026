#!/usr/bin/env python3
"""Host-side MLX model server for the TableSalt eval pipeline.

Docker on macOS has no Metal, so the container orchestrates and this process
does the inference. It exposes the smallest useful surface:

    GET  /health
    POST /load            {"repo": "mlx-community/PaddleOCR-VL-8bit"}
    POST /prune           {"repo": "...", "delete_weights": true}
    POST /v1/chat/completions   OpenAI shape, image as data URL

Run it:
    uv run --with mlx-vlm --with fastapi --with uvicorn \
        python scripts/mlx_server.py --port 8099

It listens on 0.0.0.0 so Docker can reach it via host.docker.internal. It has no
authentication: keep it on your machine and stop it when you are done.
"""
from __future__ import annotations

import argparse
import base64
import gc
import io
import os
import re
import shutil
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

STATE: dict = {"model": None, "processor": None, "config": None, "repo": None,
               "load_seconds": None}
# Warm-load of the next model's weights while the current one is inferring.
PREFETCH: dict = {"repo": None, "state": "idle", "error": None}
# Second residency slot: the next model, fully loaded, waiting for /load to swap it in.
SECOND: dict = {"model": None, "processor": None, "config": None, "repo": None,
                "state": "idle", "error": None, "seconds": None}
SECOND_LOCK = threading.Lock()
LOCK = threading.Lock()
APP = FastAPI(title="tablesalt mlx server")


class LoadRequest(BaseModel):
    repo: str


class PrefetchRequest(BaseModel):
    repo: str


class PruneRequest(BaseModel):
    repo: str
    delete_weights: bool = True


def _imports():
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    return load, generate, apply_chat_template, load_config


@APP.get("/health")
def health():
    return {"ok": True, "repo": STATE["repo"], "loaded": STATE["model"] is not None,
            "load_seconds": STATE["load_seconds"], "cache": os.environ.get("HF_HUB_CACHE", ""),
            "prefetch": {k: PREFETCH[k] for k in ("repo", "state", "error")},
            "next": {k: SECOND[k] for k in ("repo", "state", "error", "seconds")}}


@APP.post("/preload")
def preload(request: PrefetchRequest):
    """Load the next model into the second slot while the current one is inferring."""
    if SECOND["repo"] == request.repo and SECOND["state"] in ("loading", "ready"):
        return {"preload": request.repo, "state": SECOND["state"], "cached": True}
    if STATE["repo"] == request.repo and STATE["model"] is not None:
        return {"preload": request.repo, "state": "current", "cached": True}
    with SECOND_LOCK:
        _evict_second()
        SECOND.update({"repo": request.repo, "state": "loading", "error": None})
    threading.Thread(target=_preload_worker, args=(request.repo,), daemon=True).start()
    return {"preload": request.repo, "state": "loading", "cached": False}


def _preload_worker(repo: str) -> None:
    started = time.time()
    try:
        load, _, _, load_config = _imports()
        model, processor = load(repo)
        cfg = load_config(repo)
        with SECOND_LOCK:
            if SECOND["repo"] != repo:  # superseded while loading: drop the work
                return
            SECOND.update({"model": model, "processor": processor, "config": cfg,
                           "state": "ready", "seconds": round(time.time() - started, 1)})
    except Exception as exc:  # a failed preload must never fail the run
        with SECOND_LOCK:
            if SECOND["repo"] == repo:
                SECOND.update({"model": None, "processor": None, "config": None,
                               "state": "failed",
                               "error": f"{type(exc).__name__}: {exc}"[:400]})


def _take_second(repo: str) -> tuple[bool, float]:
    """Swap a preloaded model in. Returns (swapped, seconds_the_load_was_hidden)."""
    deadline = time.time() + 900
    while True:
        with SECOND_LOCK:
            if SECOND["repo"] != repo:
                return False, 0.0
            if SECOND["state"] == "ready":
                hidden = SECOND.get("seconds") or 0.0
                _evict()
                STATE.update({k: SECOND[k] for k in ("model", "processor", "config", "repo")})
                STATE["load_seconds"] = 0.0
                _evict_second()
                return True, hidden
            if SECOND["state"] == "failed":
                # best effort: drop it and let the caller do a normal synchronous load
                _evict_second()
                return False, 0.0
        if time.time() > deadline:  # still loading: fall through to a synchronous load
            return False, 0.0
        time.sleep(0.5)


def _evict_second() -> None:
    SECOND.update({"model": None, "processor": None, "config": None, "repo": None,
                   "state": "idle", "error": None, "seconds": None})
    gc.collect()


@APP.post("/load")
def load_model(request: LoadRequest):
    with LOCK:
        if STATE["repo"] == request.repo and STATE["model"] is not None:
            return {"loaded": request.repo, "seconds": 0, "cached": True}
        swapped, hidden = _take_second(request.repo)
        if swapped:
            return {"loaded": request.repo, "seconds": 0, "cached": True,
                    "preloaded": True, "load_hidden_seconds": hidden}
        _evict()
        load, _, _, load_config = _imports()
        started = time.time()
        try:
            model, processor = load(request.repo)
            config = load_config(request.repo)
        except Exception as exc:  # surface the real reason to the orchestrator
            STATE.update({"model": None, "processor": None, "config": None, "repo": None})
            raise HTTPException(status_code=500, detail=f"load failed: {type(exc).__name__}: {exc}")
        STATE.update({"model": model, "processor": processor, "config": config,
                      "repo": request.repo, "load_seconds": round(time.time() - started, 1)})
        return {"loaded": request.repo, "seconds": STATE["load_seconds"], "cached": False}


@APP.post("/prefetch")
def prefetch(request: PrefetchRequest):
    """Warm the HF cache for the next model while the current one is still inferring."""
    if PREFETCH["repo"] == request.repo and PREFETCH["state"] in ("running", "done"):
        return {"prefetch": request.repo, "state": PREFETCH["state"], "cached": True}
    PREFETCH.update({"repo": request.repo, "state": "running", "error": None})
    threading.Thread(target=_prefetch_worker, args=(request.repo,), daemon=True).start()
    return {"prefetch": request.repo, "state": "running", "cached": False}


def _prefetch_worker(repo: str) -> None:
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo)
        PREFETCH.update({"repo": repo, "state": "done", "error": None})
    except Exception as exc:  # a failed warm-up must never fail the run
        PREFETCH.update({"repo": repo, "state": "failed",
                         "error": f"{type(exc).__name__}: {exc}"[:400]})


@APP.post("/prune")
def prune(request: PruneRequest):
    with LOCK:
        if STATE["repo"] == request.repo:
            _evict()
    with SECOND_LOCK:
        if SECOND["repo"] == request.repo:
            _evict_second()
    deleted = 0
    if request.delete_weights:
        cache = Path(os.environ.get("HF_HUB_CACHE",
                                    Path.home() / ".cache" / "huggingface" / "hub"))
        snapshot = cache / ("models--" + request.repo.replace("/", "--"))
        # huggingface_hub >= 2 keeps the weights in a cache-wide shared blob store and
        # leaves only symlinks here, so resolve the store paths before removing the repo.
        store_paths = [Path(os.path.realpath(link)) for link in snapshot.rglob("*")
                       if link.is_symlink()] if snapshot.exists() else []
        if snapshot.exists():
            deleted = sum(f.stat().st_size for f in snapshot.rglob("*")
                          if f.is_file() and not f.is_symlink())
            shutil.rmtree(snapshot, ignore_errors=True)
        deleted += _sweep_shared(cache, store_paths)
    return {"pruned": request.repo, "freed_gb": round(deleted / 1e9, 2)}


def _sweep_shared(cache: Path, store_paths: list[Path]) -> int:
    """Collect shared-store payloads whose last reference was just removed."""
    try:
        from huggingface_hub.utils._shared_blobs import sweep_shared_blob
    except Exception:  # older hub without a shared store: nothing to do
        return 0
    freed = 0
    for path in dict.fromkeys(store_paths):
        try:
            freed += sweep_shared_blob(path, cache_dir=cache)
        except Exception:
            pass
    return freed


@APP.post("/v1/chat/completions")
def chat_completions(payload: dict):
    if STATE["model"] is None:
        raise HTTPException(status_code=409, detail="no model loaded; POST /load first")
    prompt, images = _parse_messages(payload.get("messages") or [])
    if not images:
        raise HTTPException(status_code=400, detail="no image in request")
    parallel = int(payload.get("max_tokens") or 4096)
    temperature = float(payload.get("temperature") or 0.0)
    _, generate, apply_chat_template, _ = _imports()
    with LOCK:
        try:
            formatted = apply_chat_template(STATE["processor"], STATE["config"], prompt,
                                            num_images=len(images))
        except Exception:
            formatted = prompt
        started = time.time()
        try:
            result = generate(STATE["model"], STATE["processor"], formatted, images[0],
                              max_tokens=parallel, temperature=temperature, verbose=False)
        except TypeError:  # older signatures want keyword arguments
            result = generate(STATE["model"], STATE["processor"], prompt=formatted,
                              image=images[0], max_tokens=parallel,
                              temperature=temperature, verbose=False)
        except Exception as exc:
            raise HTTPException(status_code=500,
                                detail=f"generate failed: {type(exc).__name__}: {exc}")
        seconds = round(time.time() - started, 2)
    text = getattr(result, "text", None)
    if text is None:
        text = result[0] if isinstance(result, tuple) else str(result)
    usage = {
        "prompt_tokens": int(getattr(result, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(result, "generation_tokens", 0) or 0),
        "seconds": seconds,
    }
    return {"id": f"chatcmpl-{int(time.time() * 1000)}", "object": "chat.completion",
            "model": STATE["repo"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": usage}


def _parse_messages(messages: list[dict]) -> tuple[str, list]:
    from PIL import Image

    texts, images = [], []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        for block in content or []:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "image_url":
                url = (block.get("image_url") or {}).get("url", "")
                match = re.match(r"data:image/[a-z]+;base64,(.*)", url, re.S)
                raw = base64.b64decode(match.group(1) if match else url)
                images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
    return "\n".join(texts), images


def _evict() -> None:
    global STATE
    STATE.update({"model": None, "processor": None, "config": None, "repo": None})
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(APP, host=args.host, port=args.port, log_level="info", workers=1)


if __name__ == "__main__":
    main()
