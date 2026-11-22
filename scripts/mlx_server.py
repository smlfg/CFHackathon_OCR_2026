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

STATE: dict = {"model": None, "processor": None, "config": None, "repo": None, "load_seconds": None}
LOCK = threading.Lock()
APP = FastAPI(title="tablesalt mlx server")


class LoadRequest(BaseModel):
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
    return {
        "ok": True,
        "repo": STATE["repo"],
        "loaded": STATE["model"] is not None,
        "load_seconds": STATE["load_seconds"],
        "cache": os.environ.get("HF_HUB_CACHE", ""),
    }


@APP.post("/load")
def load_model(request: LoadRequest):
    with LOCK:
        if STATE["repo"] == request.repo and STATE["model"] is not None:
            return {"loaded": request.repo, "seconds": 0, "cached": True}
        _evict()
        load, _, _, load_config = _imports()
        started = time.time()
        try:
            model, processor = load(request.repo)
            config = load_config(request.repo)
        except Exception as exc:  # surface the real reason to the orchestrator
            STATE.update({"model": None, "processor": None, "config": None, "repo": None})
            raise HTTPException(
                status_code=500, detail=f"load failed: {type(exc).__name__}: {exc}"
            ) from exc
        STATE.update(
            {
                "model": model,
                "processor": processor,
                "config": config,
                "repo": request.repo,
                "load_seconds": round(time.time() - started, 1),
            }
        )
        return {"loaded": request.repo, "seconds": STATE["load_seconds"], "cached": False}


@APP.post("/prune")
def prune(request: PruneRequest):
    with LOCK:
        if STATE["repo"] == request.repo:
            _evict()
    deleted = 0
    if request.delete_weights:
        cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
        snapshot = cache / ("models--" + request.repo.replace("/", "--"))
        if snapshot.exists():
            deleted = sum(f.stat().st_size for f in snapshot.rglob("*") if f.is_file())
            shutil.rmtree(snapshot, ignore_errors=True)
    return {"pruned": request.repo, "freed_gb": round(deleted / 1e9, 2)}


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
            formatted = apply_chat_template(
                STATE["processor"], STATE["config"], prompt, num_images=len(images)
            )
        except Exception:
            formatted = prompt
        started = time.time()
        try:
            result = generate(
                STATE["model"],
                STATE["processor"],
                formatted,
                images[0],
                max_tokens=parallel,
                temperature=temperature,
                verbose=False,
            )
        except TypeError:  # older signatures want keyword arguments
            result = generate(
                STATE["model"],
                STATE["processor"],
                prompt=formatted,
                image=images[0],
                max_tokens=parallel,
                temperature=temperature,
                verbose=False,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"generate failed: {type(exc).__name__}: {exc}"
            ) from exc
        seconds = round(time.time() - started, 2)
    text = getattr(result, "text", None)
    if text is None:
        text = result[0] if isinstance(result, tuple) else str(result)
    usage = {
        "prompt_tokens": int(getattr(result, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(result, "generation_tokens", 0) or 0),
        "seconds": seconds,
    }
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "model": STATE["repo"],
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": usage,
    }


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
