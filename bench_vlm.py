#!/usr/bin/env python3
"""Run a local MLX OCR model over a TableSalt dataset and keep the raw output.

Usage
    python bench_vlm.py --model lighton8 --out raw/lighton
    python bench_vlm.py --model paddle8  --out raw/paddle
    python bench_vlm.py --model lighton8 --prompt grounding --out raw/lighton-ground

One text file per table in --out plus a meta.json with seconds and token counts.
Nothing is interpreted here: the raw model text is what a converter gets later.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODELS = {
    # name: (hub repo, chat prompt, longest edge for the page, temperature)
    "lighton8": ("fwgpiyawudk/LightOnOCR-3-0.8B-MLX-8bit", "", 2048, 0.2),
    "paddle8": ("mlx-community/PaddleOCR-VL-8bit", "Table Recognition:", 0, 0.0),
    "dots8": ("mlx-community/dots.ocr-8bit", "document", 0, 0.0),
}


def page(path: Path, longest_edge: int):
    """PNG, RGB, longest edge capped (never upscaled) - LightOn's client behaviour."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    if longest_edge:
        w, h = im.size
        scale = longest_edge / max(w, h)
        if scale < 1:
            im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--gt", default="data/dev")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=None, help="override the model's chat prompt")
    ap.add_argument("--only", nargs="*", default=None, help="table ids, e.g. table_000")
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    repo, prompt, longest_edge, temperature = MODELS[args.model]
    if args.prompt is not None:
        prompt = args.prompt

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, processor = load(repo)
    config = load_config(repo)
    formatted = apply_chat_template(processor, config, prompt, num_images=1)

    manifest = json.loads((Path(args.gt) / "manifest.json").read_text())
    meta = []
    for entry in manifest:
        tid = entry["id"]
        if args.only and tid not in args.only:
            continue
        target = out / f"{tid}.txt"
        if target.exists():
            print(f"{tid}: cached")
            continue
        image = page(Path(args.gt) / entry["image"], longest_edge)
        started = time.time()
        result = generate(
            model, processor, formatted, image,
            max_tokens=args.max_tokens,
            temperature=temperature,
            verbose=False,
        )
        seconds = round(time.time() - started, 2)
        text = getattr(result, "text", None) or str(result)
        target.write_text(text)
        stats = {
            "id": tid,
            "tier": entry["tier"],
            "style": entry["style"],
            "has_spans": entry["has_spans"],
            "seconds": seconds,
            "prompt_tokens": int(getattr(result, "prompt_tokens", 0) or 0),
            "generation_tokens": int(getattr(result, "generation_tokens", 0) or 0),
        }
        meta.append(stats)
        print(f"{tid} {entry['tier']:<3} {seconds:6.2f}s "
              f"{stats['generation_tokens']:5d} tok -> {len(text)} chars")

    meta_path = out / "meta.json"
    previous = json.loads(meta_path.read_text()) if meta_path.exists() else []
    known = {m["id"] for m in previous}
    meta_path.write_text(json.dumps(previous + [m for m in meta if m["id"] not in known],
                                   indent=1))
    if meta:
        total = sum(m["seconds"] for m in meta)
        print(f"\n{len(meta)} tables, {total:.1f}s, {total / len(meta):.1f}s/table")


if __name__ == "__main__":
    main()
