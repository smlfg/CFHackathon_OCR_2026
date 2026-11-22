#!/usr/bin/env python3
"""End-to-end entrypoint: table images in -> TableSalt cells JSON out.

Same contract as the submission (§7), so it can be tested exactly like the container:

    python experiments/vlm_scripts/predict_lighton.py /path/to/images /path/to/predictions

For every `<table_id>.png|.jpg` in the input directory it writes
`<table_id>.json` with the schema from README §5. Each file is written as soon as
the table is done, so a timeout keeps everything finished so far.

The OCR model runs through MLX (Apple silicon); the container needs a CUDA/CPU
runtime of the same weights instead - see the README section on shipping.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bench_vlm import MODELS, page
from lighton_cells import check_cells, text_to_cells

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", help="directory of table images")
    ap.add_argument("predictions", help="directory for the JSON predictions")
    ap.add_argument("--model", default="lighton8", choices=sorted(MODELS))
    ap.add_argument("--prompt", default=None, help="override the model's chat prompt")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--check", action="store_true", help="validate every file")
    args = ap.parse_args()

    repo, prompt, longest_edge, temperature = MODELS[args.model]
    if args.prompt is not None:
        prompt = args.prompt

    images = Path(args.images)
    out = Path(args.predictions)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p
        for p in images.iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")
    )
    if not files:
        raise SystemExit(f"no images in {images}")

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    model, processor = load(repo)
    config = load_config(repo)
    formatted = apply_chat_template(processor, config, prompt, num_images=1)

    started_all = time.time()
    for image_path in files:
        target = out / f"{image_path.stem}.json"
        if target.exists():
            continue
        started = time.time()
        result = generate(
            model,
            processor,
            formatted,
            page(image_path, longest_edge),
            max_tokens=args.max_tokens,
            temperature=temperature,
            verbose=False,
        )
        text = getattr(result, "text", None) or str(result)
        cells = text_to_cells(text)
        note = ""
        if not cells:
            note = "  EMPTY (no table recognised)"
        elif args.check:
            problems = check_cells(cells)
            if problems:
                note = f"  {len(problems)} warning(s): {problems[0]}"
        target.write_text(json.dumps(cells, ensure_ascii=False))
        print(
            f"{image_path.name:<16} {len(cells):3d} cells {time.time() - started:5.1f}s{note}",
            flush=True,
        )

    print(f"\n{len(files)} images, {time.time() - started_all:.1f}s -> {out}")


if __name__ == "__main__":
    main()
