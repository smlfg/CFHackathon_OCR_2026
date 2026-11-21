#!/usr/bin/env python3
"""Fetch revision-pinned TATR + ResNet-18 weights into /app/models at Docker build time.

Both Hugging Face repositories are pinned to exact commit SHAs so builds after
the deadline reproduce identical weights offline.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

MODELS = [
    (
        "microsoft/table-transformer-structure-recognition",
        "f4d4bdc85c3fe4b1fa49658882a5d38bbdd0f343",
        "table-transformer-structure-recognition",
        False,
    ),
    (
        "timm/resnet18.a1_in1k",
        "491b427b45c94c7fb0e78b5474cc919aff584bbf",
        "resnet18.a1_in1k",
        True,
    ),
]

DEST = Path("/app/models")
CACHE_HUB = DEST / "hub"


def main() -> int:
    from huggingface_hub import snapshot_download

    DEST.mkdir(parents=True, exist_ok=True)
    CACHE_HUB.mkdir(parents=True, exist_ok=True)

    ok = True
    for repo_id, revision, name, cache_only in MODELS:
        print(f"[weights] {repo_id} @ {revision[:12]} ...", flush=True)
        if cache_only:
            path = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=str(CACHE_HUB),
            )
            # timm looks up revision=None -> refs/main in offline mode; pin refs/main
            # explicitly to the immutable revision SHA we just downloaded.
            repo_cache = CACHE_HUB / f"models--{repo_id.replace('/', '--')}"
            refs_main = repo_cache / "refs" / "main"
            refs_main.parent.mkdir(parents=True, exist_ok=True)
            refs_main.write_text(revision + "\n", encoding="utf-8")
        else:
            target = DEST / name
            path = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=str(target),
                allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt"],
            )
            shutil.rmtree(Path(path) / ".cache", ignore_errors=True)

        files = [p for p in Path(path).rglob("*") if p.is_file() and ".cache" not in p.parts]
        total_mb = sum(p.stat().st_size for p in files) / 1e6
        has_weights = any(p.suffix in (".safetensors", ".bin") for p in files)
        print(f"[weights]   -> {path} ({len(files)} files, {total_mb:.1f} MB)", flush=True)
        if not has_weights:
            print(f"[weights] ERROR: no weight file found for {repo_id}", file=sys.stderr)
            ok = False

    # Warm-load TATR once online so timm populates any exact hub cache metadata it expects offline.
    from transformers import TableTransformerForObjectDetection
    tatr_dir = DEST / "table-transformer-structure-recognition"
    m = TableTransformerForObjectDetection.from_pretrained(str(tatr_dir))
    print(f"[weights] warm-load OK: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params", flush=True)
    print(f"[weights] hub cache entries: {sorted(p.name for p in CACHE_HUB.iterdir())}", flush=True)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
