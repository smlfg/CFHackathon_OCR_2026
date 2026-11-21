#!/usr/bin/env python3
"""Download the TATR structure model into the image at build time.

The official submission run is offline, so every weight the pipeline needs has
to be baked into the image. This runs once, during `docker build`, and it is the
only step in the whole pipeline that touches the network.

Two weight sets are needed, and it is easy to miss the second one:

  1. microsoft/table-transformer-structure-recognition
     The detector itself (28.8M params, DETR with a ResNet-18 backbone).
     Saved as a plain directory, so `from_pretrained(<dir>)` finds it.

  2. timm/resnet18.a1_in1k
     The backbone. `TableTransformerConvEncoder` builds its ResNet-18 through
     timm, and timm fetches the ImageNet weights from the Hub at construction
     time. Without this, the model cannot be constructed offline — which is
     exactly what the build-time preflight check caught.
     This one must land in the *standard HF cache layout* (HF_HOME/hub/...), not
     a custom directory, because that is where timm looks it up by repo id.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

MODELS = [
    # (repo id, destination directory name)
    ("microsoft/table-transformer-structure-recognition",
     "table-transformer-structure-recognition"),
    ("timm/resnet18.a1_in1k", "resnet18.a1_in1k"),
]
DEST = Path("/app/models")
# timm resolves its backbone through the normal HF cache, so it has to be
# cached in the hub/ layout, not in a flat directory.
HF_HOME = Path("/app/models")
CACHE_HUB = HF_HOME / "hub"


def main() -> int:
    from huggingface_hub import snapshot_download

    DEST.mkdir(parents=True, exist_ok=True)
    CACHE_HUB.mkdir(parents=True, exist_ok=True)
    ok = True
    for repo_id, name in MODELS:
        # The timm backbone must sit in the standard hub/ cache layout, because
        # timm looks it up by repo id. The detector is loaded by path, so a
        # flat directory is fine (and simpler to point at).
        cache_only = repo_id.startswith("timm/")
        if cache_only:
            print(f"[weights] fetching {repo_id} into the HF cache ...", flush=True)
            path = snapshot_download(repo_id=repo_id, cache_dir=str(CACHE_HUB))
        else:
            target = DEST / name
            print(f"[weights] fetching {repo_id} ...", flush=True)
            path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(target),
                allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt"],
            )
            # snapshot_download leaves a .cache/huggingface bookkeeping dir behind.
            # It is useless at runtime and only makes the image bigger.
            shutil.rmtree(Path(path) / ".cache", ignore_errors=True)

        files = sorted(p for p in Path(path).rglob("*")
                       if p.is_file() and ".cache" not in p.parts)
        total = sum(p.stat().st_size for p in files)
        print(f"[weights] {repo_id} -> {path}")
        for p in files[:12]:
            print(f"           {p.stat().st_size / 1e6:8.2f} MB  {p.relative_to(path)}")
        print(f"[weights]   total {total / 1e6:.1f} MB ({len(files)} files)")

        has_weights = any(p.suffix in (".safetensors", ".bin")
                           for p in Path(path).rglob("*")
                           if p.is_file() and ".cache" not in p.parts)
        if not has_weights:
            print(f"[weights] ERROR: {repo_id} has no weight file", file=sys.stderr)
            ok = False

    if not ok:
        return 1
    print("[weights] all weight sets present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
