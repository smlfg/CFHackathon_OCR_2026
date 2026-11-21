# CUDA benchmark image: TATR (Table Transformer) -> Tesseract, end to end.
#
# Built on the 4070 laptop (NVIDIA driver + nvidia-container-toolkit), but the
# Dockerfile itself is platform-pinned to linux/amd64 so it can be built on the
# arm64 Mac as well:
#
#   docker buildx build --platform linux/amd64 \
#     -f benchmarks/cuda/Dockerfile -t tablesalt-tatr-cuda:amd64 --load .
#
# Weights are baked in (the official run is offline): TATR structure model,
# Tesseract + deu/eng language data. Nothing is downloaded at run time.
#
# Run (GPU):
#   docker run --rm --gpus all \
#     -v "$PWD/data:/app/data:ro" -v "$PWD/out:/app/out" \
#     tablesalt-tatr-cuda:amd64 --gt data/dev --out out/cuda_dev
#
# Run (CPU, for comparison on the same machine):
#   docker run --rm -e TATR_DEVICE=cpu ... tablesalt-tatr-cuda:amd64 ...
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TABLESALT_ROOT=/app \
    TATR_DEVICE=cuda \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# System packages: Tesseract 5 + German/English data, libGL/libglib for OpenCV,
# poppler + PyMuPDF so PDF inputs can be rasterised inside the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-deu \
        python3 \
        python3-pip \
        python3-venv \
        libgl1 \
        libglib2.0-0 \
        poppler-utils \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies. Torch comes from the CUDA wheel index; the rest are
# pinned to the same versions the repo's venvs use so the numbers transfer.
#
# Retries: the torch wheel is ~2.5 GB and a long download over a flaky link
# ends in "SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC" more often than anyone
# likes. pip's own retry does not cover a mid-stream TLS record failure, so we
# retry the whole install.
COPY benchmarks/cuda/requirements.txt /tmp/requirements.txt
RUN set -eux; \
    for attempt in 1 2 3 4 5; do \
        echo "=== pip install attempt $attempt ==="; \
        if python3 -m pip install --no-cache-dir --upgrade pip \
            && python3 -m pip install --no-cache-dir \
                --extra-index-url https://download.pytorch.org/whl/cu124 \
                -r /tmp/requirements.txt; then \
            break; \
        fi; \
        if [ "$attempt" = "5" ]; then echo "pip failed 5 times" >&2; exit 1; fi; \
        echo "retrying in 15s..."; sleep 15; \
    done; \
    python3 -c "import torch, transformers, pytesseract; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# ---- bake the model weights in -------------------------------------------
# The structure model is 110 MB fp32. Downloading it at build time is the only
# network access this image ever needs; the run itself stays offline.
#
# HF_HOME must already point at /app/models here, because the timm backbone
# (timm/resnet18.a1_in1k) has to land in the standard hub/ cache layout —
# that is where timm looks it up by repo id when it builds the ResNet-18.
ENV HF_HOME=/app/models

COPY benchmarks/cuda/fetch_weights.py /tmp/fetch_weights.py
RUN python3 /tmp/fetch_weights.py \
    && rm -f /tmp/fetch_weights.py /tmp/requirements.txt

# ---- code ------------------------------------------------------------------
COPY scripts/my_tatr_fusion.py      /app/scripts/my_tatr_fusion.py
COPY scripts/tatr_experiment.py     /app/scripts/tatr_experiment.py
COPY benchmarks/cuda/bench_tatr_cuda.py /app/bench_tatr_cuda.py
COPY score_all.py evaluate.py eval_qa.py /app/

ENV PYTHONPATH=/app:/app/scripts

# From here on the image is self-contained: the checkpoint is baked in, so the
# run must not reach the network. Offline mode is switched on only AFTER the
# weights were fetched, otherwise the build itself cannot download them.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Fail loudly if the checkpoint did not make it into the image, or if a runtime
# dependency is missing (e.g. timm for the ResNet-18 backbone). Loading the model
# here is the only way to know the image will not fail mid-benchmark.
#
# This loads the model with the network off, so it proves that BOTH weight sets
# are baked in: the detector AND the timm ResNet-18 backbone.
RUN python3 -c "\
from pathlib import Path; \
from transformers import TableTransformerForObjectDetection; \
d = Path('/app/models/table-transformer-structure-recognition'); \
assert d.is_dir(), 'TATR checkpoint missing: ' + str(d); \
print('checkpoint files:', sorted(p.name for p in d.iterdir())); \
m = TableTransformerForObjectDetection.from_pretrained(str(d), local_files_only=True); \
print('TATR loads offline:', round(sum(p.numel() for p in m.parameters()) / 1e6, 1), 'M params')"

ENTRYPOINT ["python3", "/app/bench_tatr_cuda.py"]
CMD ["--gt", "data/dev", "--out", "out/cuda_dev"]
