# Official TableSalt submission container (shared base, build-time variant lock).
#
# Organizer builds (no build args -> ships default approved variant):
#   docker build -f submission_template/Dockerfile .
#
# Controlled multi-variant builds (shares heavy CUDA + weight layers):
#   docker build --platform linux/amd64 --build-arg VARIANT=v0_tatr_naive \
#     -t tablesalt:v0 -f submission_template/Dockerfile .
#   docker build --platform linux/amd64 --build-arg VARIANT=v1_tatr_clean \
#     -t tablesalt:v1 -f submission_template/Dockerfile .
#
# Base image: minimal nvidia/cuda:12.6.3-base-ubuntu24.04 (~250 MB) + PyTorch cu124 wheel
# (~5.6 GB, bundles its own CUDA 12.4 + cuDNN) -> ~6.6 GB total (well under 15 GB cap,
# full RTX 4070 CUDA support, Ubuntu 24.04 ships Tesseract 5.3.4).
FROM nvidia/cuda:12.6.3-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_THREAD_LIMIT=1 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata \
    HF_HOME=/app/models \
    TATR_MODEL_DIR=/app/models/table-transformer-structure-recognition

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-deu \
        python3 \
        python3-pip \
        python3-venv \
        poppler-utils \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pinned Python dependencies in an isolated venv (PEP 668 compliant on Ubuntu 24.04).
COPY submission_template/requirements.txt /tmp/requirements.txt
RUN set -eux; \
    python3 -m venv /opt/venv; \
    for attempt in 1 2 3 4 5; do \
        if /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
            && /opt/venv/bin/pip install --no-cache-dir \
                --extra-index-url https://download.pytorch.org/whl/cu124 \
                -r /tmp/requirements.txt; then \
            break; \
        fi; \
        if [ "$attempt" = "5" ]; then exit 1; fi; \
        sleep 10; \
    done; \
    rm -f /tmp/requirements.txt

ENV PATH=/opt/venv/bin:$PATH

# Bake revision-pinned model weights (TATR structure + timm ResNet-18 backbone).
COPY submission_template/fetch_weights.py /tmp/fetch_weights.py
RUN python3 /tmp/fetch_weights.py && rm -f /tmp/fetch_weights.py

# Lock offline mode AFTER weights are downloaded.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/app:/app/scripts

# Minimal pipeline code (shared across all variants; no data/ copied).
COPY baselines/__init__.py /app/baselines/__init__.py
COPY baselines/baseline_tesseract.py /app/baselines/baseline_tesseract.py
COPY scripts/my_tatr_fusion.py /app/scripts/my_tatr_fusion.py
COPY scripts/coherence_postprocess.py /app/scripts/coherence_postprocess.py
COPY submission_template/predict.py /app/submission_template/predict.py
COPY submission_template/variants /app/submission_template/variants

# Build-time immutable variant selection (placed last for instant variant rebuilds).
ARG VARIANT=v1_tatr_clean
ARG GIT_SHA=unknown
ENV TABLESALT_VARIANT=${VARIANT} \
    TABLESALT_GIT_SHA=${GIT_SHA}

# Build-time preflight: prove Tesseract deu+eng, offline TATR load, and valid variant config.
RUN set -eux; \
    tesseract --version | head -1; \
    test -f "$TESSDATA_PREFIX/deu.traineddata"; \
    test -f "$TESSDATA_PREFIX/eng.traineddata"; \
    python3 -c "\
import json, os; from pathlib import Path; \
from PIL import Image, ImageDraw; import pytesseract; \
from transformers import TableTransformerForObjectDetection; \
v = os.environ['TABLESALT_VARIANT']; \
cfg = json.loads((Path('/app/submission_template/variants') / f'{v}.json').read_text()); \
print('Variant locked:', cfg['variant_id'], 'ocr_mode:', cfg['ocr_mode']); \
im = Image.new('RGB', (400, 80), 'white'); \
ImageDraw.Draw(im).text((10, 25), 'Total 1234 EUR', fill='black'); \
txt = pytesseract.image_to_string(im, lang='deu+eng', config='--psm 7').strip(); \
assert '1234' in txt and 'EUR' in txt, f'Tesseract preflight failed: {txt!r}'; \
m = TableTransformerForObjectDetection.from_pretrained('/app/models/table-transformer-structure-recognition', local_files_only=True); \
print('Offline TATR OK:', round(sum(p.numel() for p in m.parameters())/1e6, 1), 'M params')"

ENTRYPOINT ["/opt/venv/bin/python", "/app/submission_template/predict.py", "/data/images", "/data/predictions"]
