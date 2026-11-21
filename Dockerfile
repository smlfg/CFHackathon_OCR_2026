FROM python:3.12-slim

# Tesseract is the floor baseline (runtime=tesseract); it runs in-container on CPU.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TABLESALT_ROOT=/app

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline ./pipeline
COPY baselines ./baselines
COPY registry ./registry
COPY scripts ./scripts
COPY score_all.py evaluate.py eval_qa.py generate_dataset.py ./

ENTRYPOINT ["python", "-m", "pipeline"]
CMD ["status"]
