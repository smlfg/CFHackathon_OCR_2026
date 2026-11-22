# Submission contract: the container must run
#   <entrypoint> /data/images /data/predictions
# offline (no network). Build with the repository root as context:
#   docker build -t my-tablesalt -f submission_template/Dockerfile .
#
# Pipeline: tablesalt/ (OpenCV structure + Tesseract cell OCR), fallback baselines/.
# base pinned by digest (resolved 2026-11-22) so the build after the deadline is the build we tested
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu && rm -rf /var/lib/apt/lists/*
COPY submission_template/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /app
COPY tablesalt/ /app/tablesalt/
COPY baselines/ /app/baselines/
COPY submission_template/ /app/submission_template/
# one Tesseract thread per process; parallelism comes from one worker per table
ENV OMP_THREAD_LIMIT=1 \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "submission_template/predict.py", "/data/images", "/data/predictions"]
