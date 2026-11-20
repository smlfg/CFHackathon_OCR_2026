FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TABLESALT_ROOT=/app

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline ./pipeline
COPY registry ./registry
COPY scripts ./scripts
COPY score_all.py evaluate.py eval_qa.py generate_dataset.py ./

ENTRYPOINT ["python", "-m", "pipeline"]
CMD ["status"]
