# OCR eval pipeline

Runs every open-source OCR/VLM under 15B over the TableSalt dev set, scores each one with the
official scorer, and keeps a leaderboard. One command, resumable, and honest about what it skipped.

```
data -> run model -> convert to HTML -> score -> store -> leaderboard -> next model
```

## Why the split

| part | where | why |
|---|---|---|
| orchestration, conversion, scoring, metrics | **container** | reproducible, no host Python state |
| inference | **host** (`scripts/mlx_server.py`) | Docker on macOS has no Metal; CPU inference is 10–50× slower |
| Tesseract floor | **container** (`runtime: tesseract`) | CPU-only, ~0.2 s/table, no weights to ship |

The container only talks HTTP (`MLX_BASE_URL`, `OLLAMA_BASE_URL`) to the two model runtimes, so the
same image runs on a CUDA box later with a different endpoint. The Tesseract baseline runs in-process.

## The objective, everywhere

Every number this pipeline prints is the official competition score, taken from `evaluate.py`
(`W_TEDS`, `W_QA`, `TIER_WEIGHTS`) — not re-implemented:

```
S = 0.85 · TEDS_w + 0.15 · QA          TEDS_w = 0.10·T0 + 0.20·T1 + 0.30·T2 + 0.40·T3
tie-break: CellF1
```

Degradation tiers: T0 clean (10%), T1 scanner (20%), T2 phone photo (30%), T3 fax/archive (40%).

Per run this is itemised, never just a total:

- `LEADERBOARD.md` / `leaderboard.html` → "Objective decomposition" table: `T0×0.10` … `T3×0.40`,
  `TEDS_w`, `0.85×TEDS_w`, `QA`, `0.15×QA`, `S`.
- `metrics/runs/<run_id>.html` → the same line printed above the per-table view.
- `explain` prints the arithmetic for any run.
- `verify` recomputes **every stored run** with `score_all.py` itself and fails on any drift.

```bash
docker compose run --rm eval explain --model mineru
docker compose run --rm eval verify
```

`verify` output on the current dev board: 23 runs recomputed, 0 drifted.

## Usage

```bash
# once
# HF_HUB_DISABLE_SHARED_BLOBS keeps weights inside each models--<repo>/ dir, so /prune
# frees them with a plain rmtree. hub >= 2 otherwise leaves them in hub/blobs/ (GC'd
# by /prune via the .refs manifests as well, but this is the cheaper belt-and-braces).
HF_HUB_DISABLE_SHARED_BLOBS=1 \
  uv run --with mlx-vlm --with fastapi --with uvicorn python scripts/mlx_server.py --port 8099
docker compose up -d --build

# registry: every open-source model <=15B on OCRBench v2, plus the OCR specialists
docker compose run --rm eval registry build

# one model, or all of them
docker compose run --rm eval run --model paddleocr-vl --dataset dev
docker compose run --rm eval run --all --dataset dev

# floor baseline: plain Tesseract, no optimization (ships in the image)
docker compose run --rm eval run --model tesseract-baseline --dataset dev

# inspect
docker compose run --rm eval status
docker compose run --rm eval list --status ready
docker compose run --rm eval board
open metrics/leaderboard.html
```

`run` is idempotent: a model that already has a scored run is skipped unless `--force`.
`--rescore` re-converts and re-scores existing raw output without touching the model — use it
after fixing a converter.

## What a run produces

```
raw/<model>/<table_id>.txt      exact model text, never overwritten by conversion
raw/<model>/meta.json           seconds, tokens, chars per table
predictions/<model>/<table_id>.html
metrics/metrics.sqlite          runs + per-table rows
metrics/runs.jsonl              append-only history
metrics/LEADERBOARD.md          ranking table
metrics/leaderboard.html        sortable board, click a model for per-table detail
metrics/runs/<run_id>.html      image | ground truth | prediction for all 24 tables
```

Per run: `final, teds_weighted, qa_accuracy, cell_f1, t0..t3, s_per_table, tokens, empty_rate,
git_sha, prompt_hash, quant, max_edge, fmt, adapter_flag`.

## Fairness rules baked into the code

- Same images, same decode path in the container; only `max_edge` differs per model and is stored
  in the run row.
- Temperature 0, `max_tokens` from the registry, one image per request.
- Every model gets a prompt that asks for an HTML table; OCR specialists keep their native prompt
  and native format. The prompt text is stored in the run row.
- **Adapter gate**: if more than 20% of tables do not convert, or the median table has fewer than
  2 cells, the run is marked `ADAPTER_SUSPECT` and kept off the ranking. A converter bug must never
  look like a bad model.
- **Regression anchors**: `paddleocr-vl` must reproduce 0.957 and `lighton-ocr-3-0.8b` 0.9177 on
  dev (both verified from the existing `ranking.json` numbers). If those drift, the pipeline is
  wrong, not the models.

## Model registry

`registry/manual.json` holds the OCR specialists (repo, prompt, output format, image cap), the
Tesseract floor (`runtime: tesseract`, no weights) and the matching rules for OCRBench names.
`registry/models.json` is generated:

```bash
docker compose run --rm eval registry build [--offline]
```

Filter: `Open-source = Yes`, total and active parameters ≤ 15B. Candidates with no MLX/GGUF build
stay in the registry as `no_runtime` with a reason, so the board shows what was skipped.

To add a model: add one entry to `registry/manual.json` (specialists) or a `match` rule
(OCRBench), then rebuild and run it.

## Disk

Weights are downloaded by the host MLX server into `HF_HUB_CACHE` (default
`~/.cache/huggingface/hub`) and **deleted right after the model is scored** (`--keep-model` keeps
them). Before each run the pipeline checks the model's download size against free space and records
`no_space` instead of filling the disk. Raw text and predictions stay, at a few hundred KB per model.

`/prune` also collects the payloads it just unlinked from the cache-wide shared blob store that
`huggingface_hub` >= 2 uses, so `freed_gb` is real; run the server with
`HF_HUB_DISABLE_SHARED_BLOBS=1` to keep weights in per-repo dirs instead.

## Overlap (hiding the load)

Two warm-ups run while the previous model infers, both off the critical path:

- **preload** — models up to `TABLESALT_MAX_PRELOAD` (default 12B) are loaded into a second slot in
  the server (`POST /preload`), so the next `/load` swaps in instantly instead of paying a 2–6 minute
  cold start. That start is the real cost: a 3B 8-bit model needs ~130 s to load, an 8–9B 4-bit one
  200–260 s, and it was ~95 % of the measured pre-inference window (a full queue spent ~56 min there).
- **prefetch** — bigger models only get their download warmed (`POST /prefetch`).

Both are best effort: a failure just means the next `load` pays the cost itself. Pruning a repo also
evicts it from the second slot, and `/health` shows `next` alongside the loaded model.

## Output formats

`pipeline/convert.py` is the only place that knows dialects: `otsl` (PaddleOCR-VL, MinerU),
`latex`, `markdown`, `doctags` (granite-docling, SmolDocling), `html`, `auto`. If a model's output
does not convert, the board says so instead of silently scoring zero.

## Tesseract floor (non-optimized)

`runtime: tesseract` runs `baselines/baseline_tesseract.py` in-process: Tesseract's TSV words are
clustered into rows by y-center and columns by x-gap, then written as a plain grid `<table>`.
It has no span detection, no deskew/denoise/upscale and no post-processing — merged cells are wrong
by construction. That is the point: it is the number every model on the board has to beat
(`~0.342` on dev, `0.2 s/table`). The image is normalized to RGB before pytesseract, so the OCR
does not depend on whether the source file was JPEG or PNG. Verified inside the built image
(Debian Tesseract): `0.3441`, same 0.2 s/table — Tesseract builds differ, the organizer notes ±0.01.

## Endpoints

| variable | default | meaning |
|---|---|---|
| `MLX_BASE_URL` | `http://127.0.0.1:8099` (compose: `host.docker.internal:8099`) | host MLX server |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | host Ollama, for GGUF-only models |
| `TABLESALT_MAX_PARAMS` | `15` | registry size limit |
| `TABLESALT_MAX_EMPTY_RATE` | `0.20` | adapter gate threshold |
| `TABLESALT_TIMEOUT` | `900` | per-request timeout in seconds |
| `runtime: tesseract` | in-process | floor baseline; needs the `tesseract` binary, no endpoint or weights |

## Known limitations

- `mlx-community/DeepSeek-OCR-8bit` and `DeepSeek-OCR-2-8bit` fail to load under mlx-vlm 0.8.0:
  the repos ship no `processor_class`, so `AutoProcessor` cannot instantiate them. Recorded as
  `load_error`, not as a score.
- `mlx-community/GOT-OCR2_0-bf16` needs `torch`, `torchvision` and `verovio` in the server env.
  Not installed here because of disk pressure; add them to the `uv run --with ...` line if wanted.
- Models above ~10 GB of weights are skipped with `no_space` while the disk is near full; the board
  says so instead of pretending the model does not exist.

## Fully containerized fallback

```bash
docker compose --profile cpu up -d llamacpp   # llama.cpp server, CPU only
```

Slow (minutes per table for 8B) and only for tiny models, but it proves the pipeline is portable to
the linux/amd64 CUDA machine the challenge scores on.
