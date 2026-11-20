# Published benchmark snapshot

A committed copy of the dev-set evaluation produced by `python -m pipeline run --all --dataset dev`.
The pipeline writes the same files into `metrics/`, which stays gitignored — this folder is the
curated, reviewable snapshot.

## Contents (`dev/`)

| file | what it is |
|---|---|
| `LEADERBOARD.md` | the ranking table, one row per model, per-tier TEDS, QA, CellF1, final score |
| `leaderboard.html` | the same board as a standalone page (no external assets) |
| `runs.jsonl` | append-only history, one JSON object per run: scores, timings, tokens, prompt hash, quant, `max_edge`, adapter flag |
| `metrics.sqlite` | `runs` + `tables` tables behind the board, for queries |
| `runs/*.html` | per-model report: image, ground truth and prediction for all 24 tables of one run |

Dataset: `dev`, 24 tables, 6 per degradation tier (T0 clean, T1 scanner, T2 phone photo, T3 fax/archive).
Images and ground truth are tracked under `data/dev/`, so the report pages render straight from the
repo (their `src` paths were rewritten one level up for this location).

## Scoring

Official scorer only, imported from `evaluate.py` — not re-implemented:

```
S = 0.85 · TEDS_w + 0.15 · QA        TEDS_w = 0.10·T0 + 0.20·T1 + 0.30·T2 + 0.40·T3
tie-break: CellF1
```

## Reading it honestly

- Models that failed to load or produced unparseable output are listed under *Not scored* in
  `LEADERBOARD.md` with the reason. They are not scored as zero.
- `ADAPTER_SUSPECT` means our converter could not read the model's output format (or the table came
  back empty); the run is kept off the ranking, because a converter bug must not look like a bad model.
- `s/table` is wall-clock generation time on this machine (Apple Silicon, MLX), not a portable speed
  claim. `max_edge`, `quant` and the prompt hash are stored per run so numbers stay comparable.
- Raw model text (`raw/`) and converted predictions (`predictions/`) are not committed; they are
  reproducible by re-running the pipeline.

## Regenerating

```bash
docker compose up -d --build
docker compose run --rm eval run --all --dataset dev
```

`run` is idempotent and resumable; `--rescore` re-scores existing raw output without touching a model.
