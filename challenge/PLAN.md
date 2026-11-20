# Preparation plan

Order of work for the TableSalt challenge. Rule of the whole plan: **every step produces a number,
not an opinion** → score on `data/dev` (fixed yardstick) and on a generated validation set (for
decisions). Nothing gets added to the pipeline that did not move a number or fix a named error class.

Milestones in order; do not start the next one before the current one has evidence.

## M0 — Ground truth on our machine (≈40 min)

Goal: know exactly where we stand, on the hardware we have, with our own Tesseract build.

```bash
source .venv/bin/activate                     # .venv already exists in this repo
pip install -r requirements.txt pytesseract
tesseract --version                           # must exist; macOS: brew install tesseract

python selftest_workflow.py --gt data/dev     # must end with ALL INVARIANTS HOLD
python baselines/baseline_tesseract.py --gt data/dev --out predictions_tess/
python score_all.py --gt data/dev --pred predictions_tess/     # expect ≈0.348
python evaluate.py  --gt data/dev --pred predictions_tess/     # per-table TEDS, structure, cell F1
python gallery.py                              # gallery.html: images + GT + QA side by side
```

Evidence to keep: the exact score line (final, per-tier TEDS, QA) written into
[`MEASUREMENTS.md`](#measurement-log) — today's floor number.

Acceptance: we have a final score, per-tier TEDS, QA accuracy, and a browsable gallery.

## M1 — Understand the failures, not the pipeline (≈1–2 h)

Goal: name the error classes before choosing a model. This is also what the organizer said he wants
to see ("ask AI to understand why it is failing", "solve each problem one by one").

- Open the 8 worst tables from `evaluate.py` in `gallery.html`; compare image ↔ GT ↔ our cells.
- Tag each failure with exactly one primary cause: `structure` (rows/cols/spans wrong) · `text`
  (characters wrong) · `number-format` (separator/sign/unit) · `missing-empty-cells` ·
  `row-label` (QA can't find the row) · `header-row` (QA can't find the column) · `preprocessing`.
- Split the damage per tier (T0–T3) and per style (grid / hlines / borderless). That matrix tells us
  where the weight is: a fix in T3·borderless is worth ~3× the same fix in T1·grid.
- Write the findings as bullets in [`ERRORS.md`](ERRORS.md) with table id + one screenshot crop each.

Acceptance: a ranked list of the 3 error classes that cost the most score, each with an example.

## M2 — A validation set that can carry decisions (≈15 min, machine time)

Six tables per tier on dev is noise. Generate a decision set with the shipped generator (Arial is
present on macOS; on Linux set `TABLESALT_FONT`).

```bash
python generate_dataset.py --out data/val  --n 240 --seed 2    # 60 per tier
python generate_dataset.py --out data/train --n 5000 --seed 1  # only if we fine-tune
```

- `data/val` = decision set (never train on it). `data/dev` stays untouched as the official yardstick.
- Also generate 24 clean tables with `--clean` if we want to rehearse the print-and-rescan check
  (`score_all.py --real`, reported separately, never in the score).

Acceptance: `score_all.py --gt data/val` works and the number is within ±0.02 of dev for the floor.

## M3 — Architecture, one seam at a time

Pipeline shape that keeps every part measurable:

```
image → [load: png/jpg/pdf, page+table region] → [preprocess: deskew, perspective, illumination,
binarize] → [structure: rows, cols, spans, empty cells] → [text: OCR per cell crop]
→ [postprocess: number format per table, span repair, sign/unit cleanup] → cells.json
```

Candidates, cheapest first. Run each on `data/val`, keep the score table:

| # | Candidate | Effort | Expected | Notes |
|---|---|---|---|---|
| 1 | Tesseract baseline, improved clustering + number post-processing | low | ~0.4 | proves the harness, cheap points in T0/T1 |
| 2 | PaddleOCR table pipeline (`baselines/baseline_paddleocr.py`, untuned) | low | ~0.5–0.6 | 1.3 GB weights, keep revision pinned |
| 3 | Structure model (Table Transformer / TATR) + per-cell OCR | medium | higher on T2/T3 | 8 GB VRAM, needs a batching/time budget |
| 4 | Small document VLM on GPU, crops only | medium | ? | measure 15 s/table honestly |
| 5 | Fine-tune on `data/train` (generator = exact labels) | high | best | only after 1–2 numbers exist |

Rules:
- One change at a time, always scored on both dev and val, always timed (s/table).
- Preprocessing is a module with an on/off switch, so its contribution is a number not a belief.
- Post-processing has the highest ratio of score per compute: one number format per table, spans
  enforced, empty cells materialized, umlauts/signs normalized.
- Never let one unreadable image produce zero cells: keep a cheap fallback path.

Acceptance: a filled-in score table (candidate × T0–T3 × final × s/table) and one written paragraph
why the winner won.

## M4 — Container, exactly as it will be run

```bash
python flatten_dataset.py data/dev flat
docker build --platform linux/amd64 -t tablesalt -f submission_template/Dockerfile .
docker run --rm --network none --gpus all -v "$PWD/flat:/data/images:ro" -v "$PWD/out:/data/predictions" tablesalt
python score_all.py --gt data/dev --pred out
```

- Check: works with `--network none`, no weights pulled at runtime, writes each `.json` as it goes,
  under 15 s/table, image < 15 GB.
- Test the boring failure modes: a `.pdf` input, an image with text around the table, a corrupt file
  → output must still exist and be parseable (empty is bad, crash is worse).
- Hand-in: private repo from the template, `metinferhat@outlook.com` invited as collaborator, team
  name + repo URL registered.

Acceptance: fresh clone → the four commands above → score reproduced by the container.

## M5 — Show the work (after M2, keep updating)

The organizer explicitly asked for **intermediate results** and for the reasoning behind decisions;
"the difference will be how you approached the problem".

- `challenge/ERRORS.md` — error classes with examples (M1+).
- `challenge/MEASUREMENTS.md` — every run: date, change, dev/val numbers, s/table.
- `challenge/DECISIONS.md` — one ADR per decision: context, options, choice, evidence, revisit-if.
- Short status message for the organizer: current score per tier, worst error class, next step.

Acceptance: someone else can read the three files and rebuild our reasoning without asking us.

## Risk register

| Risk | Trigger | Response |
|---|---|---|
| Overfitting the 24 dev tables | dev ≫ val | decide on val only, dev as yardstick |
| Model too slow on 96 tables | > 10 s/table | crop-only inference, smaller model, batching, cheap fallback |
| VRAM > 8 GB | OOM in container | CPU/ONNX path, fp16, smaller input resolution |
| PDF / text-around-table inputs | crash in container | M4 failure-mode tests, defensive loader |
| Borderless tables with no lines | structure garbage | whitespace/column-gap detection or a structure model |
| Team still incomplete | no second pair of hands | scope to candidates 1–2 + post-processing, document deeply |

## Definition of done

1. Container builds and scores on a clean machine with the four commands in M4.
2. Score clearly above 0.348 on dev **and** on val, with T2+T3 carrying most of the gain.
3. Number format, spans and empty cells handled explicitly (they are cheap score).
4. ERRORS / MEASUREMENTS / DECISIONS written and current.
5. Predictions exist for every input even in the worst case.
