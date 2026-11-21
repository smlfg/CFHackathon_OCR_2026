# Decisions (ADRs)

Short by design: context, options, choice, evidence, revisit-if. One file so a reviewer can read the
reasoning in one place — the organizer explicitly asked for it.

Template:

```
## ADR-NNN — <title>                    (YYYY-MM-DD, status: proposed|accepted|superseded)
Context:   what forced the decision
Options:   A / B / C, each with what it would cost
Choice:    X
Evidence:  numbers (dev/val, s/table), error classes fixed, or "measured, no gain"
Revisit:   the condition under which this becomes wrong
```

## Open decisions (from [PROBLEM.md](PROBLEM.md) §10)

| # | Question | Leaning | Status |
|---|---|---|---|
| 1 | Structure: classic CV vs. table-structure model vs. hybrid | measure candidate 2 before deciding · TATR reviewed: [TATR_RESEARCH.md](TATR_RESEARCH.md) | open |
| 2 | Text: Tesseract vs. PaddleOCR vs. small VLM | OCR before VLM (time budget) | open |
| 3 | Preprocessing: how much correction is worth it | on/off module, measure | open |
| 4 | Post-processing: per-table number format, span repair, empty cells | do it — cheapest score | open |
| 5 | Fine-tune on generated data? | only after a real baseline number | open |
| 6 | Output JSON vs HTML | JSON | open |
| 7 | Fallback ladder when a model is slow/unreadable | never zero cells | open |

## Proposed

## ADR-001 — Table Transformer (TATR) as structure specialist      (2026-11-20, status: accepted)
Context:   Open decision #1. TATR is 28.8M params, deterministic, line-agnostic, emits spans and empty
           cells by construction, and is trainable on our generator — but it has no text model and was
           trained on clean 1000 px PDF renders.
Options:   A replace the VLM (no text model) / B TATR structure + per-cell OCR /
           C TATR as validator on the VLM's HTML
Choice:    **C. Not a replacement.** Structure ceiling (TATR grid + ground-truth text) is weighted
           TEDS 0.9365, 23/24 shapes exactly right — below what the VLM already delivers end to end
           (0.9612 TEDS-weighted *with* text). With Tesseract text: 0.7289 / FINAL 0.6608.
Evidence:  [TATR_RESEARCH.md](TATR_RESEARCH.md) §6, `scripts/tatr_experiment.py`.
           T3 (fax) 0.083 → 0.946 by 3×3 median denoise + NMS 0.3 on row/column duplicates — the defect
           is duplicate detections on noise, not skew, resolution or capacity. Naive deskew costs 0.21
           TEDS on T2. Structure is not our bottleneck; text is (0.946 → 0.623 on T3).
Revisit:   if T2/T3 structural errors start dominating [ERRORS.md](ERRORS.md) — then B plus a fine-tune
           on `generate_dataset.py` output becomes worth the weekend. Or if a VLM rebuilds our grid, in
           which case C (validator) drops in without a text model.

## Accepted

## ADR-002 — Tesseract floor as a pipeline runtime, not a side script   (2026-11-21, status: accepted)
Context:   The floor the organizer ships (`baselines/baseline_tesseract.py`, ≈0.34 on dev) was not
           on the 22-model board, so "what does a classic CV pipeline get?" had no comparable row.
Options:   A ingest `predictions_tess/` into the metrics store as an external run / B run Tesseract
           locally as a third runtime (`tesseract`), same convert → score → store path as the VLMs
Choice:    **B.** One registry row (`tesseract-baseline`), no weights, CPU-only, ~0.2 s/table.
           The algorithm is unchanged and deliberately non-optimized: TSV words, row/column
           clustering, plain grid, no spans, no pre/post-processing.
Evidence:  `pipeline run --model tesseract-baseline --dataset dev` → **0.3424** (T0 0.539, T1 0.490,
           T2 0.268, T3 0.309, QA 0.267, CellF1 0.159), [MEASUREMENTS.md](MEASUREMENTS.md).
           Same run inside the built image (Debian Tesseract) → 0.3441, so the row is reproducible.
           The standalone script scores 0.3435 because it keeps the two wordless tables as
           `<table></table>`; the pipeline converter scores them as missing, like for every model.
           Normalizing to RGB before pytesseract removes a double JPEG re-encode in the script
           (0.3417 → 0.3435); without it the number depends on the source file codec.
Revisit:   only if a stronger classic baseline (PaddleOCR table pipeline) replaces this floor.

## ADR-003 — CUDA benchmark container, standalone, outside the MLX board   (2026-11-21, status: accepted)
Context:   Question: how fast is TATR → Tesseract on CUDA, and how good is it on held-out tables?
           The TATR → Tesseract path already exists (`scripts/my_tatr_fusion.py`, dev FINAL 0.7268)
           and is the only structure-specialist route we measured. It had never been run in a
           container, and never on a GPU. The MLX board cannot answer this: it serves VLMs over HTTP
           and its `s/table` is an Apple Silicon number, not a CUDA number.
Options:   A add a `runtime: cuda` row to the MLX pipeline registry / B standalone benchmark
           container that imports the existing fusion script unchanged and times per stage /
           C rewrite TATR → Tesseract as a fresh CUDA-native service
Choice:    **B.** One directory (`benchmarks/cuda/`), no new runtime in `pipeline/`, no new
           abstraction over `pipeline/`. The fusion script is *imported*, not copied — the harness
           cannot drift from the algorithm it measures. Timing is per stage, because the two halves
           have different cost profiles (DETR forward = GPU-bound, Tesseract per cell = CPU-bound).
Evidence:  CPU reference on the Mac (2026-11-21): FINAL **0.7268**, CellF1 0.6235 — identical to
           the `scripts/my_tatr_fusion.py` row in [MEASUREMENTS.md](MEASUREMENTS.md), which is the
           correctness check for the harness. Stage split: TATR 6.8 %, **Tesseract 89.4 %**,
           s/table 1.43 (p50 1.40, p90 2.32), shape exactly right 23/24.
           Consequence: CUDA can touch ~7 % of the wall clock, so the GPU is not the lever here.
           The speed ceiling for this pipeline is set by the OCR stage, not by TATR.
Revisit:   if speed ever binds, the lever is the OCR stage (batched Tesseract, or a GPU OCR model),
           not TATR. Also revisit if TATR is fine-tuned (ADR-001 option B) — then the structure
           forward pass gets more expensive and the GPU share rises.
