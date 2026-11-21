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
| 1 | Structure: classic CV vs. table-structure model vs. hybrid | measure candidate 2 before deciding | open |
| 2 | Text: Tesseract vs. PaddleOCR vs. small VLM | OCR before VLM (time budget) | open |
| 3 | Preprocessing: how much correction is worth it | on/off module, measure | open |
| 4 | Post-processing: per-table number format, span repair, empty cells | do it — cheapest score | open |
| 5 | Fine-tune on generated data? | only after a real baseline number | open |
| 6 | Output JSON vs HTML | JSON | open |
| 7 | Fallback ladder when a model is slow/unreadable | never zero cells | open |

## Accepted

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
