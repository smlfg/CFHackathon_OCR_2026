# Measurements

One row per scored run. Never edit an old row; add a new one. Numbers come from
`python score_all.py --gt <dataset> --pred <dir>`; keep the raw line in `notes/` if it is long.

| Date | What changed | Dataset | T0 | T1 | T2 | T3 | QA | Final | s/table | Who |
|---|---|---|---|---|---|---|---|---|---|---|
| | _baseline Tesseract (template floor)_ | dev (24) | 0.54 | 0.46 | 0.26 | 0.35 | 0.267 | 0.348 | | organizer |
| 2026-11-21 | baseline Tesseract floor in container (tablesalt-variant:dev, Tesseract 5.5.0; Mac host 5.5.3 gives 0.348) | dev (24) | 0.539 | 0.474 | 0.256 | 0.329 | 0.250 | 0.3410 | 0.20 | us |
| 2026-11-20 | TATR structure + Tesseract per cell (nms 0.3, median denoise) | dev (24) | 0.831 | 0.877 | 0.738 | 0.623 | 0.275 | 0.6608 | 0.16 | us |
| 2026-11-20 | same, **oracle text** (structure ceiling, not a submission) | dev (24) | 0.913 | 0.931 | 0.936 | 0.946 | — | 0.9365 TEDS_w | 0.08 | us |
| 2026-11-20 | MinerU2.5 1.2B (bf16, raw VLM baseline, MLX) | dev (24) | 0.959 | 0.992 | 0.956 | 0.966 | 0.958 | 0.9659 | 2.56 | us |
| 2026-11-20 | MinerU2.5 1.2B (bf16, raw VLM baseline, MLX) | val96 (96, seed=2) | 0.986 | 0.975 | 0.945 | 0.963 | 0.960 | 0.9621 | 2.33 | us |
| 2026-11-20 | MinerU2.5 1.2B + T3 `median3` preproc (T3 only; dropped: hurt CellF1/QA) | val96 T3 (24) / dev T3 (6) | — | — | — | 0.956 / 0.956 | 0.850 / 1.000 | — | 2.08 / 2.19 | us |
| 2026-11-20 | MinerU2.5 1.2B + T3 `despeckle1` preproc (T3 only; raw+pp wins end-to-end) | val96 T3 (24) / dev T3 (6) | — | — | — | 0.970 / 0.966 | 1.000 / 1.000 | — | 2.21 / 2.29 | us |
| 2026-11-20 | MinerU2.5 1.2B + postprocess P0..P5 (`pipeline/postprocess.py`) | dev (24) | 0.979 | 1.000 | 0.977 | 1.0000 | 0.958 | 0.9862 | 2.56+pp | us |
| 2026-11-20 | MinerU2.5 1.2B + postprocess P0..P5 (`pipeline/postprocess.py`) | val96 (96, seed=2) | 0.995 | 0.983 | 0.975 | 0.9891 | 0.960 | 0.9808 | 2.33+pp | us |
| | | | | | | | | | | |

## Notes on method

- `data/dev` = official yardstick, 24 tables, 6 per tier. Never trained on.
- `data/val` = generated decision set (`--n 240 --seed 2`), used whenever dev would be noise.
- A change counts as an improvement only if it wins on **both** sets, or wins on val and is neutral
  on dev.
- Record the wall-clock per table; the 15 s/table limit is a hard gate, not a preference.
