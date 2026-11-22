# Measurements

One row per scored run. Never edit an old row; add a new one. Numbers come from
`python score_all.py --gt <dataset> --pred <dir>`; keep the raw line in `notes/` if it is long.

Rows marked *VLM track* were copied on 2026-11-22 from branch `feat/vlm-ocr-bench` (full table
there, incl. the small-model matrix); rows marked *tablesalt track* come from `docs/RESULTS.md`
and commit messages. s/table: VLM rows = Apple Silicon/MLX; tablesalt rows = per worker process.

| Date | What changed | Dataset | T0 | T1 | T2 | T3 | QA | Final | s/table | Who |
|---|---|---|---|---|---|---|---|---|---|---|
| | _baseline Tesseract (template floor)_ | dev (24) | 0.54 | 0.46 | 0.26 | 0.35 | 0.267 | 0.348 | | organizer |
| 2026-11-21 | baseline Tesseract floor in container (Tesseract 5.5.0; Mac host 5.5.3 gives 0.348) | dev (24) | 0.539 | 0.474 | 0.256 | 0.329 | 0.250 | 0.3410 | 0.20 | VLM track |
| 2026-11-20 | TATR structure + Tesseract per cell (nms 0.3, median denoise) | dev (24) | 0.831 | 0.877 | 0.738 | 0.623 | 0.275 | 0.6608 | 0.16 | VLM track |
| 2026-11-20 | same, **oracle text** (structure ceiling, not a submission) | dev (24) | 0.913 | 0.931 | 0.936 | 0.946 | — | 0.9365 TEDS_w | 0.08 | VLM track |
| 2026-11-20 | MinerU2.5 1.2B (bf16, raw VLM, MLX) | dev (24) | 0.959 | 0.992 | 0.956 | 0.966 | 0.958 | 0.9659 | 2.56 | VLM track |
| 2026-11-20 | MinerU2.5 1.2B (bf16, raw VLM, MLX) | val96 | 0.986 | 0.975 | 0.945 | 0.963 | 0.960 | 0.9621 | 2.33 | VLM track |
| 2026-11-20 | MinerU2.5 1.2B + postprocess P0..P5 | dev (24) | 0.979 | 1.000 | 0.977 | 1.0000 | 0.958 | 0.9862 | 2.56+pp | VLM track |
| 2026-11-20 | MinerU2.5 1.2B + postprocess P0..P5 | val96 | 0.995 | 0.983 | 0.975 | 0.9891 | 0.960 | 0.9808 | 2.33+pp | VLM track |
| 2026-11-21 | TATR -> clean Tesseract (`scripts/my_tatr_fusion.py`) | dev (24) | 0.899 | 0.867 | 0.764 | 0.682 | 0.508 | 0.7268 | 1.49 | VLM track |
| 2026-11-21 | same in CUDA benchmark image (Tesseract 5.3.4, amd64 emulation, offline) | dev (24) | 0.898 | 0.868 | 0.752 | 0.648 | 0.508 | 0.7124 | 3.84 | VLM track |
| 2026-11-21 | IBM Docling / TableFormer pur → + DataPrep | dev (24) | 0.770→0.915 | 0.975→0.967 | 0.896→0.920 | 0.705→0.941 | 0.758→0.850 | 0.8132→0.9239 | ~3.0 | VLM track |
| 2026-11-21 | Enes `tableparser` (CPU + PP-OCRv5 ONNX) + Layer-1 coherence | dev (24) | 0.999 | 1.000 | 0.998 | 0.979 | 0.900 | 0.9774 | 2.05 | VLM track |
| 2026-11-21 | Enes `tableparser` + Layer-1 coherence | val96 | 0.994 | 0.975 | 0.984 | 0.924 | 0.931 | 0.9551 | ~3.3 | VLM track |
| 2026-11-21 | **tablesalt** M1–M7 first full pipeline (`cd54a0c`) | dev (24) | 1.000 | 0.998 | 0.975 | 0.979 | 0.925 | 0.9748 | 3.27/worker | tablesalt track |
| 2026-11-21 | **tablesalt** final config (3 reads + digit vote) | dev (24) | 1.000 | 1.000 | 1.000 | 0.999 | 0.983 | 0.9971 | 6.41/worker (Windows) | tablesalt track |
| 2026-11-21 | **tablesalt** final config | val (200, seed 2) | 1.000 | 1.000 | 0.997 | 0.996 | 0.989 | 0.9963 | 7.38/worker (Windows) | tablesalt track |
| 2026-11-21 | **tablesalt** + self-check router (`3c1a053`) | fresh 200 (seed 987654) | – | – | 1.000 | – | – | 0.9949 → 0.9978 | – | tablesalt track |
| 2026-11-22 | `main` re-measured on Linux, Tesseract 5.5.0, local venv | dev (24) | 1.000 | 1.000 | 1.000 | 0.999 | 0.983 | **0.9971** | 1.04/worker | grokbot |
| 2026-11-22 | `main`, never-seen set (Liberation Sans) | fresh96 (seed 20261122) | 0.999 | 1.000 | 1.000 | 0.994 | 0.969 | **0.9931** | 1.19/worker | grokbot |
| 2026-11-22 | branch `grokbot-abgabe` (lint, PDF input, file mode) — identical outputs | dev / fresh96 | same | same | same | same | same | 0.9971 / 0.9931 | 1.2/worker | grokbot |
| 2026-11-22 | `grokbot-abgabe` **in the submission container**, `--network none`, 8 CPUs | dev (24) | 1.000 | 1.000 | 1.000 | 0.999 | 0.983 | **0.9971** | 4 s wall / 24 tables; image 520 MB | grokbot |
| 2026-11-22 | same container, the 24 dev images wrapped as PDFs (+ 2-page PDF, corrupt PDF, corrupt PNG) | dev (24) as PDF | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 | 0.9998 | 4 s wall | grokbot |

## Notes on method

- `data/dev` = official yardstick, 24 tables, 6 per tier. Never trained on.
- `data/val` = generated decision set (`--n 240 --seed 2`), used whenever dev would be noise.
- A change counts as an improvement only if it wins on **both** sets, or wins on val and is neutral
  on dev.
- Record the wall-clock per table; the 15 s/table limit is a hard gate, not a preference.
