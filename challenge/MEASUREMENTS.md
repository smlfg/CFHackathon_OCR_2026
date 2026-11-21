# Measurements

One row per scored run. Never edit an old row; add a new one. Numbers come from
`python score_all.py --gt <dataset> --pred <dir>`; keep the raw line in `notes/` if it is long.

| Date | What changed | Dataset | T0 | T1 | T2 | T3 | QA | Final | s/table | Who |
|---|---|---|---|---|---|---|---|---|---|---|
| | _baseline Tesseract (template floor)_ | dev (24) | 0.54 | 0.46 | 0.26 | 0.35 | 0.267 | 0.348 | | organizer |
| 2026-11-21 | baseline Tesseract floor in container (tablesalt-variant:dev, Tesseract 5.5.0; Mac host 5.5.3 gives 0.348) | dev (24) | 0.539 | 0.474 | 0.256 | 0.329 | 0.250 | 0.3410 | 0.20 | us |
| 2026-11-21 | Tesseract floor as pipeline runtime `tesseract-baseline` (non-optimized; RGB/PNG before pytesseract, no double JPEG re-encode) | dev (24) | 0.539 | 0.490 | 0.268 | 0.309 | 0.267 | 0.3424 | 0.20 | us |
| 2026-11-20 | TATR structure + Tesseract per cell (nms 0.3, median denoise) | dev (24) | 0.831 | 0.877 | 0.738 | 0.623 | 0.275 | 0.6608 | 0.16 | us |
| 2026-11-20 | same, **oracle text** (structure ceiling, not a submission) | dev (24) | 0.913 | 0.931 | 0.936 | 0.946 | — | 0.9365 TEDS_w | 0.08 | us |
| 2026-11-20 | MinerU2.5 1.2B (bf16, raw VLM baseline, MLX) | dev (24) | 0.959 | 0.992 | 0.956 | 0.966 | 0.958 | 0.9659 | 2.56 | us |
| 2026-11-20 | MinerU2.5 1.2B (bf16, raw VLM baseline, MLX) | val96 (96, seed=2) | 0.986 | 0.975 | 0.945 | 0.963 | 0.960 | 0.9621 | 2.33 | us |
| 2026-11-20 | MinerU2.5 1.2B + T3 `median3` preproc (T3 only; dropped: hurt CellF1/QA) | val96 T3 (24) / dev T3 (6) | — | — | — | 0.956 / 0.956 | 0.850 / 1.000 | — | 2.08 / 2.19 | us |
| 2026-11-20 | MinerU2.5 1.2B + T3 `despeckle1` preproc (T3 only; raw+pp wins end-to-end) | val96 T3 (24) / dev T3 (6) | — | — | — | 0.970 / 0.966 | 1.000 / 1.000 | — | 2.21 / 2.29 | us |
| 2026-11-20 | MinerU2.5 1.2B + postprocess P0..P5 (`pipeline/postprocess.py`) | dev (24) | 0.979 | 1.000 | 0.977 | 1.0000 | 0.958 | 0.9862 | 2.56+pp | us |
| 2026-11-20 | MinerU2.5 1.2B + postprocess P0..P5 (`pipeline/postprocess.py`) | val96 (96, seed=2) | 0.995 | 0.983 | 0.975 | 0.9891 | 0.960 | 0.9808 | 2.33+pp | us |
| 2026-11-21 | TATR -> clean Tesseract (`scripts/my_tatr_fusion.py`: border trim, white pad, upscale, `deu+eng`, speckle-gated median, NMS 0.3; CellF1 0.6235) | dev (24) | 0.899 | 0.867 | 0.764 | 0.682 | 0.508 | 0.7268 | 1.49 | us |
| 2026-11-21 | CUDA benchmark harness (`benchmarks/cuda/bench_tatr_cuda.py`, CPU/Mac) — reproduces the row above exactly (FINAL 0.7268, CellF1 0.6235), so the harness is faithful. Stage split: TATR 6.8 %, **Tesseract 89.4 %** | dev (24) | 0.899 | 0.867 | 0.764 | 0.682 | 0.508 | 0.7268 | 1.43 | us |
| 2026-11-21 | CUDA benchmark image `tablesalt-tatr-cuda:amd64` (Ubuntu 24.04, Tesseract **5.3.4**, CUDA 12.6, amd64 under emulation) — full pipeline in-container, offline (`--network none`) | dev (24) | 0.898 | 0.868 | 0.752 | 0.648 | 0.508 | 0.7124 | 3.84 | us |
| 2026-11-21 | Small-model eval: `img2table` pur → `+ Enes DataPrep` | dev (24) | 0.807→0.841 | 0.759→0.809 | 0.638→0.806 | 0.501→0.755 | 0.375→0.433 | 0.5868→0.7362 | ~0.7 | us |
| 2026-11-21 | Small-model eval: `RapidTable SLANet_plus` (7.4 MB ONNX + PP-OCRv6) pur → `+ Enes DataPrep` | dev (24) | 0.853→0.862 | 0.684→0.732 | 0.677→0.676 | 0.252→0.659 | 0.383→0.517 | 0.5046→0.6718 | ~0.4 | us |
| 2026-11-21 | Small-model eval: `TATR` + `PP-OCRv5 ONNX` (7.9 MB) / `PARSeq` (23M ViT) | dev (24) | 0.854 / 0.827 | 0.895 / 0.810 | 0.831 / 0.756 | 0.757 / 0.647 | 0.333 / 0.108 | 0.7441 / 0.6372 | 0.34 / 0.51 | us |
| 2026-11-21 | Small-model eval: `IBM Docling / TableFormer` pur → `+ Enes DataPrep` | dev (24) | 0.770→0.915 | 0.975→0.967 | 0.896→0.920 | 0.705→0.941 | 0.758→0.850 | 0.8132→0.9239 | ~3.0 | us |
| 2026-11-21 | Enes `tableparser` (`enescaglarr/group14-test-enes`, CPU+PP-OCRv5 ONNX) pur | dev (24) | 0.999 | 0.995 | 0.981 | 0.978 | 0.892 | 0.9704 | 2.05 | us |
| 2026-11-21 | Enes `tableparser` pur (unseen generalization set) | val96 (96, seed=2) | 0.990 | 0.968 | 0.960 | 0.905 | 0.883 | 0.9336 | ~3.3 | us |
| 2026-11-21 | Enes `tableparser` + Layer-1 Coherence (`scripts/coherence_postprocess.py`, 0 MB, <1 ms; 4/24 improved, 0 regressed; CellF1 0.9807) | dev (24) | 0.999 | 1.000 | 0.998 | 0.979 | 0.900 | 0.9774 | 2.05 | us |
| 2026-11-21 | Enes `tableparser` + Layer-1 Coherence (`scripts/coherence_postprocess.py`; **17/96 improved, 0 regressed**; CellF1 0.8467→0.9387) | val96 (96, seed=2) | 0.994 | 0.975 | 0.984 | 0.924 | 0.931 | 0.9551 | ~3.3 | us |
| | | | | | | | | | | |

## Notes on method

- `data/dev` = official yardstick, 24 tables, 6 per tier. Never trained on.
- `data/val` = generated decision set (`--n 240 --seed 2`), used whenever dev would be noise.
- A change counts as an improvement only if it wins on **both** sets, or wins on val and is neutral
  on dev.
- Record the wall-clock per table; the 15 s/table limit is a hard gate, not a preference.
