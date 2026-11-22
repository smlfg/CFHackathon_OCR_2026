# Decisions (ADRs)

What we decided, what we threw away, and the number behind each choice. The organizer asked for
this file explicitly. Every ADR uses the same five fields:

```
## ADR-NNN — <title>                    (YYYY-MM-DD, status: proposed|accepted|superseded|not built)
Context:   what forced the decision
Options:   A / B / C, each with what it would cost
Choice:    X
Evidence:  numbers (dev/val, s/table), error classes fixed, or "measured, no gain"
Revisit:   the condition under which this becomes wrong
```

**About the provenance.** During the hackathon (20–21 Nov) the team worked on three tracks at once
and wrote most decisions down only as commit messages and measurement rows. On 2026-11-22 we
reconstructed them from git history (all branches), [MEASUREMENTS.md](MEASUREMENTS.md),
[docs/RESULTS.md](../docs/RESULTS.md), [docs/LEARNLOG.md](../docs/LEARNLOG.md), the kick-off
[TRANSCRIPT.md](TRANSCRIPT.md) and our voice notes. Each ADR names the commit it was reconstructed
from. ADR-001…003 were written during the hackathon on branch `feat/vlm-ocr-bench` and are copied
here unchanged except for the status line. We only quote numbers that are in the repo or that we
re-measured on 2026-11-22. Where no number exists, the ADR says so.

**Datasets used below**

| Name | What | Notes |
|---|---|---|
| dev | `data/dev`, the 24 official tables, 6 per tier | official yardstick; we also used it for error analysis, so it is not a clean hold-out |
| val (200) | `generate_dataset.py --n 200 --seed 2`, Arial, Windows | the tablesalt decision set ([docs/RESULTS.md](../docs/RESULTS.md)) |
| val96 | `generate_dataset.py --n 96 --seed 2` | the VLM-track decision set ([MEASUREMENTS.md](MEASUREMENTS.md)). Not the same tables as val (200), so do not compare these two directly |
| fresh 200 | seed 987654 | used once, for the self-check router (commit `3c1a053`) |
| fresh96 | `--n 96 --seed 20261122`, Liberation Sans (Arial-metric), Linux | generated on 2026-11-22 after all tuning was finished, so it is never-seen data for every number below |

## Status of the seven questions from [PROBLEM.md](PROBLEM.md) §10

| # | Question | Answer | ADR |
|---|---|---|---|
| 1 | Structure: classic CV vs. table-structure model vs. hybrid | classic CV (text blobs + right-edge columns); we measured TATR and left it out | 005, 006, 001 |
| 2 | Text: Tesseract vs. PaddleOCR vs. small VLM | Tesseract per cell, 3 reads + vote; VLMs measured (22 models) and not shipped | 004, 005, 008 |
| 3 | Preprocessing: how much correction is worth it | illumination + Otsu + deskew + text-edge homography; not ablated on its own | 007 |
| 4 | Post-processing | one number format per table, lexicon snap, span rules | 009 |
| 5 | Fine-tune on generated data? | no | 010 |
| 6 | Output JSON vs HTML | JSON | 011 |
| 7 | Fallback ladder | pipeline → second opinion → Tesseract floor → one empty cell | 012, 013, 014 |

## Timeline in one paragraph

**Day 1 (20 Nov).** We read the challenge and wrote down the problem ([PROBLEM.md](PROBLEM.md)).
We built a VLM benchmark board and ran 22 open models on dev (ADR-004). We also measured TATR
(ADR-001).

**Day 2 (21 Nov).** Three tracks ran in parallel:

- (a) The VLM/TATR track built postprocessing, a CUDA harness and a 3.56 GB multi-variant container.
  This work is on branch `feat/vlm-ocr-bench` and was never merged.
- (b) A classic-CV + Tesseract pipeline (`tablesalt/`) was built milestone by milestone, following
  [agent.md](../agent.md).
- (c) A self-check router was added on top of (b).

(b) and (c) were merged into `main` through PR #1 and are the submission (ADR-005).

**Day 3 (22 Nov, before hand-in).** We cleaned the code, added PDF input handling and fixed
file permissions. We wrote these ADRs and did not change any scoring behaviour.

---

## ADR-001 — Table Transformer (TATR) as structure specialist      (2026-11-20, status: superseded by ADR-005)
*Written during the hackathon on branch `feat/vlm-ocr-bench` (state at `0ea8b01`). Copied with paths adjusted; the last Evidence sentence and the "Superseded because" note were added on 22 Nov.*
- **Context:** Open decision #1. TATR is 28.8M params, deterministic, line-agnostic, emits spans and empty cells by construction, and is trainable on our generator — but it has no text model and was trained on clean 1000 px PDF renders.
- **Options:** A replace the VLM (no text model) / B TATR structure + per-cell OCR / C TATR as validator on the VLM's HTML
- **Choice:** **C. Not a replacement.** Structure ceiling (TATR grid + ground-truth text) is weighted TEDS 0.9365, 23/24 shapes exactly right — below what the VLM already delivers end to end (0.9612 TEDS-weighted *with* text). With Tesseract text: 0.7289 / FINAL 0.6608.
- **Evidence:** `challenge/TATR_RESEARCH.md` §6, `scripts/tatr_experiment.py` (both on the branch). T3 (fax) 0.083 → 0.946 by 3×3 median denoise + NMS 0.3 on row/column duplicates — the defect is duplicate detections on noise, not skew, resolution or capacity. Naive deskew costs 0.21 TEDS on T2. Structure is not our bottleneck; text is (0.946 → 0.623 on T3). Later on 21 Nov, TATR → clean Tesseract (`scripts/my_tatr_fusion.py`) reached dev FINAL **0.7268**, 1.49 s/table (commit `3cbde1b`).
- **Revisit:** if T2/T3 structural errors start dominating ERRORS.md — then B plus a fine-tune on `generate_dataset.py` output becomes worth the weekend.

**Superseded because** the classic-CV structure in `tablesalt/` reaches TEDS-S 1.000 on dev
(ADR-006). TATR's structure ceiling with perfect text is 0.9365, so it has nothing left to add here.

## ADR-002 — Tesseract floor as a pipeline runtime, not a side script   (2026-11-21, status: accepted, branch `feat/vlm-ocr-bench`)
*Written during the hackathon (commit `76a98f6`), copied (shortened).*
- **Context:** The floor the organizer ships (`baselines/baseline_tesseract.py`, ≈0.34 on dev) was not on the 22-model board, so "what does a classic CV pipeline get?" had no comparable row.
- **Options:** A ingest `predictions_tess/` into the metrics store as an external run / B run Tesseract locally as a third runtime (`tesseract`), same convert → score → store path as the VLMs
- **Choice:** **B.** One registry row (`tesseract-baseline`), no weights, CPU-only, ~0.2 s/table.
- **Evidence:** `pipeline run --model tesseract-baseline --dataset dev` → **0.3424** (T0 0.539, T1 0.490, T2 0.268, T3 0.309, QA 0.267, CellF1 0.159). Same run inside the built image → 0.3441.
- **Revisit:** only if a stronger classic baseline replaces this floor (it did: ADR-005).

## ADR-003 — CUDA benchmark container, standalone, outside the MLX board   (2026-11-21, status: accepted, branch `feat/vlm-ocr-bench`)
*Written during the hackathon (commit `7305ffa`/`ad228ca`), copied (shortened).*
- **Context:** How fast is TATR → Tesseract on CUDA? The MLX board cannot answer this: its `s/table` is an Apple Silicon number.
- **Options:** A `runtime: cuda` row in the MLX registry / B standalone benchmark container that imports the fusion script unchanged and times per stage / C CUDA-native rewrite
- **Choice:** **B** (`benchmarks/cuda/` on the branch).
- **Evidence:** CPU reference reproduces FINAL **0.7268** exactly. Stage split: TATR 6.8 %, **Tesseract 89.4 %**, 1.43 s/table. In-container (Tesseract 5.3.4, amd64 emulation, `--network none`): 0.7124, 3.84 s/table. Consequence: the GPU can touch ~7 % of the wall clock.
- **Revisit:** if speed ever binds, the lever is the OCR stage, not TATR.

## ADR-004 — Benchmark 22 open VLM/OCR models before choosing     (2026-11-20, status: accepted; reconstructed 2026-11-22 from commits `7e0d32b`, `c4422f2`)
- **Context:** Decision #2 (text model) had no data. The kick-off framed the problem as "cheap, local, structure-aware", but nobody knew how far small open models get on these tables.
- **Options:** A pick one model by reputation / B run every open model ≤ 15B (OCRBench v2 registry + OCR specialists) on dev with the official scorer and keep a leaderboard
- **Choice:** **B.** `pipeline/` + `registry/` + `scripts/mlx_server.py` (MLX on a Mac, resumable).
- **Evidence:** Board on branch `feat/vlm-ocr-bench` (`benchmarks/dev/LEADERBOARD.md`), 23 rows: best raw model MinerU2.5 1.2B dev **0.9659** (2.1–2.56 s/table, Mac MLX), Chandra OCR 2 0.9640 (14.9 s), PaddleOCR-VL 0.9570 (1.5 s). The floor is 0.3424. With the deterministic postprocess P0–P5 MinerU reached dev **0.9862** and val96 **0.9808** (commit `76a98f6`).
- **Revisit:** inputs that the generator does not produce (real scans, text around the table) — that is where a VLM's generality would beat the specialised classic pipeline (see ADR-015).

## ADR-005 — Ship the classic-CV + Tesseract pipeline (`tablesalt/`), not the VLM or TATR   (2026-11-21, status: accepted; reconstructed 2026-11-22 from PR #1 / commits `cd54a0c`, `3c1a053`, `973cc2d`)
- **Context:** By the evening of 21 Nov there were three end-to-end candidates and one container slot.
- **Options:** A MinerU2.5 VLM + P0–P5 (needs GPU weights in the image) / B TATR → Tesseract fusion / C `tablesalt/`: OpenCV structure + Tesseract per cell, CPU only / D teammate pipeline "Enes tableparser" + Layer-1 coherence (PP-OCRv5 ONNX, branch)
- **Choice:** **C**, merged into `main` with PR #1.
- **Evidence:** the four candidates side by side (numbers from MEASUREMENTS.md, docs/RESULTS.md and our container run on 22 Nov):

| candidate | dev FINAL | other set | s/table | image |
|---|---|---|---|---|
| A MinerU2.5 + P0–P5 | 0.9862 | val96 0.9808 | 2.56 + pp (Mac MLX; CUDA not measured) | multi-variant CUDA image 3.56 GB (`0ea8b01`) |
| B TATR → Tesseract | 0.7268 | – | 1.49 | 5.7 GB benchmark image |
| D Enes + coherence | 0.9774 | val96 0.9551 | 2.05 | not built |
| **C tablesalt** | **0.9971** | val (200) 0.9963; fresh96 **0.9931** | ~1.2 per worker; 4 s wall for all 24 tables (8 CPUs, container) | **520 MB** |

  C wins on score on every set we have and is ~7× smaller. It needs no GPU and no weights, so
  nothing has to be downloaded at build time and nothing can drift.
  **Caveat (honest):** C is built on generator knowledge: font size, alignment rules, the word
  lists the labels are snapped to ([docs/NOTES.md](../docs/NOTES.md) §5). The hidden set comes from
  the same generator with a private seed, so this is legitimate here. It is also exactly why C would
  degrade on real-world documents.
- **Revisit:** if the hidden set contains inputs the generator cannot make (real scans, extra text around the table). Then route those to A (ADR-015).

## ADR-006 — Structure from text blobs and right-aligned edges, not from ruling lines   (2026-11-21, status: accepted; reconstructed from `cd54a0c`, LEARNLOG M3/M4)
- **Context:** Three styles (grid, hlines, borderless). Line detection only works for one of them.
- **Options:** A detect ruling lines (fails on borderless) / B whitespace-corridor search / C dilate text horizontally to ~0.85 × font height → one blob per cell; rows = y-clusters; columns = shared right edges of right-aligned numbers (grid: vertical rules)
- **Choice:** **C**, with B only as last resort.
- **Evidence:** First full pipeline (M1–M7) dev TEDS-S **0.999**, final config dev TEDS-S **1.000**, val (200) TEDS-S 0.998–0.999 ([docs/RESULTS.md](../docs/RESULTS.md)). Lesson recorded: smeared words make long horizontal runs that look like rules, so real rules must span almost the whole table. Tried `blob_join_x` 0.85 → 0.65: val 0.9963 → 0.9934, **reverted**.
- **Revisit:** tables whose numbers are not right-aligned (real documents).

## ADR-007 — Preprocessing: illumination division, Otsu, projection-profile deskew, text-edge homography   (2026-11-21, status: accepted; reconstructed from `cd54a0c`, LEARNLOG M2)
- **Context:** T2/T3 have shadow, skew, perspective, noise; T2+T3 carry 70 % of the weight.
- **Options:** A none / B generic deskew only / C background division (morphological closing) + Otsu + deskew by projection-profile sharpness + perspective from four straight text edges (label left edge, number right edge, first/last row) → homography
- **Choice:** **C.**
- **Evidence:** No isolated ablation number for C exists in the repo; it is only measured as part of the M1–M7 pipeline (dev 0.348 → 0.9748). Related measurements: on the TATR track a naive deskew cost 0.21 TEDS on T2 (ADR-001); on the VLM track T3 `median3` preprocessing hurt CellF1/QA and was dropped, while `despeckle1` won only on T3 (MEASUREMENTS.md). Lesson recorded: the output canvas must be computed from the transformed corners, otherwise the homography crops the table.
- **Revisit:** when we ablate each step on fresh data; this is the cheapest open experiment.

## ADR-008 — OCR: Tesseract per blob, batched, three crop variants + digit vote   (2026-11-21, status: accepted; reconstructed from `cd54a0c`, docs/RESULTS.md)
- **Context:** Text was the bottleneck once structure was right (also the TATR finding, ADR-001).
- **Options:** A one Tesseract call per page / B one read per cell / C several reads per cell with different crops, choose by role and vote on digits / D a VLM or PP-OCR per cell
- **Choice:** **C**: one Tesseract process per table (image list, not 100 process starts); reads `margin:median:psm` = `1:3:7`, `3:0:7`, `2:0:7`; `--psm 13` rescue for empty reads.
- **Evidence:** val (200), each row measured ([docs/RESULTS.md](../docs/RESULTS.md)): crop cap/margin/median tuning 0.9799 → 0.9906; 2 reads + role pick 0.9936; psm 7×2 + psm 13 rescue 0.9952; 3 reads + digit vote **0.9963** (s/table 5.23 → 7.38 per worker). Rejected with numbers: variant `1:3:13,3:0:13` 0.9785; `1:3:13,3:0:7` 0.9931; numeric whitelist pass: no gain (0.9936 either way) and slower, so **off**.
- **Revisit:** if the 15 s/table budget binds on a slower machine — drop to psm 7 ×2 + psm 13 rescue (val 0.9952, 5.23 s/table, i.e. −0.0011).

## ADR-009 — Postprocess: one number format per table, lexicon snap, deterministic span rules   (2026-11-21, status: accepted; reconstructed from `cd54a0c`, LEARNLOG M5/M7)
- **Context:** Number format and labels drive QA (15 % of the score). The generator uses one format per table and fixed word lists.
- **Options:** A leave OCR text as read / B per-cell regex cleanup / C table-level majority vote of the number format and re-print every number from its digits; fuzzy-snap labels and headers (`rapidfuzz`) to the generator's pools; `—` / `see note 3` always cs=2, `–` never, `n/a` by position; missing minus from geometry
- **Choice:** **C.**
- **Evidence:** Only measured together with the rest of M1–M7 (dev QA 0.267 → 0.925, final config 0.983). The VLM track made the same point in isolation: P0–P5 took MinerU dev 0.9659 → 0.9862, and the Unicode-dash rule alone was worth +0.0124 on dev (branch ERRORS.md).
- **Revisit:** any input not produced by the generator. The lexicon snap is the most generator-specific part of the system (see ADR-005 caveat).

## ADR-010 — No fine-tuning   (2026-11-21, status: accepted; reconstructed 2026-11-22)
- **Context:** The generator can produce unlimited exact training data (decision #5).
- **Options:** A fine-tune TATR / a small VLM on generated data / B stay zero-shot + deterministic code
- **Choice:** **B.**
- **Evidence:** The shipped pipeline has no learned weights and reaches dev 0.9971 / fresh96 0.9931, so there is ≤ 0.007 left to win on generator data. Fine-tuning also costs build time and a pinned weight download. The voice note from 20 Nov also warns that synthetic data from the same generator is not independent evidence (winner's curse on 24 dev tables).
- **Revisit:** if real-world inputs become part of the target.

## ADR-011 — Output JSON, not HTML   (2026-11-21, status: accepted)
- **Context:** Decision #6. The scorer accepts both, and JSON wins if both are present.
- **Choice:** JSON cell list `{text, r, c, rs, cs}`, empty cells written explicitly. The validator (`tablesalt/validate.py`) checks that the grid is covered exactly once before writing.
- **Evidence:** Same score by construction (the scorer converts JSON → HTML with `cells_json_to_html`).
- **Revisit:** never for this challenge.

## ADR-012 — Fallback ladder: never zero cells, never a crashed run   (2026-11-21, extended 2026-11-22, status: accepted)
- **Context:** Decision #7. An unparseable or missing file scores 0, and one crash must not end the run.
- **Options:** A fail fast / B per-table `try/except` → Tesseract floor → `[]` (21 Nov) / C same, but the last resort is one empty cell, not `[]` (22 Nov)
- **Choice:** **C**: pipeline → self-check second opinion (ADR-013) → organizer's Tesseract floor → `[{"text":"","r":0,"c":0,"rs":1,"cs":1}]`. Each file is written atomically (temp file + rename) as soon as it is done.
- **Evidence:** 0 fallbacks on dev, val (200) and fresh96. In a container test with a corrupt PNG and a corrupt PDF, both got the one-cell file and the run continued (2026-11-22). **Bug found 2026-11-22:** atomic writes used `mkstemp`, which creates `0600` files, and the container runs as root. A host user scoring the bind mount got "Permission denied", so every table would have scored 0 in our own docker test. Fixed with `chmod 0644` and a test.
- **Revisit:** –

## ADR-013 — Self-check + second-opinion router (CPU only)   (2026-11-21, status: accepted; reconstructed from `3c1a053`)
- **Context:** Some layout errors are visible without ground truth: header sequence broken, a label not in any pool, mixed number formats, units, wrong grid.
- **Options:** A always run several layout settings and pick (cost ×4 on every table) / B check the result against generator rules and re-read only flagged tables with other settings; keep the read with fewer problems; skip near a 12 s deadline
- **Choice:** **B** (`tablesalt/selfcheck.py`, `SECOND_OPINIONS` in `tablesalt/run.py`). `TABLESALT_SECOND_OPINION=0` turns it off.
- **Evidence:** fresh 200 (seed 987654): final 0.9949 → **0.9978**, T2 0.996 → 1.000. 0 of 224 correct tables flagged (no false alarms). Dev unchanged (0 flagged). fresh96: 0 flagged.
- **Revisit:** this is the seed of the routing idea in ADR-015.

## ADR-014 — Accept PDFs and other raster formats; render page 1   (2026-11-22, status: accepted)
- **Context:** The contract says `.png`/`.jpg`, but at the kick-off the organizer said "PDF files or PNG files" and mentioned "other surprises" ([PROBLEM.md](PROBLEM.md) §8–9). Before this change, a PDF was silently skipped, so it got no prediction file at all.
- **Options:** A ignore (score 0 for that table) / B poppler `pdftoppm` (system package) / C `pypdfium2` wheel (no system libs), first page at 200 dpi
- **Choice:** **C.** Multi-page PDFs: only page 1 is read (the contract is one table per file). `.tif/.bmp/.webp` are decoded by OpenCV.
- **Evidence:** The 24 dev images, each wrapped as a PDF: container FINAL 0.9998 (PNG/JPG originals: 0.9971). The re-render removes some JPEG noise; we do not read more into it. Added image size: +10 MB (510 → 520 MB).
- **Revisit:** if PDFs contain several tables or text around the table — then region detection is needed.

## ADR-015 — OCR routing decision table ("mixture of experts" by gate)   (2026-11-22, status: proposed, not built)
- **Context:** Samuel planned a mixture-of-experts / decision-tree router: send each table to the expert that handles it best. On 21 Nov there was no time left to build it. The data above is enough to write the routing table down; we deliberately did **not** add half-built code the night before the hand-in.
- **Options:** A learned router (needs labelled routing data we don't have) / B rule table gated by signals we already compute without ground truth / C no routing
- **Choice:** **B, proposed.** The gate is the self-check (ADR-013) plus the pipeline's own failure.

| Signal (available at inference, no GT) | Expert | Evidence |
|---|---|---|
| self-check clean | tablesalt read as is | 0/224 false flags; dev 0.9971, fresh96 0.9931 |
| self-check flagged | tablesalt second opinions (other layout settings) | fresh 200: 0.9949 → 0.9978 (**built**) |
| still flagged, or pipeline raised (implausible column count, no blobs) | **VLM expert**: MinerU2.5 1.2B + P0–P5 | dev 0.9862 / val96 0.9808 as a general reader; not containerised for CUDA in `main` |
| VLM unavailable or over the time budget | Tesseract floor | dev 0.3424 |
| nothing readable | one empty cell | ADR-012 |

- **Evidence:** Today the third row falls straight through to the Tesseract floor (0.34). On generator data this never happens: 0 fallbacks on dev, val (200) and fresh96. The router therefore buys nothing measurable on the hidden set's distribution. It only pays on off-distribution inputs, and for those we have no labelled set.
- **Revisit:** when a real-document test set exists (see `eval/real/` if present) **and** a GPU image budget is accepted. Cost: the 520 MB image grows to ≈ 3.6 GB (`0ea8b01`), and the build gets a pinned weight download.

## ADR-016 — Capacity / auto-research pipeline   (2026-11-20, status: not built)
- **Context:** Idea from 20 Nov (voice note "AgenticSolveAutoResearchDocker"). Several agents or containers would each optimise one pipeline component (preprocessing, structure, OCR) against the same baseline. Proven improvements and their decision records would be pooled in one shared pipeline. A capacity sweep would show how score scales with model size.
- **Options:** A build the orchestration / B do it by hand: one track per person/agent, numbers into MEASUREMENTS/RESULTS
- **Choice:** **B happened in practice; A was not built.**
  - The VLM board (ADR-004) covers the capacity part: models from 0.26B to 14B on one scorer. Score does not grow with size: MinerU2.5 1.2B 0.9659 vs Ministral 3 14B (8B build) 0.7728.
  - The tablesalt OCR tuning in [docs/RESULTS.md](../docs/RESULTS.md) is the manual version of the auto-research loop: one change, one measured row, regressions reverted.
- **Evidence:** The voice note from 20 Nov ("Agentenoptimierung … belastbare Validierung") names the risk of automating this. Picking the best of many variants on the same 24 dev tables overstates the winner (winner's curse). Any automated loop needs a fresh seed per round plus a final untouched set, like fresh96 here.
- **Revisit:** after the hackathon, if the pipeline is reused on real documents.
