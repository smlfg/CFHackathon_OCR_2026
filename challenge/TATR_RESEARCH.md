# Table Transformer (TATR) — research note

**Date:** 2026-11-20 · **Source:** [microsoft/table-transformer](https://github.com/microsoft/table-transformer) ·
**Question:** does TATR earn a slot in our pipeline, and where?

Read this together with [PLAN.md](PLAN.md) §M3 candidate 3 and [DECISIONS.md](DECISIONS.md) open decision #1.

---

## 0. Verdict first

**Measured 2026-11-20** (§6): with two cheap patches outside the model (median denoise + NMS on
rows/columns), TATR's grid reaches **weighted TEDS 0.9365 on `data/dev`, 23/24 tables with exactly the
right shape** — but only **0.7289 / FINAL 0.6608 once real OCR carries the text**. The VLM line is
0.9659. So: *structure good enough, text is the wall.*

| Claim | Confidence | Basis |
|---|---|---|
| TATR is a **structure detector, not an OCR**. It outputs row / column / span / header boxes. Text has to come from somewhere else. | verified | `src/inference.py`, README; measured §6 |
| As a **drop-in replacement** for our best VLM (MinerU2.5 1.2B, final 0.9659) it loses. Its own grid is 0.937 with *perfect* text; with Tesseract it is 0.661. | **measured** | §6 rows C and D |
| Its grid is genuinely good on degraded input, **including fax** (T3 0.946 with oracle text), once duplicates are suppressed. | **measured** | §6 row C |
| As a **fine-tuned structure specialist** on our generator it is the cheapest route to exact rows / cols / spans we can own. | medium | 28.8M params, exact synthetic labels, unmeasured |
| As a **validator** for VLM output it is cheap (~0.1 s/table, 110 MB) and can flag structurally implausible tables without a second VLM call. | medium | runtime measured, signal unmeasured |
| The repo is **archived** (2026-09-21). No upstream fixes. MIT-licensed and self-contained, so shipping it is legally fine. | verified | GitHub repo page |

**Recommendation:** not the submission path. Two openings left, both cheap: (a) **validator** on top of
the VLM's HTML (row / column / span disagreement → fallback ladder), (b) **fine-tune** on the generator
if structure ever becomes the binding constraint — it currently is not.


---

## 1. What the repo actually is

Three things shipped together, which is why it looks bigger than it is:

1. **TATR models** — DETR (ResNet-18 backbone) as an object detector. Two separate models:
   - **detection**: page image → table / table-rotated boxes (AP 0.970, AP50 0.995 on PubTables-1M test)
   - **structure**: cropped table image → objects of 6 classes: `table`, `table column`, `table row`,
     `table column header`, `table projected row header`, `table spanning cell`
   - structure model: **28.8M params**, fp32 checkpoint **110 MB**
2. **PubTables-1M** — 575,305 annotated pages (detection), 947,642 annotated tables (structure),
   from PubMed Central PDFs. License **CDLA-Permissive-2.0**, 117 GB, on HF.
3. **GriTS** — the grid table similarity metric from the same group.

Published numbers (README, TATR-v1.0 on PubTables-1M test): GriTS_Top 0.9849, GriTS_Con 0.9850,
GriTS_Loc 0.9786, Acc_Con 0.8243.

The follow-up paper *Aligning benchmark datasets for TSR* (ICDAR 2023) matters more for us than the
headline numbers: with the architecture held fixed, cleaning up annotations moved TATR **exact match on
ICDAR-2013 from 65% → 75%** (PubTables-1M), **42% → 65%** (FinTabNet), **69% → 81%** (combined).
Lesson: on this task family, *label quality* moves the number as much as the model does — which is
directly relevant to us because our generator produces exact labels.

## 2. How inference actually works (read from the source, not the paper)

```
page image ──detection (MaxResize 800)──► table boxes (+ "table rotated" = 90° CCW)
   └─ crop with 10 px pad, rotate if flagged
        └─ structure (MaxResize 1000) ──► rows, columns, spanning cells, column header
             └─ postprocess: threshold 0.5 → NMS → align_headers → align_supercells
                  └─ cells = row ∩ column, minus subcells covered >50% by a spanning cell
                       └─ slot OCR words into cells (postprocess.slot_into_containers)
                            └─ cells → HTML / CSV
```

Consequences that decide integration:

- **Line-agnostic.** Rows and columns are learned objects, not detected rules. That is exactly the
  weakness of classic CV on our `borderless` style — TATR should be strongest there, not weakest.
- **Merged cells are a first-class class** (`table spanning cell`, plus `projected row header`), and
  cells are derived by subtraction, so `rowspan` / `colspan` come out of geometry, not heuristics.
- **Empty cells exist by construction** — every row × column intersection is emitted, including blank
  ones. Directly relevant to our `missing-empty-cells` error class.
- **Text is a separate problem.** TATR consumes word boxes (`{"bbox": [...], "text": "..."}`) and
  assigns them to cells by overlap. Text quality = our OCR quality, unchanged.
- **Single header only.** `align_headers()` in `src/inference.py`: *"we are not supporting tables with
  multiple headers… this ignores any subsequent rows labelled as a header"*. Our GT has two-row
  headers (`rowspan=2` in `<thead>`). The cells still exist; the header *classification* is flattened.
- **No arbitrary skew / perspective.** The detection model has exactly one rotation class, 90° CCW.
  T1–T3 skew and phone perspective are out of distribution → **deskew must happen before TATR**, or
  axis-aligned row boxes will overlap and the grid intersection will be wrong.
- **HTML writer is rudimentary** (`cells_to_html`): one `<thead>` element per header row, no `scope`,
  no `<tbody>`. If we use TATR we write our own cells → HTML from `row_nums` / `column_nums`.
- **`fitz` (PyMuPDF, AGPL) is imported only for `Rect`.** Replace with 4 floats before shipping.

## 3. Domain gap to *our* data — the part that matters

| PubTables-1M (training) | TableSalt (test) |
|---|---|
| PDF pages rendered by PyMuPDF, max side 1000 px | 500–1110 px images (measured on `data/dev`) |
| table cropped, padded 30 px, max side 1000 px | one table per image, already cropped |
| clean raster, axis-aligned | T1 scan, T2 phone photo (perspective, shadow), T3 fax (binarized, blurred) |
| Latin scientific typesetting, mostly grid/hlines | Arial-like, grid / hlines / **borderless**, German number formats (`79.036,73 %`) |
| up to ~125 objects | ~13 rows × 4 cols ≈ 25 objects → queries are not exhausted |

Two useful consequences:

- **Resolution is not the problem.** Our images are already at ≈ the training resolution; nothing gets
  destroyed by `MaxResize(1000)`.
- **The degradation tiers were the predicted problem — and only T3 was.** Prediction was "T2/T3 fall off
  a cliff". Measured: T2 holds at 0.93 and only T3 collapses (0.08) — not because of skew, but because
  fax speckle noise produces duplicate row / column detections. Fixed with a median filter + NMS (§6).

There is **no published evaluation of TATR on photographed or faxed tables**. Anyone quoting one is
guessing. We measure it or we don't claim it.

## 4. Fit, honestly costed

| | TATR | our best VLM (MinerU2.5) |
|---|---|---|
| structure on borderless | strong by design | learned, decent |
| merged cells | explicit class + geometric derivation | OTSL/HTML tokens, occluded-cell errors |
| empty cells | emitted by construction | frequently dropped |
| text | none | built in |
| number formats | none | none (post-processing either way) |
| skew / perspective | needs deskew in front | handled implicitly |
| determinism / debuggability | deterministic bbox output, easy to diff | sampling, harder to attribute |
| weight size | 2 × 110 MB fp32 (≈28 MB int8) | 1.2B params |
| runtime | estimate 1–3 s/table CPU, measure | 2.6 s/table (measured) |
| trainable on our generator | **yes** — 28.8M, VOC XML, hours on one GPU | no (out of scope for a weekend) |
| current dev score | n/a (no text) | **0.9659 final** |

The honest reading: TATR's value is *not* "better OCR". It is that it is **small, deterministic and
trainable**, and that its output is a grid — which is exactly the intermediate representation our
`pipeline` currently only gets implicitly from a VLM's markup.

## 5. Where it could pay off, in order of expected value

1. **Structural validator on top of the VLM (cheapest to try, now the best bet).** Run TATR structure
   only. Compare its row / column count and span geometry with the VLM's HTML. Disagreement = low
   confidence = route to the fallback ladder (open decision #7). No extra text model, 110 MB, 0.1 s/table.
2. **Fine-tuned structure specialist.** `generate_dataset.py` gives exact HTML with spans; deriving row /
   column / span boxes from cell boxes is arithmetic; 5,000 tables → DETR-R18 fine-tune. Worth it only if
   structure ever becomes the binding constraint — measured today, it is not.
3. **Ensemble seat / tie-breaker** between two VLMs — same machinery as #1.
4. **Full replacement.** Dead on measurement: 0.7289 TEDS with real OCR, and the Tesseract stage alone
   costs 0.21 TEDS on T3.

## 6. Results — measured 2026-11-20 (`scripts/tatr_experiment.py`)

TATR structure only (`microsoft/table-transformer-structure-recognition`, HF v1.0, 28.8M),
Apple M-series MPS, `MaxResize(1000)`, threshold 0.5, cells = row ∩ column minus spanning subcells.
TEDS via our own `evaluate.py`; "oracle" = TATR's grid filled with **ground-truth text**, so this
column measures structure only.

| # | configuration | T0 | T1 | T2 | T3 | weighted TEDS | shape exactly right |
|---|---|---|---|---|---|---|---|
| A | as-is | 0.913 | 0.931 | 0.928 | **0.083** | 0.589 | 18/24 |
| B | + entropy deskew (naive) | 0.913 | 0.931 | 0.719 | 0.084 | 0.527 | 14/24 |
| — | + NMS on rows/cols (0.3) | 0.913 | 0.931 | 0.928 | 0.149 | 0.616 | 18/24 |
| **C** | **+ 3×3 median denoise + NMS 0.3** | 0.913 | 0.931 | 0.936 | **0.946** | **0.9365** | **23/24** |
| C′ | C without spanning cells | 0.902 | 0.926 | 0.915 | 0.930 | 0.9217 | 23/24 |
| D | C + Tesseract per cell (no oracle) | 0.831 | 0.877 | 0.738 | 0.623 | 0.7289 | 23/24 |

Runtime: **0.08–0.17 s/table** on MPS (structure only, no OCR).

### What the numbers say

1. **The T3 collapse is not a capacity or a resolution problem — it is duplicate detections on
   sensor noise.** Without denoise, one real row produces 2–3 overlapping boxes and one column 3–4;
   a 13×7 table comes out as 20×39 and TEDS 0.08. Two cheap fixes, both outside the model:
   a **3×3 median filter** before inference and **greedy NMS on `table row` / `table column`**
   (IoU ≥ 0.3, stable from 0.2–0.3). Together: T3 0.083 → **0.946**.
   The official TATR post-process NMS-es headers and spanning cells but *not* rows and columns —
   it relies on DETR's set prediction, which fails off-distribution. That is the one real defect for
   our data, and it is ours to patch.
2. **Naive deskew hurts.** Entropy-based rotation on already-upright images costs 0.21 TEDS on T2.
   Whatever rectification we do must be gated on a skew estimate, not applied unconditionally.
3. **The grid itself is good on our data, including fax.** 23/24 tables get exactly the right
   row × column shape, merged cells included. The borderless hypothesis holds: T2 borderless scores
   0.936 with no line information at all.
4. **Text is the bottleneck, structure is not.** Same grid, oracle text → TEDS 0.946 on T3;
   Tesseract text → 0.623. That is 0.32 TEDS lost to characters alone. On this pipeline the
   challenge's real question ("OCR quality or table structure?") has a measured answer for the
   degraded tiers: **structure**.
   Official score of D: **FINAL 0.6608** (TEDS_w 0.7289, QA 0.275, CellF1 0.4966) — far below the
   VLM line (0.9659), for exactly that reason.
5. **Residual structure error is spanning cells, not the grid** (−0.015 if we drop them), plus one
   shape failure. Cheap post-processing territory, not a model problem.

### Verdict against the gate in §6

C = **0.9365** → inside the 0.90–0.95 band: **validator / fine-tuning base, not the primary model.**
It is below what the VLM already delivers end to end (0.9612 TEDS-weighted *with* text), so a
structure-only path can only win by being paired with text at least as good as the VLM's — at which
point the VLM is the cheaper answer.

Reproduce:

```bash
python scripts/tatr_experiment.py --gt data/dev --out out/tatr_dn --nms 0.3 --denoise
python scripts/tatr_experiment.py --gt data/dev --out pred_tatr_tess --nms 0.3 --denoise --text tesseract
python score_all.py --gt data/dev --pred pred_tatr_tess
```


## 7. If we do integrate — the traps

- HF Hub only has **v1.0**. v1.1 (better, aligned FinTabNet) ships as `TATR-v1.1-*.pth` + their own
  `src/` (pinned torch 1.13, `timm`, `PyMuPDF`). Convert weights or accept v1.0; don't mix.
- Pin `revision=` for every HF download; the repo is archived but the Hub artifacts are not frozen.
- Replace `fitz.Rect` usage (AGPL) with plain floats — we only need intersection areas.
- Keep `num_queries=125`; do not lower it, we are nowhere near the limit.
- Write our own cells → HTML; do not reuse `cells_to_html`.
- `align_headers` flattens multi-row headers — our GT has them, so expect a systematic loss on
  two-row-header tables that is *ours to fix in post-processing*, not the model's fault.
- Two models or one? Our images are already a single cropped table → skip the detection model entirely.
  Saves ~110 MB and one forward pass per table.

## 8. Neighbours worth one hour each, if TATR fails §6C

| option | why it exists | catch |
|---|---|---|
| **Docling / TableFormer** (IBM, MIT) | structure + spans with a transformer, ships an end-to-end doc pipeline incl. OCR | heavier dependency tree; slower; still no number-format logic |
| **PP-StructureV3** (PaddleOCR) | structure + text + OTSL, already in our stack as a VLM provider | licence/deps, and we already score a Paddle line |
| **img2table / classic CV** | no weights, fast, pure geometry | dies on borderless, which is where the weight is |
| **our current VLM + TATR validator** | no new text model, uses what we already trust | only as good as the disagreement signal |

## 9. Sources

- Repo + README, pretrained table, evaluation table, archived banner — https://github.com/microsoft/table-transformer
- `src/inference.py` (transforms, class maps, cell derivation, header limitation) — repo, read 2026-11-20
- PubTables-1M, CVPR 2022 — https://arxiv.org/abs/2110.00061 (training renders: max side 1000 px, 30 px crop padding)
- Aligning benchmark datasets for TSR, ICDAR 2023 — https://arxiv.org/abs/2303.00716 (65→75 / 42→65 / 69→81)
- GriTS, ICDAR 2023 — https://arxiv.org/abs/2203.12555
- HF model card, 28.8M params, MIT — https://huggingface.co/microsoft/table-transformer-structure-recognition
- PubTables-1M dataset card, CDLA-Permissive-2.0, 117 GB — https://huggingface.co/datasets/bsmock/pubtables-1m
- Our own numbers — [../metrics/LEADERBOARD.md](../metrics/LEADERBOARD.md)
