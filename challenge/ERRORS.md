# Error analysis

Filled in during plan step M1 (and kept current). One line per finding: what breaks, on which tier
and style, how much it costs, one example table.

## Error classes (measured on `MinerU2.5 1.2B`, `dev` + `val96` = 120 tables)

| Class | Definition | Cost | Typical tier/style | Examples |
|---|---|---|---|---|
| `unicode-dash` | ASCII `-` output instead of `–` (1×1) or `—` (`colspan=2`) | high (`+0.0124` dev / `+0.0085` val96) | all tiers (39/96 val96, 11/24 T3) | `dev/table_003`, `val96/table_039` |
| `split-placeholder` | `colspan=2` placeholder (`—`, `n/a`, `see note 3`) split into `(ph, "")` or `("-", "-")` | high (`+0.0056` val96 post-P4) | `borderless` & `hlines` (17/96 val96, 6/24 T3) | `dev/table_014`, `val96/table_007`, `val96/table_079` |
| `illegal-numeric-colspan` | number merged with empty neighbor as `cs=2` (or phantom column `cs=2`) | high (`+0.0055` dev, `+0.0032` val96 w/ P0) | T2/T3 `borderless` & `hlines` (4/96 val96 + 1 dev) | `dev/table_023`, `val96/table_043`, `val96/table_055`, `val96/table_038` |
| `missing-corner-rowspan2` | empty `(0,0)` corner in 2-row header split into two 1×1 cells | medium (`+0.0014` val96) | `borderless` / `hlines` (4/96 val96, 1 dev) | `dev/table_008`, `val96/table_007`, `val96/table_079` |
| `collapsed-header-row` | 2 header rows fused into 1 (`nrows` off by 1) → **dominant QA killer** | very high on QA (15/19 val96 QA misses) | `hlines` / `borderless` (4/96 val96, 1 dev) | `dev/table_016` (0/5 QA), `val96/table_050` (0/5 QA), `val96/table_094` (1/5 QA) |
| `number-format` | space-thousands dropped `(2079)` or misread as dot `21.545` | low-medium (2/96 val96, both T3) | T3 `grid` / `hlines` | `val96/table_003`, `val96/table_043` |
| `text` | umlauts dropped (`Vorräte`→`Vorrate`, `Mär`→`Mar`), `Item`→`ltem`, rare digit `6`→`8` | low (9/96 text + 1/96 digit in val96; 0/24 dev) | T2/T3 | `val96/table_003`, `val96/table_075`, `val96/table_067` |

## Per tier × style, floor baseline (`predictions_tess`, dev 24 tables)

| | grid | hlines | borderless |
|---|---|---|---|
| T0 | 0.0842 | 0.7687 | 0.7650 |
| T1 | 0.1170 | 0.6474 | 0.6637 |
| T2 | 0.0168 | 0.4465 | 0.3040 |
| T3 | 0.0346 | 0.5054 | 0.4279 |

## Per tier × style, `MinerU2.5 1.2B` (`val96`, 96 tables: raw → +P0..P5)

| | grid | hlines | borderless |
|---|---|---|---|
| T0 | 0.9985 → 1.0000 | 0.9795 → 0.9858 | 0.9814 → 1.0000 |
| T1 | 0.9958 → 1.0000 | 0.9691 → 0.9864 | 0.9588 → 0.9632 |
| T2 | 0.9868 → 0.9979 | 0.9254 → 0.9686 | 0.9228 → 0.9592 |
| T3 | 0.9832 → 0.9988 | 0.9252 → 0.9834 | 0.9813 → 0.9852 |

## Top 3 things to fix (with verified `pipeline/postprocess.py` results)

1. **Unicode dash normalization (`P1`, text-only):** `+0.0124 FINAL` on `dev` (`0.9659 → 0.9783`, `T3: 0.9658 → 0.9837`) and `+0.0085 FINAL` on `val96` (`0.9621 → 0.9705`, `T3: 0.9632 → 0.9718`).
2. **Pixel-geometry boundary straddle & illegal numeric colspan fusion (`P0` + `P3` + `P5`):** collapses phantom columns (`val96/table_043`), splits illegal numeric `cs=2` cells via deskewed pure-NumPy ink column projection (`dev/table_023`, `val96/table_055`, `0.9891` T3 on `val96`, `1.0000` T3 on `dev`), and merges split `cs=2` placeholders only when ink straddles the column boundary (`0/120` GT harmed). Together `P0..P5` reach **`0.9862` FINAL (`1.0000` T3)** on `dev` and **`0.9808` FINAL (`0.9891` T3)** on `val96` with 0 regressions.
3. **Remaining unfixed wall — collapsed 2-row headers (`dev/table_016`, `val96/table_050`, `070`, `094`):** responsible for 100% of `dev` QA misses (`5/5`) and 15/19 `val96` QA misses. Note: T3 image preprocessing (`median3`, `upscale15_median3`, `despeckle1`) did *not* beat `raw + P0..P5` end-to-end (`val96` T3: `0.9891` raw+pp vs `0.9881` despeckle1+pp, `0.9740` median3+pp).
