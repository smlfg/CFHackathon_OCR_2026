# Error analysis

Filled in during plan step M1 (and kept current). One line per finding: what breaks, on which tier
and style, how much it costs, one example table.

## Error classes

| Class | Definition | Cost | Typical tier/style | Examples |
|---|---|---|---|---|
| `structure` | rows/columns/spans wrong, cells merged or split | high | T2/T3 · borderless | |
| `text` | characters wrong, umlauts lost | medium | T3 | |
| `number-format` | separator/sign/thousands wrong; format not normalized per table | high (also QA) | all | |
| `empty-cell` | empty cells not written, or invented | medium | all | |
| `row-label` | first-column label misread → QA lookup fails | high (QA) | T2/T3 | |
| `header-row` | column header in the first three rows misread → QA fails | high (QA) | T2/T3 | |
| `preprocessing` | skew/perspective/shadow/binarization not corrected | indirect | T2/T3 | |

## Per tier × style, floor baseline

| | grid | hlines | borderless |
|---|---|---|---|
| T0 | | | |
| T1 | | | |
| T2 | | | |
| T3 | | | |

Cell = mean TEDS for that slice on dev (or val once generated).

## Submission pipeline (`tablesalt/`, main) — remaining errors, 2026-11-22

Measured with `tools/worst_tables.py` on **fresh96** (`generate_dataset.py --n 96 --seed 20261122`,
Liberation Sans, generated after all tuning). FINAL 0.9931, TEDS-S 0.9997, QA 0.969, 0 fallbacks.
Structure is essentially solved; what is left is **digits and signs**, and it hits QA hardest
(15 failed questions, 12 of them aggregates where one wrong digit spoils the sum).

| Class | What we see | Tiers | Examples (fresh96) |
|---|---|---|---|
| `text` (digit) leading `5` → `9` | `53,326` → `93,326`, `55.517,72` → `95.517,72` | **T0 too**, all styles | `table_072`, `table_012`, `table_000` |
| `number-format` (sign) minus lost | `-5,918.44` → `5,918.44`, `-552` → `-52` | T2/T3 | `table_039`, `table_047`, `table_054` |
| `text` (digit) dropped/added leading digit | `88.112,81` → `8.112,81`, `5.682,96` → `682,96` | T3 | `table_027`, `table_047`, `table_087` |
| `empty-cell` en dash read as empty | `–` → `""` | T3 | `table_027` |
| `structure` em-dash merge missed | `—` (cs=2) → `""` + `–` | T3 grid | `table_039` (only TEDS-S < 1 in the set) |

**Caveat on the `5` → `9` class:** dev (Arial) shows it far less (dev QA 0.983, CellF1 0.998).
LEARNLOG M6 already describes it ("tight crop cuts the flat top of the 5 → 9"). fresh96 is rendered
with Liberation Sans, whose `5` differs slightly from Arial's, so part of this class is probably the
font, not the pipeline. We did **not** tune on fresh96, to keep it untouched.

## Top 3 things to fix (update as they get fixed)

1. Leading-digit `5`/`9` confusion: a 4th crop variant with more top margin only for the first
   digit, or a vote that prefers the variant whose first glyph has a flat top. Measure on a fresh
   Arial seed, not on fresh96.
2. Lost minus sign on T2/T3: the geometry check (`lead_dash`) misses dashes merged into noise.
3. En dash read as empty on T3: an empty body cell in a column of numbers is suspicious in this
   generator, so it could be re-read with `psm 13`.

The error analysis of the VLM track (MinerU2.5, `unicode-dash`, `split-placeholder`,
`collapsed-header-row`, …) is on branch `feat/vlm-ocr-bench`, `challenge/ERRORS.md`.
