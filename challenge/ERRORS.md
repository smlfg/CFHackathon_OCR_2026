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

## Top 3 things to fix (update as they get fixed)

1.
2.
3.
