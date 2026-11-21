# NOTES – what the generator and the scorer really do (milestone M0)

Source of truth: `generate_dataset.py`, `evaluate.py`, `eval_qa.py`, `score_all.py`.
Every heuristic in `tablesalt/` points back to a line in this file.

## 1. Table content (`make_table_spec`)

| Item | Rule |
|---|---|
| Body rows | `randint(6, 12)` |
| Columns | `randint(4, 7)`, column 0 = row labels |
| Header rows | 2 with p = 0.45, else 1 → total rows 7–14 |
| Row labels | `rng.sample(pool, n_rows)` from **one** of 4 pools (EN finance, DE finance, elements, ML variants) → unique within a table |
| Column labels | `(pool * 2)[: n_cols - 1]` from one of 5 pools → **always `pool[0..n-2]` in order**, never repeated (max 6 data columns, pools have 6) |
| Stub header (0,0) | one of `Position`, `Item`, `Metric`, `""` |
| 2 header rows | (0,0) has `rs=2`; row 0 = group labels partitioning columns 1..n-1 into consecutive runs of `randint(1,3)` columns; names from `Actual, Plan, Delta, Reported, Adjusted` (shuffled, `gi % 5`); row 1 = column labels |
| Body merge | p = 0.06 per position, only `j < n_cols-1`, always **`cs=2`, horizontal**, text `n/a`, `—` (U+2014 em dash) or `see note 3` |
| Single specials | p = 0.05: `–` (U+2013 en dash), `n/a`, `""` |
| Unit | one per table from `"", " %", " €", " kg", " ms", ""`, appended to ~10 % of numbers with a normal space |
| Number range | `uniform(-9999, 99999)` |

Pools: see `tablesalt/lexicon.py` (copied verbatim).

### Number formats (one per table)
* `en`: `f"{v:,.2f}"` → `-1,234.56`, always 2 decimals, ASCII hyphen-minus.
* `de`: en with `,`↔`.` swapped → `-1.234,56`.
* `int`: `f"{int(v):,}"` with separator `,` or **normal space U+0020** (not NBSP) → `12,345` / `12 345`. `int(-0.4) = 0` → no `-0`.
* parens with p = 0.4: negatives become `(1,234.56)` (the `-` is removed, unit goes after `)`).

## 2. Rendering (`render_table`)

* Fonts: Arial regular/bold, **20 px** for body and header (headers bold). Cap height ≈ 14–15 px.
* `pad=60` around the table, `cell_pad_x=14`, `cell_pad_y=10`.
* Column width = widest single-span text in that column + 28 (min 40). Row height = text height + 28 (rs=2 cells contribute half).
* ⇒ **gap between texts of neighbouring cells ≥ 28 px**; word spaces ≈ 6 px. One horizontal dilation of ~0.85·h joins a cell's words and never two cells.
* Alignment: headers centred; column 0 body **left** (x0 + 14); numbers (anything `parse_num` accepts, units included) **right** (x1 − 14); `n/a`, dashes, `see note 3`, empty centred.
* Header fill (228) with p = 0.6; zebra stripes (240) with p = 0.4 for non-grid styles.
* Styles:
  * `grid`: every cell's rectangle with width 2 (merged cells have no inner border), outer box width 3.
  * `hlines`: exactly 3 rules: top (w3), under the header (w2), bottom (w3), each from `col_x[0]` to `col_x[-1]`.
  * `borderless`: no rules.
* Every cell is **one line of text**.

## 3. Degradation (`TIER_PARAMS`)

| | rotate | perspective | shadow | bleed | blur | noise σ | specks | contrast | binarize | JPEG |
|---|---|---|---|---|---|---|---|---|---|---|
| T0 | – | – | – | – | – | – | – | – | – | PNG |
| T1 | ±1.5° | – | – | – | 0.3–0.6 | 6 | – | 0.95 | – | 55 |
| T2 | ±4° | ≤ 4 % per corner | gradient to 0.55–0.8 | – | 0.3–1.0 | 10 | – | 0.85 | – | 45 |
| T3 | ±4° | ≤ 6 % per corner | – | mirrored ghost 12–20 % | 0.3–1.0 | 12 | 0.2 % | 0.9 | <110→0, >200→255 | 35 |

Rotation uses `expand=True` and paper-coloured fill; perspective jitters each corner independently
(→ rows/columns converge, up to ~1 row height across a T3 table — deskew alone is not enough).

## 4. Scoring

* `evaluate.norm_text`: NFKC, NBSP→space, collapse whitespace, strip. Nothing else: `–` vs `-` vs `—` count, case counts.
* TEDS: APTED on `table → tr → td`; th/td unified; rename cost 1 if colspan/rowspan differ, else normalized Levenshtein of the text. A missing row/cell or wrong span costs a whole node.
* TEDS-S: same without text (structure only).
* JSON → HTML by `cells_json_to_html`: rows from `r`, sorted by `c`. **Every cell incl. empty ones must be written.**
* QA (`eval_qa.auto_answer`): row = first row whose **column 0** equals the label (casefold, whitespace-normalized); column = first column whose text in **rows 0–2** equals the header; answer = that cell; numbers compared by value (`1,234.56` = `1.234,56`; `12,345` is thousands), tolerance 0.011. Aggregates need **all** listed rows found.
* Final = 0.85 · (0.1 T0 + 0.2 T1 + 0.3 T2 + 0.4 T3) + 0.15 · QA. Tie-break cell F1.

## 5. Consequences used in the pipeline

1. Blobs (dilated text components) = cells. Rows = clusters of blob y-centres.
2. Columns: right-aligned numbers share a right edge per column (merged/centred cells don't disturb it); `grid` → vertical rules; empty-corridor search only as last resort.
3. Perspective: left edge of row labels + a right number edge + first/last text row are 4 straight lines → homography, for all styles.
4. rs=2 stub text sits exactly half-way between header rows 0 and 1 → shows up as its own one-blob "row".
5. Group labels are centred over their run → DP partition of columns 1..n-1.
6. `—` and `see note 3` are always cs=2; `–` never; `n/a` decided by centre position. Dash width tells em (≈ 20 px) from en (≈ 11 px).
7. Column headers = `pool[:n-1]` once the pool is identified; row labels snap to one pool, one-to-one.
8. Numbers are re-printed from their digits in the table's majority format (2 decimals for en/de).
