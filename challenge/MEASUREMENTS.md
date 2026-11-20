# Measurements

One row per scored run. Never edit an old row; add a new one. Numbers come from
`python score_all.py --gt <dataset> --pred <dir>`; keep the raw line in `notes/` if it is long.

| Date | What changed | Dataset | T0 | T1 | T2 | T3 | QA | Final | s/table | Who |
|---|---|---|---|---|---|---|---|---|---|---|
| | _baseline Tesseract (template floor)_ | dev (24) | 0.54 | 0.46 | 0.26 | 0.35 | 0.267 | 0.348 | | organizer |
| | | | | | | | | | | |

## Notes on method

- `data/dev` = official yardstick, 24 tables, 6 per tier. Never trained on.
- `data/val` = generated decision set (`--n 240 --seed 2`), used whenever dev would be noise.
- A change counts as an improvement only if it wins on **both** sets, or wins on val and is neutral
  on dev.
- Record the wall-clock per table; the 15 s/table limit is a hard gate, not a preference.
