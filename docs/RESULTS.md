# Results

Every measurement, newest at the bottom. Written by `python tools/eval_local.py --gt <set> --note "..."`.
`data/val` = `python generate_dataset.py --out data/val --n 200 --seed 2` (Arial, Windows).
s/table = mean time per table inside one worker (Tesseract 5.4.0, Windows, 12 workers).

| Date | Change | Set | Final | T0 | T1 | T2 | T3 | QA | TEDS-S | s/table | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-21 | Tesseract floor baseline (organizer script) | dev (24) | 0.3480 | 0.539 | 0.458 | 0.256 | 0.350 | 0.267 | - | - | - |
| 2026-11-21 | M1-M7 first full pipeline (blobs, edges, spans, lexicon, number format) | dev (24) | 0.9748 | 1.000 | 0.998 | 0.975 | 0.979 | 0.925 | 0.999 | 3.27 | 0 |
| 2026-11-21 | M1-M7 first full pipeline | val (200) | 0.9799 | 1.000 | 0.999 | 0.980 | 0.978 | 0.951 | 0.998 | 4.44 | 0 |
| 2026-11-21 | OCR crop cap40/margin1/median3; long-rule filter; n/a + dash rules; label snap 40 | val (200) | 0.9906 | 1.000 | 1.000 | 0.993 | 0.990 | 0.971 | 0.998 | 4.32 | 0 |
| 2026-11-21 | 2 OCR reads/blob + role-based pick; minus geometry; n/a merge rule; whitelist pass on | val (200) | 0.9936 | 1.000 | 1.000 | 0.996 | 0.994 | 0.976 | 0.999 | 6.29 | 0 |
| 2026-11-21 | same, whitelist pass off | val (200) | 0.9936 | 1.000 | 1.000 | 0.996 | 0.994 | 0.976 | 0.999 | 5.23 | 0 |
| 2026-11-21 | OCR variants 1:3:13,3:0:13 (margin:median:psm) | val (200) | 0.9785 | 0.999 | 0.998 | 0.991 | 0.989 | 0.901 | 0.999 | 4.94 | 0 |
| 2026-11-21 | OCR variants 1:3:7,3:0:13 (margin:median:psm) | val (200) | 0.9935 | 1.000 | 1.000 | 0.997 | 0.994 | 0.975 | 0.999 | 5.30 | 0 |
| 2026-11-21 | OCR variants 1:3:13,3:0:7 (margin:median:psm) | val (200) | 0.9931 | 1.000 | 1.000 | 0.997 | 0.994 | 0.973 | 0.999 | 5.05 | 0 |
| 2026-11-21 | psm 7 x2 + psm 13 rescue for empty reads | val (200) | 0.9952 | 1.000 | 1.000 | 0.997 | 0.996 | 0.983 | 0.999 | 5.23 | 0 |
| 2026-11-21 | 3 reads + digit vote: 1:3:7,3:0:7,2:0:7 | val (200) | 0.9963 | 1.000 | 1.000 | 0.997 | 0.996 | 0.989 | 0.999 | 8.65 | 0 |
| 2026-11-21 | 3 reads + digit vote: 1:3:7,3:0:7,2:3:7 | val (200) | 0.9958 | 1.000 | 1.000 | 0.997 | 0.996 | 0.986 | 0.999 | 7.20 | 0 |
| 2026-11-21 | default: 3 reads (1:3:7,3:0:7,2:0:7) + digit vote, clean timing | val (200) | 0.9963 | 1.000 | 1.000 | 0.997 | 0.996 | 0.989 | 0.999 | 7.38 | 0 |
| 2026-11-21 | blob_join_x 0.85 -> 0.65 (REVERTED: worse) | val (200) | 0.9934 | 1.000 | 1.000 | 0.997 | 0.991 | 0.983 | 0.998 | 7.54 | 0 |
| 2026-11-21 | final config (3 reads, join 0.85) | dev (24) | 0.9971 | 1.000 | 1.000 | 1.000 | 0.999 | 0.983 | 1.000 | 6.41 | 0 |
