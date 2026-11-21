# Parking

Ideen ohne Zahl. Hier landet alles, was sich gut anfühlt, aber keine Fehlerklasse aus
[ERRORS.md](ERRORS.md) fixt und keine Zahl auf `val96` bewegt. Nichts davon wird vor der Abgabe
umgesetzt.

Format: `- <Datum> — <Idee> — <welche Zahl fehlt, um sie zu rechtfertigen>`

- 2026-11-21 — Ordner `baselines/` aufräumen, `scripts/` konsolidieren — keine; reine Hygiene,
  kostet Zeit im Feature-Freeze.
- 2026-11-21 — Mehr Tabellen-Datensätze (train/val/…) definieren — keine; Idee ist gerade erstellt.
  Definition: neue Datensätze `data/train`, `data/val` … aus dem UNCHANGED `generate_dataset.py`:
  Layout pro Tabelle `table_NNN/`: `image.png|jpg` (PNG nur T0, sonst JPEG) + `gt/cells.json` (alle
  Zellen `{text, r, c, rs, cs}` incl. leere, top-left-Position, Spannen), `gt/table.html` (rowspan/colspan),
  `gt/table.md` (nur ohne Spannen), `gt/qa.json` (≈5 lookup+aggregate) + `manifest.json` (id, image, style,
  tier, has_spans, n_rows, n_cols, n_qa). Verteilung: styles grid/hlines/borderless; T0–T3; T2+T3-dominiert;
  4–7 Spalten (erste Stub-Spalte), 6–12 Zeilen, 1–2 Header-Zeilen, gelegentliche Body-Merges, leere Zellen,
  Units (`% € kg ms`), ein Number-Format pro Tabelle (en/de/int); englische+deutsche Labels. Erzeugung:
  `python generate_dataset.py --out data/train --n 5000 --seed 1`. data/dev ist der feste Maßstab,
  scoring unverändert. Traps-Option: PDF-Inputs + Text-um-Tabelle (§8/§9 offen) in einem Val-Satz.
  Runner: scripts/gen_new_datasets.sh. Beleg: gescorte Läufe mit Zahl in MEASUREMENTS.md.
- 2026-11-21 — Makefile für Windows-User (kein existierendes) — keine; Windows abgedeckt per README (README.md:55,263: Arial-Aktivierung, `.venv\Scripts\activate`, PowerShell-Beispiele), ein Makefile wäre nur Komfort: FINAL läuft über `score_all.py --gt data/dev --pred out` (Stand dev 0.9862 / val96 0.9808, QA 0.960), baut nichts bewiesenes auf ein neues Lineal.
