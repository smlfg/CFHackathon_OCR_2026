# AGENTS.md – TableSalt-Pipeline (Python + Tesseract)

Diese Datei ist die Arbeitsanweisung für einen Coding-Agenten (Cursor, Codex, Claude Code …).
Lies sie komplett, bevor du Code schreibst. Die Wettbewerbsregeln stehen in `README.md` –
bei Widerspruch gilt die README.

---

## 1. Ziel

Baue eine Pipeline **Tabellenbild → `<table_id>.json`** (Liste von Zellen mit `text`, `r`, `c`,
`rs`, `cs`) in **Python**, mit **Tesseract über `pytesseract`** als OCR und **OpenCV/NumPy** für
die Bildverarbeitung. Keine großen KI-Modelle in dieser Phase.

Maßstab ist ausschließlich:

```
python score_all.py --gt <dataset> --pred <predictions>
final = 0.85 × (0.1·T0 + 0.2·T1 + 0.3·T2 + 0.4·T3 TEDS) + 0.15 × QA
```

Startwert (Tesseract-Baseline): **≈ 0.35**. Jede Änderung muss sich an dieser Zahl messen lassen.

## 2. Architektur-Idee: erst Gerüst, dann Text

```
Bild
 └─ 1. preprocess   gerade drehen, Perspektive entzerren, Schatten entfernen, binarisieren
 └─ 2. style        grid / hlines / borderless erkennen (aus dem Bild, NICHT aus manifest.json)
 └─ 3. structure    Zeilen- und Spaltengrenzen → leeres Raster mit (r, c)
 └─ 4. spans        Kopfzeilen (1 oder 2) und zusammengeführte Zellen (rs/cs)
 └─ 5. ocr          jede Zelle einzeln ausschneiden und mit Tesseract lesen
 └─ 6. postprocess  Zahlenformat vereinheitlichen, Labels korrigieren, leere Zellen = ""
 └─ 7. write        gültiges JSON sofort schreiben (json.dump)
```

Jeder Schritt ist ein eigenes Modul mit klarer Ein-/Ausgabe, damit er einzeln testbar und
austauschbar ist (z. B. später Tesseract → PaddleOCR).

## 3. Harte Regeln (nie brechen)

1. **Nicht verändern:** `generate_dataset.py`, `score_all.py`, `evaluate.py`, `eval_qa.py`,
   `selftest_workflow.py`, `flatten_dataset.py`, `data/dev/`. Nur lesen.
2. **Container-Vertrag aus `submission_template/` beibehalten:** Der Container bekommt
   `/data/images` und schreibt `/data/predictions/<table_id>.json`. Den ENTRYPOINT/CMD-Mechanismus
   des Template-Dockerfiles übernimmst du unverändert – nur die Interna austauschen.
3. **Zur Laufzeit kein Netzwerk**, keine Dateien außer `/data/images` lesen, nichts an der
   Bewertungsumgebung erkennen oder sonderbehandeln. Tier/Style dürfen **nicht** aus Dateinamen
   oder `manifest.json` gelesen werden – nur aus dem Bild selbst.
4. **Jede Tabelle sofort schreiben**, nicht alle am Ende. Pro Tabelle `try/except`: Wenn etwas
   abstürzt, trotzdem eine gültige Datei schreiben (Fallback, siehe §8). Ein Absturz darf nie den
   ganzen Lauf beenden.
5. **Zeitlimit:** 5 min Start + 15 s pro Tabelle. Ziel: < 8 s pro Tabelle im Container.
6. **Keine großen Dateien committen**, nichts unter `data/` außer `data/dev/` (siehe `.gitignore`).
7. **Deterministisch:** Gleiches Bild → gleiche Ausgabe. Keine Zufallszahlen ohne festen Seed.
8. **Messen vor Behaupten:** Eine Änderung gilt erst als Verbesserung, wenn `score_all.py` auf dem
   Validierungsset (§6) es zeigt. Verschlechterungen werden zurückgenommen.

## 4. Zuerst lesen und verstehen (Meilenstein 0)

Bevor du Pipeline-Code schreibst:

1. `selftest_workflow.py` und die Tesseract-Baseline laufen lassen (README §3). Score notieren.
2. **`generate_dataset.py` gründlich lesen.** Der Testsatz kommt aus genau diesem Code. Notiere in
   `docs/NOTES.md`:
   - wie Zeilen, Spalten, Ränder, Abstände, Schriftgrößen gezeichnet werden (Pixelmaße!),
   - wie die drei Styles Linien zeichnen (welche Linien bei `hlines` genau),
   - wie zweizeilige Header aufgebaut sind (welche Zelle bekommt `rs=2`, welche `cs>1`),
   - wie zusammengeführte Body-Zellen (`n/a`, `—`, `see note 3`) gespannt werden (Richtung, Länge),
   - **exakte Zeichen**: welches Minuszeichen, welcher Gedankenstrich, Leerzeichen-Art bei
     `12 345` (normales oder geschütztes Leerzeichen?), Einheiten-Schreibweise,
   - die Listen möglicher **Zeilen-Labels, Spaltenköpfe und Einheiten** (Vokabular),
   - `TIER_PARAMS`: maximale Drehung, Perspektive, Blur, Binarisierung pro Tier,
   - Grenzen: 4–7 Spalten, 6–12 Body-Zeilen, 1–2 Header-Zeilen.
3. `evaluate.py` und `eval_qa.py` lesen: Wie wird Text normalisiert, wie werden Zahlen verglichen,
   was genau ist „structure-only TEDS“?
4. Drei `data/dev`-Beispiele je Tier mit `gt/table.html` vergleichen.

Diese Notizen sind die Grundlage für alle Heuristiken. Nichts raten, was im Generator steht.

## 5. Projektstruktur

> **Umgesetzt als `tablesalt/`** (nicht `pipeline/`): `pipeline/` enthält bereits die
> VLM-Evaluations-Pipeline des Teams. Aufruf daher `python -m tablesalt.run …`. Sonst gilt die
> Struktur unten unverändert.

```
pipeline/
  __init__.py
  config.py          alle Schwellwerte an einer Stelle (keine Magic Numbers im Code)
  types.py           Dataclasses: Grid, CellBox, Cell
  io_utils.py        Bild laden, JSON atomar schreiben (tmp-Datei + os.replace)
  preprocess.py      Schritt 1
  style.py           Schritt 2
  structure.py       Schritt 3 (Linien- und Weißraum-Variante)
  spans.py           Schritt 4
  ocr.py             Schritt 5
  postprocess.py     Schritt 6
  lexicon.py         Vokabular aus NOTES.md + Fuzzy-Matching
  validate.py        prüft Ausgabe: jede (r,c)-Position genau einmal belegt, Spans im Raster
  run.py             CLI: python -m pipeline.run --images DIR --out DIR [--workers N] [--debug DIR]
tools/
  eval_local.py      flatten + run + score_all in einem Befehl
  debug_overlay.py   zeichnet erkanntes Gitter + OCR-Text über das Bild (PNG)
  worst_tables.py    listet die schlechtesten Tabellen aus evaluate.py nach Tier
tests/               pytest
docs/
  NOTES.md           Erkenntnisse aus Meilenstein 0
  RESULTS.md         Tabelle: Datum, Änderung, final, T0–T3, QA (jede Messung eintragen)
submission_template/
  Dockerfile         angepasst (siehe §9)
  predict.py         ruft nur pipeline.run.main(images_dir, out_dir) auf
```

Python ≥ 3.10, Type Hints, kleine Funktionen, Docstrings. Abhängigkeiten in `requirements.txt`:
`numpy`, `opencv-python-headless`, `Pillow`, `pytesseract`, `rapidfuzz` (Fuzzy-Matching), `pytest`.

## 6. Daten und Messung

**Schriftart:** Dev- und Testset sind mit Arial gerendert. Unter Linux:

```
sudo apt install ttf-mscorefonts-installer
export TABLESALT_FONT=/usr/share/fonts/truetype/msttcorefonts/Arial.ttf
export TABLESALT_FONT_BOLD=/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf
```

(Pfade prüfen. Ohne Arial weichen generierte Daten vom Testset ab.)

**Validierungsset:** `data/dev` (24 Tabellen) ist zu klein für Entscheidungen.

```
python generate_dataset.py --out data/val --n 200 --seed 2
python flatten_dataset.py data/val data/val_flat
```

**Ein Befehl zum Messen** (`tools/eval_local.py`): Pipeline auf `data/val_flat` laufen lassen,
danach `score_all.py` und `evaluate.py`, Ergebnis in `docs/RESULTS.md` anhängen, Laufzeit pro
Tabelle mitprotokollieren.

**Fehler eingrenzen:** structure-only TEDS niedrig → Schritte 1–4 schuld. Struktur gut, TEDS
schlecht → OCR/Postprocessing schuld. Immer zuerst die schlechtesten Tabellen mit
`debug_overlay.py` ansehen.

## 7. Umsetzung pro Schritt

### 7.1 preprocess.py
- Graustufen. **Beleuchtung ausgleichen:** Hintergrund schätzen (große Median-Unschärfe oder
  morphologisches Closing) und Bild durch Hintergrund teilen → Schatten (T2) und Durchscheinen
  (T3) werden hell.
- Leichtes Entrauschen (Median 3×3), dann **binarisieren** (adaptiv oder Otsu nach Normalisierung).
- **Schräglage:** Winkel im Bereich der Generator-Grenzen (aus `TIER_PARAMS`) suchen. Methode:
  Projektionsprofil – Bild in kleinen Schritten drehen, den Winkel wählen, bei dem die
  Zeilensummen die größte Varianz haben (Textzeilen werden scharfe Spitzen). Bei vorhandenen
  Linien alternativ HoughLinesP auf langen horizontalen Linien.
- **Perspektive (T2/T3):** Bei `grid`/`hlines` die äußeren Tabellenlinien finden, 4 Ecken
  bestimmen, `cv2.warpPerspective` auf ein Rechteck. Bei `borderless` aus der Textmaske die
  linke/rechte Textkante pro Zeile fitten. Wenn unsicher: nur Drehung korrigieren.
- Zurückgeben: korrigiertes Graubild **und** Binärbild (OCR liest später vom korrigierten Graubild,
  Strukturerkennung nutzt das Binärbild).

### 7.2 style.py
- Lange Linien mit morphologischem Opening finden (horizontaler Kernel ~ 1/3 Bildbreite,
  vertikaler Kernel ~ 1/4 Tabellenhöhe). Brüche bei T3 vorher per Closing überbrücken.
- Innere vertikale Linien vorhanden → `grid`; nur horizontale → `hlines`; keine → `borderless`.
- Auf dem Validierungsset die Treffsicherheit messen (hier darf `manifest.json` zur **Auswertung**
  benutzt werden, nicht in der Pipeline). Ziel: > 98 %.

### 7.3 structure.py
Ergebnis: `Grid(row_bounds: list[int], col_bounds: list[int])` im korrigierten Bild.
- **Linien entfernen** aus dem Binärbild, bevor Text analysiert wird.
- **`grid`:** Zeilen- und Spaltengrenzen = Positionen der erkannten Linien (nah beieinander
  liegende Linien zusammenfassen).
- **Zeilen bei `hlines`/`borderless`:** horizontales Projektionsprofil der Textmaske → Lücken
  zwischen Textzeilen = Zeilengrenzen.
- **Spalten bei `hlines`/`borderless`:** vertikales Projektionsprofil **nur über die Body-Zeilen**
  (Kopfzeilen mit spannenden Labels würden Lücken überdecken). Breite Lücken = Spaltengrenzen.
  Mindestbreite der Lücke aus der Schriftgröße ableiten, nicht fest verdrahten.
- **Plausibilität nutzen:** 4–7 Spalten, 7–14 Zeilen gesamt. Liegt das Ergebnis außerhalb, die
  Schwellwerte schrittweise anpassen und neu versuchen (Lücken nach Breite sortieren, die
  breitesten nehmen, bis die Spaltenzahl plausibel ist).

### 7.4 spans.py
- **Header-Zeilen:** bei `hlines` liegt die Linie unter dem Header → Anzahl Header-Zeilen direkt
  ablesbar. Sonst: zwei Header-Zeilen, wenn in Zeile 0 ein Textblock über mehrere Spaltengrenzen
  reicht.
- **`grid`:** Zwei Nachbarzellen sind verbunden, wenn das Liniensegment zwischen ihnen fehlt
  (Anteil schwarzer Pixel auf der Grenze innerhalb des Zellbereichs messen).
- **`hlines`/`borderless`:** Ein Textblock, dessen Ausdehnung eine Spaltengrenze deutlich
  überquert, spannt diese Spalten (`cs`). Ein Header-Label in Spalte 0 mit leerer Zelle darunter →
  `rs=2` (genaue Konvention aus NOTES.md übernehmen).
- **Body-Merges:** nach den Regeln aus NOTES.md (`n/a`, `—`, `see note 3`).
- Ausgabe muss `validate.py` bestehen: jede Rasterposition genau einmal abgedeckt.

### 7.5 ocr.py
- Jede Zelle aus dem **korrigierten Graubild** ausschneiden, Ränder um ein paar Pixel nach innen
  kürzen (Linienreste weg), weißen Rand von ~10 px anfügen, hochskalieren, bis Großbuchstaben
  ~30–40 px hoch sind.
- **Leer-Erkennung vor OCR:** Anteil dunkler Pixel im Ausschnitt unter Schwellwert → `""`, kein
  Tesseract-Aufruf (spart Zeit, verhindert erfundenen Text aus Krümeln).
- Tesseract: `pytesseract.image_to_data(img, lang="eng+deu", config="--oem 1 --psm 7 -c preserve_interword_spaces=1")`.
  Wörter und Confidence zurückgeben.
- **Zweiter Durchgang für Zahlenzellen:** Sieht der erste Text nach Zahl aus, nochmals mit
  `-c tessedit_char_whitelist=<erlaubte Zeichen aus NOTES.md>` lesen und das Ergebnis mit höherer
  Confidence nehmen.
- Mehrzeilige Zellen (falls im Generator möglich): `--psm 6`.
- Parallelisierung über Tabellen (nicht über Zellen): `multiprocessing.Pool`, Umgebungsvariable
  `OMP_THREAD_LIMIT=1`, damit Tesseract-Prozesse sich nicht gegenseitig bremsen.

### 7.6 postprocess.py
- **Zahlenformat pro Tabelle per Mehrheitsentscheid** bestimmen (Englisch / Deutsch / Integer mit
  Komma / Integer mit Leerzeichen; Minus vs. Klammern). Dann unsichere Zellen in dieses Format
  bringen (z. B. `1,234,56` → `1.234,56` bei deutschem Format; fehlendes Komma vor den letzten zwei
  Ziffern ergänzen, wenn alle sicheren Zellen zwei Nachkommastellen haben).
- Typische OCR-Verwechslungen in Zahlenzellen reparieren: `O→0`, `l/I→1`, `S→5`, `B→8`.
- **Labels und Spaltenköpfe** gegen das Vokabular aus `lexicon.py` fuzzy matchen (`rapidfuzz`),
  nur ersetzen, wenn Ähnlichkeit über Schwellwert. Das ist für QA entscheidend: Zeilen werden über
  die exakte erste Spalte gefunden, Spalten über die Kopfzeilen.
- Exakte Zeichen aus NOTES.md erzwingen (richtiges Minus, richtiger Gedankenstrich, Einheiten).

## 8. Robustheit und Zeit

- Pro Tabelle ein Zeitbudget (z. B. 10 s). Ist es überschritten, teure Zusatzdurchgänge
  (zweiter OCR-Durchgang) überspringen.
- **Fallback-Kette:** volle Pipeline → bei Exception oder unplausiblem Raster die Logik der
  Tesseract-Baseline (`baselines/baseline_tesseract.py`) → im schlimmsten Fall `[]`. Immer eine
  gültige JSON-Datei schreiben.
- JSON mit `json.dump(cells, f, ensure_ascii=False)` schreiben, nie von Hand zusammenbauen.
- Jede Zelle **inklusive leerer Zellen** ausgeben.

## 9. Docker

- Basis-Image und Tesseract-Version des Templates beibehalten (Debian, Tesseract 5.5.0).
- Zusätzlich installieren: `tesseract-ocr-deu` (deutsche Labels, Umlaute wie `Mär`) und die
  Python-Abhängigkeiten. `opencv-python-headless` statt `opencv-python` (keine GUI-Bibliotheken
  nötig).
- `COPY pipeline/ /app/pipeline/` – Build-Kontext ist das Repo-Root. Nichts aus `data/`.
- `ENV OMP_THREAD_LIMIT=1`.
- Test genau wie offiziell (README §7):

```
python flatten_dataset.py data/dev flat
docker build -t my-tablesalt -f submission_template/Dockerfile .
docker run --rm --network none -v "$PWD/flat:/data/images:ro" -v "$PWD/out:/data/predictions" my-tablesalt
python score_all.py --gt data/dev --pred out
```

Der Container-Score muss dem lokalen Score entsprechen (± 0.01). Laufzeit messen und in
RESULTS.md eintragen.

## 10. Meilensteine (in dieser Reihenfolge)

| # | Ziel | Fertig, wenn |
|---|---|---|
| M0 | Generator/Scorer verstanden | `docs/NOTES.md` vollständig, Baseline-Score notiert |
| M1 | Durchgehendes Gerüst | `pipeline.run` erzeugt gültiges JSON für alle Bilder (darf anfangs die Baseline-Logik nutzen), `eval_local.py` funktioniert |
| M2 | Preprocessing | Debug-Bilder von T2/T3 sind gerade und schattenfrei |
| M3 | `grid`-Struktur | structure-only TEDS auf `grid` deutlich über Baseline |
| M4 | `hlines` + `borderless` | Spaltenzahl auf Validierungsset in > 90 % korrekt |
| M5 | Header + Spans | Zweizeilige Header und Body-Merges korrekt in validate + TEDS |
| M6 | Zell-OCR | TEDS T0/T1 nahe structure-only TEDS |
| M7 | Postprocessing | QA-Genauigkeit steigt messbar |
| M8 | Docker + Zeit | Container läuft offline, < 8 s/Tabelle, Score = lokal |
| M9 | Fehlerjagd | Schlechteste Tabellen pro Tier analysieren, gezielt verbessern, T3 zuerst |

Nach jedem Meilenstein: Score messen, in RESULTS.md eintragen, kleiner Git-Commit.

## 11. Tests (pytest)

- `validate.py`: Raster-Abdeckung, Spans im Bereich, keine Überlappungen.
- `postprocess.py`: Zahlenformat-Erkennung und -Korrektur mit festen Beispielen
  (`1,234.56`, `1.234,56`, `12 345`, `(1,234.56)`, `-12,345`).
- `structure.py`: kleine selbst gezeichnete Testbilder (OpenCV) mit bekannten Grenzen.
- Ende-zu-Ende: 4 generierte Tabellen (je Tier eine) laufen fehlerfrei durch.

## 12. Arbeitsweise mit dem Team

- Das Team ist neu in Computer Vision. Nach jedem Meilenstein in `docs/LEARNLOG.md` auf Deutsch in
  wenigen Sätzen erklären: was gebaut wurde, warum, welche Fachbegriffe dabei vorkamen.
- Vor großen Architekturänderungen (z. B. anderes OCR, ML-Modell) erst fragen.
- Keine Schwellwerte „nach Gefühl“ ändern – immer mit Messung auf `data/val`.
