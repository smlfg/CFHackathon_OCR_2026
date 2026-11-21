# img2table — research note

**Datum:** 2026-11-21 · **Quelle:** [xavctn/img2table](https://github.com/xavctn/img2table) ·
**Frage:** verdient img2table einen Platz in unserer Pipeline — und wenn ja, wo?
Zusammen mit [PROBLEM.md](../PROBLEM.md) §4/§5, [TATR_RESEARCH.md](../TATR_RESEARCH.md) §6 und
[DECISIONS.md](../DECISIONS.md) #1/#2. Maßstab ist
[../../baselines/baseline_tesseract.py](../../baselines/baseline_tesseract.py): dieselbe Familie, nur
mit besserer Linienerkennung.

---

## 0. Verdikt zuerst

**Heuristischer OpenCV-Wrapper, keine Engine.** Kein gelerntes Modell im Tabellenpfad:
Sauvola-Schwelle → Zeilen/Spalten per Morphologie → Gitter aus Linienschnittpunkten → optional
Whitespace-Pfad für rahmenlose Tabellen → OCR-Wörter per Bounding-Box-Überlappung in Zellen — unser
Baseline-Ablauf, plus echte Liniendetektion und ein Borderless-Pfad.

**Gemessen** (dieser Lauf, Mac-Host, Tesseract 5.5.3, `data/dev` 24, `data/val96` 96): bestes
End-to-End **0.3856 dev / 0.3026 val96** FINAL (TEDSw 0.4124 / 0.3167) gegen Baseline
**0.3417 / 0.3082**. Also +0.044 auf dev, −0.006 auf val96 — nach unserer Regel („gewinnt auf val96")
**kein Gewinn**. Struktur-Decke (Gitter + GT-Text): **0.7188 dev, 0.4793 val96**; TATR erreicht mit
demselben Trick 0.9365, der VLM 0.98. Laufzeit 0.02–0.07 s (Gitter), 0.26–0.38 s (mit OCR).

**Merged Cells:** ja, aber nur geometrisch, nur bei *fehlender Innenkante*, und **nicht im
Zellobjekt** — `TableCell` hat kein `rs`/`cs`. Spans existieren als mehrfach belegte Gitterslots und
werden erst in `ExtractedTable.html` per Largest-Rectangle-Suche rekonstruiert (§3). Exakt in
**5 von 19** Span-Tabellen auf dev.

**Empfehlung:** kein Slot im Submission-Pfad — kein Primärmodell, kein Validator (TATR ist billiger
und besser), keine Fallback-Stufe (42 % leere Tabellen auf val96). Dokumentationswert: eine Zeile.

---

## 1. Claim-Tabelle

| Claim | Confidence | Basis |
|---|---|---|
| Reine OpenCV-Heuristik, kein NN im Tabellenpfad; 2.0.0 ergänzt einen Cython-Metrik-Kern | verifiziert | README, `docs/table-detection-algorithms.md`, `tables/*.pyx` |
| MIT, 2.0.0 vom 2026-05-10, aktiv (903 ★, 119 Forks, 76 Issues, Push 2026-11-12, nicht archiviert) | verifiziert | GitHub-API, PyPI-JSON |
| Default-OCR ist ein **subprocess** auf das `tesseract`-Binary (hOCR) | verifiziert | `ocr/tesseract.py:64-90` |
| 9 OCR-Backends: Tesseract, PaddleOCR, EasyOCR, docTR, Surya, RapidOCR, GCP, AWS, Azure | verifiziert | README, `pyproject.toml` Extras |
| `TableCell` hat **kein** `rowspan`/`colspan`; Spans nur im `html`-Property | verifiziert | `tables/extraction/__init__.py:44-52`, `_utils.py:150-190` |
| L-förmige Merges (rs **und** cs > 1) werden im HTML-Export in Streifen zerlegt | verifiziert | `_utils.py:52-77` |
| `get_content` **löscht leere Zeilen/Spalten**; nur eine Unique-Zelle → Tabelle kollabiert auf 1 Zelle | verifiziert | `tables/types.py:264-296` |
| Tabelle existiert nur, wenn OCR danach ≥2 Zeilen übrig lässt | verifiziert + **gemessen** | `document/_types.py:126-131`; ohne Text 18/24 leer |
| `hlines`-Stil liefert **0 Tabellen**, solange `borderless_tables=False` (`get_cells` braucht beide Richtungen) | **gemessen** | §4 Zeile 2 vs. 3 |
| Struktur-Decke 0.7188 dev / 0.4793 val96; e2e 0.3856 dev / 0.3026 val96 | **gemessen** | §4 |
| 2.0.0 wirft `TypeError`, wenn im erkannten Titelbereich ein hOCR-Wort `value=None` hat (val96 `table_059`) | **gemessen** + verifiziert | Traceback `ocr/_types.py:47` |
| `detect_rotation=True` (Huang 2020, ≤45°) hilft oder schadet auf T1–T3 | **ungemessen** | nur Code/README |

---

## 2. Fakten, hart geprüft

| | |
|---|---|
| Version / Lizenz | **2.0.0** (2026-05-10; Vorgänger 1.4.1 am 2025-02-09 — Rewrite mit Cython-Kern) · **MIT** |
| Python | `>=3.10, <3.15` |
| Kern-Deps | `numpy`, `opencv-contrib-python>=4` (contrib nötig: `cv2.ximgproc.niBlackThreshold`), `pypdfium2>=4.30`, `beautifulsoup4`, `xlsxwriter` |
| Modelle | **keine** im Tabellenpfad; nur optionale OCR-Extras laden Gewichte. Deterministisch: feste Schwellen, kein Sampling |
| Cython / Größe | 4 Cython-Module, Wheels für **manylinux x86_64 cp310–cp314** (Docker ohne Compiler); Paket **7.2 MB**, frisches venv mit Deps **196 MB** (cv2 93, numpy 34, pypdfium2 15, PIL 13) → neu in unserem Container ~110 MB |
| OCR-Backends | Tesseract (Default, externes Binary), PaddleOCR, EasyOCR, docTR, Surya, RapidOCR (ONNX), GCP/AWS/Azure (Netz, irrelevant) |
| PDF | `document.PDF` über `pypdfium2`, 200 DPI; bei nativen PDFs `pdf_text_extraction=True` = **zeichengenauer Text ohne OCR** |
| Offline | Tesseract-Pfad ja (Binary + tessdata im Image); alle anderen Backends laden beim ersten Aufruf Modelle → vendoren |

---

## 3. Algorithmus — und die Span-Frage

```
Bild → threshold_dark_areas (Sauvola) → compute_img_metrics (char_length, line_sep, contours)
     → detect_lines: Smoothing → Canny → Textmasken weg → Morphologie H/V → Längenfilter
     → get_cells(h,v): Rechtecke aus Linienschnittpunkten → get_tables: clustern, Duplikate weg,
       add_semi_bordered_cells (Lücken aus Konturen) → optional Borderless (ARLSA-Whitespace)
     → has_valid_shape(): ≥2 Zeilen/Spalten, ≥4 Zellen, ≤20 % Zellüberlappung
     → get_content(ocr): Wörter mit >50 % Fläche in der Zelle; leere Zeilen/Spalten LÖSCHEN
```

1. **Ein Zellobjekt darf an mehreren Gitterslots hängen.** Fehlt eine Innenkante, wird dieselbe Region
   mehrfach eingetragen — das *ist* der Merged-Cell-Mechanismus. Sichtbare Innenkanten ergeben
   Einzelzellen: Liniengeometrie, kein Raten.
2. **Der Export rät Spans zurück.** `build_html` gruppiert Slots pro Zellobjekt und sucht das größte
   Rechteck (`_find_largest_rectangle`); ein Merge über Zeilen *und* Spalten wird in `html_cell_span`
   **zerlegt** (eine Zeile oder eine Spalte), weil HTML kein L-förmiges `td` kennt — für TEDS eine
   strukturelle Lüge.
3. **Struktur und Text sind gekoppelt.** `get_content` löscht jede Zeile ohne OCR-Text, danach
   greift `max(nb_rows, nb_cols) >= 2`: wenig OCR heißt **Tabelle verschwindet** (`[]`, kein leeres
   Gitter). Ohne Text 18/24 leer, mit OCR 5/24 (dev) bzw. 40/96 (val96).

---

## 4. Messungen

Eigener Lauf, `img2table==2.0.0`, Python 3.14, Mac-Host, System-Tesseract **5.5.3**, `lang="eng"`,
`min_confidence=50`, größte gefundene Tabelle, Ausgabe = `ExtractedTable.html` (inkl. Span-Rekonstruktion),
Score mit `score_all.py`. „Struktur" = Gitter ohne OCR, Zellen mit GT-Text gefüllt (wie TATR §6C).

| # | Konfiguration | Set | FINAL | TEDSw | QA | CellF1 | T0 | T1 | T2 | T3 | s/Tab | leer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0a | **baseline_tesseract.py** | dev | 0.3417 | 0.3549 | 0.2667 | 0.1530 | .539 | .476 | .256 | .323 | 0.20 | – |
| 0b | **baseline_tesseract.py** | val96 | 0.3082 | 0.3082 | 0.3083 | 0.0959 | .483 | .450 | .257 | .232 | – | – |
| 1 | img2table **Default** (psm 11, 1×, kein borderless) | dev | – | – | – | – | – | – | – | – | 0.06 | **22/24** |
| 2 | Struktur, **kein** borderless | dev | 0.1891 | 0.1857 | 0.2083 | 0.2161 | .346 | .348 | .032 | .180 | 0.015 | 18/24 |
| 3 | Struktur, borderless, **1.5×** | dev | **0.7188** | 0.7133 | 0.7500 | 0.8222 | .914 | .868 | .636 | .644 | 0.043 | 2/24 |
| 4 | Struktur, borderless, 1.5× | val96 | **0.4793** | 0.4650 | 0.5604 | 0.6333 | .868 | .750 | .423 | .253 | 0.051 | 26/96 |
| 5 | **e2e** 2×, psm 11, borderless | dev | **0.3856** | 0.4124 | 0.2333 | 0.3239 | .586 | .505 | .365 | .359 | 0.36 | 5/24 |
| 6 | e2e 2×, psm 3, borderless | dev | 0.3097 | 0.3247 | 0.2250 | 0.2597 | .656 | .519 | .172 | .259 | 0.38 | 7/24 |
| 7 | e2e 2×, psm 11, borderless | val96 | **0.3026** | 0.3167 | 0.2229 | 0.1731 | .546 | .498 | .219 | .242 | 0.34 | **40/96** +1 Crash |
| — | *TATR Struktur + GT-Text (Referenz)* | dev | 0.9365 | 0.9365 | – | – | .913 | .931 | .936 | .946 | 0.08 | – |
| — | *VLM MinerU + PP (Referenz)* | val96 | 0.9808 | 0.9808 | 0.960 | – | .995 | .983 | .975 | .989 | 2.33 | – |

Zeile 3: **14/24 exakte Shape** (grid 5/8, hlines 6/8, borderless 3/8), **5/19 exakte Span-Menge**.
Zeile 5/7: nur **7/24** exakte Shape — der Rest sind verschobene Zeilen.

1. **Ohne `borderless_tables=True` unbrauchbar** (Zeile 1/2): `hlines` hat keine vertikalen Linien,
   `get_cells` liefert nichts → Default-Aufruf ist auf 2 von 3 Stilen ein garantierter Nullpunkt.
2. **Struktur ist der Engpass:** mit perfektem Text 0.71 (dev) / 0.48 (val96), TATR 0.9365. **T2/T3
   tragen 0.7 des Scores und sind am schwächsten** (0.636/0.644 dev, 0.423/0.253 val96): die
   Fax-Binarisierung zerstört die dünnen Linien, die der einzige Strukturträger sind.
3. **Auf val96 kippt es unter den Baseline** (0.3026 vs 0.3082) bei 42 % leeren Tabellen; dev war der
   freundliche Fall (24 Tabellen, 6 pro Tier = Rauschen).

---

## 5. Wo es an den drei Achsen scheitert

| Achse | Urteil |
|---|---|
| **Degradation T1/T2** | Schwelle + Morphologie ohne Rotationsmodell; `Line.reprocess()` biegt nur ±5° gerade, eine 3°-Linie zerfällt in Fragmente. T2 dev 0.636 / val96 0.423 (Struktur). `detect_rotation` **ungemessen** |
| **Degradation T3 (Fax)** | Schwächste Stelle: val96 T3 **0.253** Struktur, 0.242 e2e. Bei TATR war T3 per Median+NMS auf 0.946 patchbar — hier fehlt die gelernte Struktur, die man patchen könnte |
| **Struktur — Borderless** | eigener Pfad existiert, aber schwächster Stil: **3/8** exakte Shape; `borderless_tables=True` erzeugt mehrere Kandidaten (dev `table_006`: 3) → größte raten |
| **Struktur — Merges** | fehlende Innenkanten → mehrfach belegte Slots. Erkennt sie, aber kein `rs`/`cs` im Output, Largest-Rectangle-Rekonstruktion, L-Formen zerlegt. 5/19 exakt |
| **Struktur — leere Zellen** | jede Gitterzelle wird als `<td>` geschrieben (auch leer) — formal korrekt; **aber** leere Zeilen/Spalten werden vorher gelöscht → Zeilenshift = TEDS-Katastrophe |
| **Content** | Wort muss mit >50 % Fläche in der Zelle liegen, `confidence ≥ 50`: Randüberstands-Wörter (lange Labels, zentrierte Header) fallen raus. `x_wconf \d{1,2}` schneidet `100` auf `10` ab (`ocr/tesseract.py:118`) — im Code verifiziert, auf unserem Material nicht beobachtet |

---

## 6. Vergleich mit `baselines/baseline_tesseract.py`

**Es ist unser Baseline mit besserer Linienerkennung.** Dieselbe Architektur: Tesseract-Wörter,
Zeilen/Spalten aus Geometrie, Zellen über Position, HTML raus, keine Spans im Export.

| Bauteil | baseline_tesseract.py | img2table |
|---|---|---|
| Zeilen/Spalten | Wort-y-Clustering + x-Lücken-Histogramm (18/40 px) | echte Linien (Canny + Morphologie), skalenadaptiv |
| Borderless | implizit — das ist der ganze Baseline | eigener Pfad, opt-in, mit Struktur-Score und Ablehnung |
| Merges | keine, per Konstruktion falsch | aus fehlenden Innenkanten, Export per Rechteck-Suche |

**Ertrag:** dev +0.044 FINAL, val96 **−0.006**. Der CellF1-Gewinn (0.173 vs 0.096) zeigt, dass das
Gitter besser ist; QA und TEDS fressen ihn durch die gelöschten Zeilen auf. „img2table statt Baseline"
heißt also: derselbe Fehlerpfad, +110 MB Dependencies, 0.3 s statt 0.2 s pro Tabelle und ein Paket,
das auf 42 % von val96 leer zurückkommt.

---

## 7. Integrationsskizze (falls doch jemand will)

```python
from img2table.document import Image
from img2table.ocr import TesseractOCR

def predict(img_path):
    pil = upscale(PILImage.open(img_path).convert("RGB"), 2.0)   # 500–1110 px ist zu klein
    try:
        tables = Image(png_bytes(pil)).extract_tables(
            ocr=TesseractOCR(lang="deu", psm=11),
            borderless_tables=True,       # sonst 0 Tabellen auf hlines/borderless
            implicit_rows=False, implicit_columns=False, min_confidence=50)
    except Exception:                     # 2.0.0 wirft TypeError (Titelbereich + None-Wort)
        return fallback_cells(img_path)
    if not tables:                        # 21 % dev, 42 % val96
        return fallback_cells(img_path)
    best = max(tables, key=lambda t: (t.bbox.x2 - t.bbox.x1) * (t.bbox.y2 - t.bbox.y1))
    return html_to_cells(best.html)       # Spans gibt es NUR hier
```

- **s/Table:** 0.02–0.07 (Gitter), 0.26–0.38 mit Tesseract auf dem Mac; `linux/amd64` mit einem
  Kern eher 0.4–0.8. Laufzeit ist nicht das Problem.
- **Docker:** manylinux-`cp312`-Wheel (kein Compiler), `tesseract-ocr` + `tesseract-ocr-deu` (im
  Template vorhanden), ~110 MB cv2/pypdfium2 zusätzlich.
- **Fallback:** muss zwingend außen herum — zwei Abbruchgründe (`[]` und Exception) sind gemessen.
  Als *unterste* Sprosse unter „Tesseract pro Zelle" vertretbar, sonst nicht. **Ungeprüft:**
  `detect_rotation=True` auf T1–T3 (naives Deskew kostet bei TATR −0.21 TEDS) und die anderen
  OCR-Backends — dichterer Text könnte die 42 % leeren Tabellen senken, das wäre die einzige Zahl,
  die die Bewertung noch drehen könnte.

---

## 8. Quellen (abgerufen 2026-11-21)

- Repo, README (Features, OCR-Backends, `extract_tables`-Parameter, Caveats „white/light background",
  „no CNN") und Repo-Metadaten (903 ★, 119 Forks, 76 Issues, `archived: false`, Push 2026-11-12, MIT) —
  https://github.com/xavctn/img2table · https://api.github.com/repos/xavctn/img2table
- PyPI (2.0.0 vom 2026-05-10, `requires_python >=3.10,<3.15`, `requires_dist`, 31 Wheels inkl.
  manylinux x86_64 cp310–cp314, `license_expression: MIT`) — https://pypi.org/pypi/img2table/json
- Algorithmus-Doku (Thresholding, Line Detection, Bordered, Borderless/ARLSA) —
  https://github.com/xavctn/img2table/blob/main/docs/table-detection-algorithms.md
- Quellcode gelesen 2026-11-21: `tables/extractor.py`, `tables/types.py` (`get_content`,
  Zeilenlöschung), `tables/extraction/__init__.py` (`TableCell` ohne Spans),
  `tables/extraction/_utils.py` (`build_html`, `create_all_rectangles`, `html_cell_span`),
  `ocr/_types.py` (Wort→Zelle >50 %, `_group_words_by_parent`), `ocr/tesseract.py` (hOCR, `x_wconf`),
  `tables/common/threshold.py` (`cv2.ximgproc.niBlackThreshold`), `document/_types.py`, `pyproject.toml`
- Eigene Messungen: `score_all.py` auf `data/dev` (24) und `data/val96` (96), Host-Tesseract 5.5.3,
  Vergleichslauf `baselines/baseline_tesseract.py`; Referenzzahlen aus
  [../TATR_RESEARCH.md](../TATR_RESEARCH.md) §6 und [../MEASUREMENTS.md](../MEASUREMENTS.md)
