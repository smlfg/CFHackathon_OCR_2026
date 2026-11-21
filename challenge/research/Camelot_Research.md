# Camelot (camelot-py) — research note

**Datum:** 2026-11-21 · **Quelle:** [camelot-dev/camelot](https://github.com/camelot-dev/camelot) ·
**Frage:** verdient Camelot einen Platz in unserer Pipeline — und wenn ja, wo?

Zusammen mit [PROBLEM.md](../PROBLEM.md) §4 (Eingabe-Vertrag), [TATR_RESEARCH.md](../TATR_RESEARCH.md)
§6 (was Struktur ohne Text wert ist) und [DECISIONS.md](../DECISIONS.md) §Offene Entscheidungen #1/#2.

---

## 0. Verdikt zuerst

**Camelot ist eine PDF-Bibliothek. Unser Eingabe-Vertrag sind Bilder.** Das ist nicht ein Detail, das
ist der ganze Punkt:

- `lattice`, `stream`, `network`, `hybrid` liefern auf einem PDF **ohne Textlayer gar nichts** — nicht
  ein leeres Gitter, sondern `[]`. Verifiziert im Code: `BaseParser._document_has_no_text()` gibt `True`
  zurück, sobald `horizontal_text` leer ist, und `extract_tables()` bricht davor ab (`base.py:126-147`,
  `base.py:276`). Die Warnung lautet wörtlich *„camelot only works on text-based pages"*.
- Nur `flavor="ml"` (neu in 2.0, opt-in) überschreibt dieses Gate und OCRt die gerenderte Seite.
  Dieser Pfad ist aber **TATR-Struktur + RapidOCR-Text, verpackt in PDF** — also genau die Kombination,
  die wir am 2026-11-20 schon gemessen haben: Struktur-Decke 0.9365, mit Tesseract-Text **0.6608 final**.
  Der VLM liegt bei 0.9808. Camelot fügt dieser Messung nichts hinzu außer einer weiteren Abhängigkeitskette.

| Variante | Ergebnis auf unserem Input | Aufwand |
|---|---|---|
| `lattice` / `stream` / `network` / `hybrid` auf Bild→PDF | **0 Tabellen**, Warnung, TEDS 0 | Wrapper bauen für ein garantiertes Null-Ergebnis |
| `flavor="ml"` + `[ocr]` auf Bild→PDF | funktioniert, ist aber TATR+RapidOCR — Größenordnung 0.66 final, nicht 0.97 | img2pdf + PyTorch/transformers/timm + ONNX-OCR + HF-Modelle vendoren |
| `lattice`/`stream` auf einem **echten textbasierten PDF** (Hidden-Set-Fall aus PROBLEM §8) | exakter Text, keine OCR-Fehler, Zehntel einer Sekunde | ~30 Zeilen Branch hinter einem Textlayer-Check |

**Empfehlung:** kein Submission-Pfad. Ein Restwert bleibt: ein billiger PDF-Branch für den einen
Hidden-Input, der laut Transcript ein echtes PDF sein könnte — als Versicherung, nicht als Modellentscheidung.

---

## 1. Claim-Tabelle

| Claim | Confidence | Basis |
|---|---|---|
| Camelot 2.0.0 ist aktuell, MIT, aktiv gepflegt (letzter Push 2026-11-20, nicht archiviert) | verifiziert | GitHub-API, PyPI-JSON |
| Ghostscript ist **keine** Pflichtabhängigkeit mehr — seit 1.0.0 ist `pdfium` der Default-Backend, Ghostscript nur Extra | verifiziert | README, Docs „alternative image conversion backends", PyPI `requires_dist` |
| Die vier klassischen Flavors brauchen einen PDF-Textlayer; ohne ihn: 0 Tabellen | verifiziert | `parsers/base.py:126-147`, `:276` |
| `flavor="ml"` ist der einzige Pfad, der ohne Textlayer überhaupt Text produziert (TATR + RapidOCR) | verifiziert | `parsers/ml.py`, CHANGELOG 2.0.0, Docs how-it-works |
| Die offizielle FAQ sagt weiterhin „Does Camelot work with image-based PDFs? **No**" — Widerspruch zu 2.0 | verifiziert | Docs FAQ vs. README/CHANGELOG |
| Camelot kann `rowspan`/`colspan` **nicht exportieren**; `to_html` ist `df.to_html()` über eine flache Textmatrix | verifiziert | `core.py:1015-1032`, `Table.data` `core.py:674` |
| Span-Information existiert intern als Kantenflags (`Cell.hspan/vspan`) und wäre mit eigenem Exporter rekonstruierbar | verifiziert (Code), Aufwand geschätzt | `core.py:503-512`, `lattice.py` |
| Auf einem textbasierten PDF ist der Text **exakt** (keine OCR, keine Zeichenfehler) | verifiziert | Pipeline liest PDF-Textoperatoren über `playa` |
| Auf unserem Bild-Input ist Camelot schlechter als der VLM, weil es dieselbe Struktur-Decke mit schwächerem Text kombiniert | **abgeleitet, nicht gemessen** | TATR §6 + ML-Backend-Beschreibung |
| Skew/Perspektive brechen Lattice, weil die Linienerkennung reine OpenCV-Morphologie ohne Rotationsmodell ist | plausibel, **ungemessen** | how-it-works (Erosion/Dilation), kein Deskew-Schritt im Code |
| Fax-Rauschen erzeugt doppelte Zeilen/Spalten wie bei unserem TATR-Lauf (T3 0.083) | plausibel, **ungemessen** — Camelot ist nicht unser TATR-Skript | TATR §6 A vs. C |

---

## 2. Fakten, hart geprüft

| | |
|---|---|
| Version | **2.0.0**, Release 2026-06-04 (vorher 1.0.9, 2025-08-10) |
| Lizenz | **MIT** (Repo + PyPI-Metadaten) |
| Status | aktiv: 3.858 Stars, 549 Forks, 58 offene Issues, letzter Push **2026-11-20**, `archived: false` |
| Python | `>=3.10` (3.9 in 2.0 gedroppt) |
| Kern-Dependencies | `click`, `numpy`, `openpyxl`, `pandas`, `tabulate`, `opencv-python-headless`, `pypdfium2`, `pillow`, **`playa-pdf`** (ersetzt pdfminer/pypdf in 2.0) |
| Extras | `[plot]` matplotlib · `[ml]` `torch`, `transformers<5`, `timm` · `[ocr]` `rapidocr-onnxruntime` · `[ghostscript]` |
| Ghostscript | **optional**. Default-Renderer ist `pdfium` (gebündeltes Wheel, keine Systemabhängigkeit). Die Annahme „Camelot braucht Ghostscript" gilt nur für Versionen < 1.0 |
| Textlayer nötig? | ja für `lattice`/`stream`/`network`/`hybrid`; nein für `flavor="ml"` |
| OCR eingebaut? | nur im `ml`-Backend, via **RapidOCR (ONNX)**, nicht Tesseract. `ocr="auto"` = OCR nur, wenn die Seite keinen Textlayer hat |
| Scanned PDF ohne Textlayer | klassische Flavors: **0 Tabellen + Warnung**. `ml`: Struktur aus dem Bild, Text aus OCR |
| Modelle im ml-Backend | `microsoft/table-transformer-detection` + `microsoft/table-transformer-structure-recognition-v1.1-all`, `from_pretrained` zur Laufzeit |
| Output | `Table.df` (pandas), `to_csv/json/excel/html/markdown/sqlite` — **immer flach, ohne Spans** |

Der Ghostscript-Punkt ist die eine Korrektur an der Aufgabenstellung: die Abhängigkeit ist real, aber
seit 1.0 Hintergrundrauschen. Das Problem ist ein anderes, nämlich der fehlende Textlayer.

---

## 3. Algorithmen — was jeder wirklich tut

| Flavor | Eingabe, die er braucht | Was er zurückgibt | Spans? |
|---|---|---|---|
| **lattice** | gerenderte Seite + **Linien** (Vektor oder Raster) | Gitter aus Liniensegmenten/Schnittpunkten, Zellen = Zeile ∩ Spalte | erkennt Spans geometrisch, **exportiert sie nicht** |
| **stream** | nur Text-Bounding-Boxes | Zeilen aus y-Überlappung, Spalten aus dem **Modus der Wörter pro Zeile**, dann x-Bereiche | nein |
| **network** | nur Text-Bounding-Boxes | Alignment-Netzwerk, Tabellen wachsen vom Seed-Box nach außen, Header separat | nein |
| **hybrid** | beides | network-Ergebnis, Zeilen/Spalten-Grenzen von lattice verfeinert | nein |
| **ml** | gerenderte Seite (Text optional) | TATR-Boxen → dieselbe Gitterrepräsentation, Text aus Textlayer oder OCR | ja intern (`apply_spans`), nein im Export |

Der entscheidende Satz steht in der Doku selbst: *„the model supplies the structure, the page supplies
the text"* — genau unsere Architektur, nur dass wir den Text nicht aus einem PDF-Textlayer ziehen können.

**Spans, konkret:** `Table.data` baut eine 2D-Liste aus `cell.text` (`core.py:674-679`), `to_html` ruft
`self.df.to_html()` (`core.py:1032`). In einer Zelle mit offener Innenkante wird der Text per
`shift_text=['l','t']` in die Ankerzelle geschoben, optional per `copy_text` dupliziert. Für TEDS ist
das fatal: ein zweizeiliger Header mit `rowspan=2` wird zu „Zeile 0 hat alle Labels, Zeile 1 ist leer".
Die Wahrheit steckt in `Cell.hspan/vspan` (`not left or not right`), also in den Kantenflags — ein
eigener Exporter könnte `rs`/`cs` daraus rechnen. Das ist ein von uns zu schreibendes und zu testendes
Bauteil, kein Camelot-Feature.

---

## 4. Wo es an unseren drei Achsen scheitert

| Achse | Was Camelot sieht | Bewertung |
|---|---|---|
| **Degradation** | nur, was den Render überlebt. Kein Deskew, kein Perspektivmodell, keine Rauschbehandlung | Lattice ist reine OpenCV-Morphologie mit festen Kernen; schräge Linien zerfallen in Fragmente. Unser eigener TATR-Lauf hat gezeigt, dass selbst ein gelernter Detektor bei Skew kippt — ein Morphologie-Kernel kippt früher. **Ungemessen**, aber die Mechanik ist eindeutig |
| **Degradation, T3** | 1-bit-Fax, hochskaliert auf 300 dpi | Unser T3-Defekt war *doppelte* Zeilen/Spalten auf Sensorrauschen (0.083 → 0.946 mit Median+NMS). Ob Camelots `ml`-Grid solche Duplikate zusammenfasst, haben wir **nicht** geprüft — es ist genau die Stelle, an der unser eigener TATR-Lauf ohne Patch versagt hat |
| **Struktur** | Linien (lattice) oder Wortabstände (stream) | Lattice ist auf unserem `grid`/`hlines`-Stil im Vorteil und deterministisch. Auf `borderless` fällt es komplett aus; `stream` rät Spalten über den **Modus der Wörter pro Zeile**, was bei leeren Zellen und zweizeiligen Headern systematisch daneben liegt. Spans werden erkannt, aber nicht ausgegeben |
| **Content** | bei textbasiertem PDF: die Zeichen selbst | Die einzige echte Stärke: **null OCR-Fehler**, `79.036,73 %` kommt exakt so heraus wie gedruckt. Bei Bild-Input ist dieser Vorteil weg — dann liest RapidOCR, und deutsche Zahlformate sind für RapidOCR ungemessen |

Kurz: kein Vorsprung auf irgendeiner unserer Achsen, weil Camelot die Achse „Bild" gar nicht betritt.
Auf Text-PDFs ist es stark — nur ist das nicht der Benchmark.

---

## 5. Integrationsskizze (und was daran bricht)

```python
import img2pdf, camelot

pdf_bytes = img2pdf.convert("/data/images/t0001.png")     # verlustfrei, kein Re-Encode
tables = camelot.read_pdf(pdf_bytes, flavor="ml", ocr="auto", resolution=300)
# tables[0].cells -> eigene rs/cs-Rekonstruktion; tables[0].df ist flach
```

`read_pdf` akzeptiert seit 2.0 auch `bytes` — der PDF-Zwischenschritt bleibt also im Speicher.
`img2pdf` ist **LGPLv3**, 0.6.3, hängt an Pillow + pikepdf, und bettet JPEG/PNG byte-identisch ein
(kein Qualitätsverlust, ~500–700 Byte Overhead). Das ist der billigste Teil des Ganzen.

Was bricht:

1. **`flavor="lattice"` etc. → `[]`.** Kein Textlayer, kein Ergebnis. Der Wrapper wäre umsonst.
2. **`flavor="ml"` funktioniert, ist aber ein Neuaufguss.** TATR-Struktur + RapidOCR-Text. Unsere
   Struktur-Decke ist 0.9365 (oracle text), mit Tesseract 0.6608 final; RapidOCR ist ein anderer
   OCR-Stack, also kein direkter Vergleich — aber dieselbe Decke, ungetesteter Textpfad. VLM: 0.9808.
3. **Spans fehlen im Output.** Wir müssten einen eigenen Exporter gegen `table.cells` schreiben und
   validieren. Zusätzliches Bauteil ohne Benchmark-Beleg.
4. **Offline-Gate und zweiter Modellstack.** Die TATR-Checkpoints kommen per `from_pretrained` von
   HuggingFace — im Container vendored und gepinnt. Dazu `torch` + `transformers` + `timm` + ONNX-OCR,
   für einen Pfad, den `scripts/tatr_experiment.py` in 0.08–0.17 s/Table (MPS) schon abdeckt.
5. **Laufzeit:** 300-dpi-Render + Detektion + Struktur + OCR pro Tabelle; die Doku nennt ~1 s/Seite
   mit GPU. Kein 15-s-Ausschluss, aber Aufwand ohne Ertrag.

---

## 6. Der eine Fall, in dem Camelot sinnvoll ist

PROBLEM §8 sagt: mindestens ein Hidden-Input könnte ein **echtes PDF** sein. Wenn dieses PDF einen
Textlayer hat, ist Camelot der billigste korrekte Pfad, den wir haben:

- Text kommt zeichengenau aus dem PDF, keine OCR-Fehler, kein GPU-Bedarf; `lattice` für liniierte,
  `stream`/`hybrid` für titellose Tabellen, `flavor="auto"` wählt pro Seite. Zehntel Sekunden.

Der Branch gehört hinter einen billigen Test (`pdftotext`/`playa` liefert Zeichen?) und in die
Fallback-Leiter aus DECISIONS #7. Confidence: **mittel**. Messbar ist er auf `data/dev` nicht, weil
dev ausschließlich Bilder enthält — er wäre eine unbelegte Versicherung, kein Score-Gewinn.

---

## 7. Was wir nicht geprüft haben

- Keine eigene Messung auf `data/dev` — es gibt keinen Grund, einen Tag in einen Pfad zu stecken,
  dessen Null-Ergebnis im Code steht.
- Ob Camelots `ml`-Grid Zeilen/Spalten-Duplikate auf Faxrauschen unterdrückt (der Patch, der bei uns
  0.083 → 0.946 brachte), und wie RapidOCR deutsche Zahlformate (`1.234,56`, `-x`, `(x)`) liest.
- Ob `flavor="ml"` bei 300-dpi-Upscaling eines 500–1110-px-Bildes andere TATR-Eingaben sieht als unser
  `MaxResize(1000)`-Pfad, und ob die offizielle FAQ („No") oder README/CHANGELOG („ja, mit `[ml,ocr]`")
  den aktuellen Stand beschreibt. Der Code sagt: ml funktioniert, alles andere nicht.

---

## 8. Quellen (abgerufen 2026-11-21)

- Repo, README, MIT, „Five parsers", Ghostscript-Hinweis — https://github.com/camelot-dev/camelot
- Repo-Metadaten (Stars, Issues, `archived: false`, `pushed_at` 2026-11-20, Lizenz MIT) — https://api.github.com/repos/camelot-dev/camelot
- PyPI-Metadaten: 2.0.0, `requires_python >=3.10`, `requires_dist`, Extras `ml`/`ocr`/`ghostscript` — https://pypi.org/pypi/camelot-py/json
- CHANGELOG 2.0.0 (2026-06-04): ml-Backend, OCR, playa-pdf, `engine="combined"` als Default — https://raw.githubusercontent.com/camelot-dev/camelot/master/CHANGELOG.md
- Docs „How it works": Stream/Lattice/Network/Hybrid/ML, FinTabNet-Zahlen (ml TEDS 0.371 vs. network 0.200) — https://camelot-py.readthedocs.io/en/latest/user/how-it-works.html
- Docs „Advanced usage": Abschnitt „Working with image-based / scanned PDFs", Backend-Wahl, `erode_iterations`, `line_scale` — https://camelot-py.readthedocs.io/en/latest/user/advanced.html
- Docs FAQ (widersprüchlich zu 2.0: „Does Camelot work with image-based PDFs? No") — https://camelot-py.readthedocs.io/en/latest/user/faq.html
- Quellcode: `camelot/parsers/base.py` (`_document_has_no_text`, `extract_tables`-Gate), `camelot/core.py` (`Table.data`, `to_html`, `Cell.hspan/vspan`), `camelot/parsers/ml.py` (TATR-Checkpoints, `ocr="auto"`, `apply_spans`) — gelesen 2026-11-21
- img2pdf 0.6.3, LGPLv3, Pillow + pikepdf, verlustfreie Einbettung — https://pypi.org/pypi/img2pdf/json
- Eigene Zahlen: [../TATR_RESEARCH.md](../TATR_RESEARCH.md) §6, [../MEASUREMENTS.md](../MEASUREMENTS.md)
