# Docling / TableFormer (IBM) — Recherchenotiz

**Datum:** 2026-11-21 · **Quelle:** [docling-project/docling](https://github.com/docling-project/docling) ·
**Frage:** Verdient Doclings Tabellen-Pfad (TableFormer) einen Platz in unserer Pipeline — und welchen?

Gelesen zusammen mit [../PROBLEM.md](../PROBLEM.md) §6/§10, [../TATR_RESEARCH.md](../TATR_RESEARCH.md)
und [../DECISIONS.md](../DECISIONS.md) offene Entscheidung #1. Geprüfter Stand: `docling` **2.151.1**
(PyPI-Upload 2026-11-19), `docling-ibm-models` **4.0.4**, **TableFormer V1 weiter Default**, V2 ist
Preview. Jede Zahl hier ist an dieses Datum gebunden.

---

## 0. Verdict zuerst

**Rolle: Struktur-Modell — und auch da nur als Messkandidat, nicht als Submission-Weg. Als
Gesamtpipeline verliert es (Text = fremde OCR), und es ist kein PDF-only-Tool: es liest Bilder
problemlos, schiebt aber ein Layout-Modell davor, das auf Fotos/Faxe nicht trainiert ist.**

Der Grund ist derselbe wie bei TATR und ist gemessen: Doclings Tabellenpfad ist **Struktur + fremde
OCR**. Auf einem Foto gibt es keine Textebene, also läuft `do_cell_matching` gegen OCR-Wörter. Damit
landet Docling genau in dem Loch, das wir am 2026-11-20 ausgemessen haben: Struktur 0.9365 (oracle
text) → **0.7289 / FINAL 0.6608** mit Tesseract, gegen **0.9862** unseres VLM. Docling tauscht nur den
Struktur-Prädiktor; den Textengpass erbt es unverändert.

| Claim | Confidence | Basis |
|---|---|---|
| Docling akzeptiert Bilder (PNG/JPEG/TIFF/BMP/WEBP/GIF) und fährt sie durch dieselbe `StandardPdfPipeline` wie PDFs — es ist **kein** PDF-only-Tool. | verified | `document_converter.py`, `InputFormat.IMAGE: ImageFormatOption()`; `docs/usage/supported_formats.md` |
| Auf Bildern ist der Textweg **OCR**, nicht Modell: TableFormer liefert OTSL-Struktur + eine Box pro Zelle, den Text holt `do_cell_matching` aus OCR-Wörtern. | verified | Modellkarte TableFormerV2; `table_structure_model.py` (`_words_and_ocr_cells`, `_uncovered`) |
| Als Gesamtpipeline kann Docling unseren VLM **nicht** schlagen; die Obergrenze ist der Tesseract-Text, nicht die Struktur. | high | unsere Messung TATR §6 D + identische Architektur-Annahme; Docling selbst ungemessen |
| Die **Struktur**-Ausgabe ist vertraglich besser als TATR: `row_span`/`col_span` + Grid-Indizes stehen direkt im JSON, leere Zellen sind per `grid` ableitbar. | verified | `docling_core/types/doc/items/table/table_data.py` |
| Als Struktur-Modell ist TableFormer **schwerer** als TATR (53,2M vs 28,8M Params, 213 MB vs 110 MB, plus torch/opencv/transformers/docling-core). | verified | Safetensors-Header (selbst gezählt), PyPI `requires_dist` |
| Der 15-s-Deckel überlebt: Struktur ist auf CPU ~1 s/Seite, nicht 15. | medium | Docling Tech Report Tab. 1 (0,92 pages/s, Xeon 16 Threads, OCR aus); auf unserer Hardware **nicht gemessen** |
| Für unser Kernproblem (T3-Fax, Fotos) gibt es **keine** veröffentlichte Docling-Zahl. | verified | OmniDocBench-README listet Docling nur als Tool, ohne Leaderboard-Zeile; keine Paper-Eval auf Fotos/Fax |
| Der Docker-/Offline-Pfad ist machbar, aber mit zwei Fallen: Modelle werden per Default **beim ersten Lauf** geladen, und `OcrAutoOptions` wählt **nie** Tesseract. | verified | `docs/usage/advanced_options.md` §Model prefetching; `docs/getting_started/installation.md` |

**Empfehlung:** nicht integrieren. Wenn Struktur je der Engpass wird, ist die Reihenfolge
(a) TATR fine-tunen (wir haben das Modell + Label-Generator), (b) TableFormerV2 als CPU-Vergleich,
(c) Docling gar nicht.

---

## 1. Was der Stack wirklich ist

Drei Pakete, die oft als eines gelesen werden:

1. **`docling`** — Orchestrator: Backends, Layout, Reading Order, Assembler, Export (MD/HTML/JSON).
   Code **MIT**.
2. **`docling-parse`** — PDF-Backend (Textzellen, Bboxen). Für ein Foto liefert
   `ImageDocumentBackend` eine Seite mit einem Bitmap.
3. **`docling-ibm-models`** — Gewichte-Laufzeit: Layout (RT-DETR „Heron"), **TableFormer**, V2,
   Picture-Classifier. Eigenständig installierbar (`torch`, `torchvision`, `transformers`, `opencv`,
   `safetensors`, `huggingface_hub`) — kein `docling` nötig.

Für ein Bild läuft intern:

```
Bild ──ImageDocumentBackend──► eine "Seite"
  └─ Layout (Heron, RT-DETR)  ──► TABLE-Region(en)
       └─ TableFormer (448×448 Crop) ──► OTSL-Tokens + Zellboxen
            └─ do_cell_matching: OCR-Wörter → Zellen
                 └─ TableData{table_cells[], num_rows, num_cols} → JSON/HTML
```

Bruchpunkt ist der **Layout-Schritt**, nicht TableFormer: Heron (RT-DETR) ist auf Dokument-Layout in
DocLayNet-Taxonomie trainiert (Tech Report arXiv 2509.11720) — Fax/Foto ist nicht der Zielkorpus.
Liefert es keine TABLE-Box, gibt es null Zellen. Ausweg: `TableCropsLayoutOptions` (Passthrough, das
ganze Bild ist die Tabelle) — genau unser Fall, siehe `docs/examples/experimental/process_table_crops.py`.

---

## 2. Modell-Fakten (geprüft, nicht zitiert)

| | TableFormer V1 (Default) | TableFormer V2 (Preview) |
|---|---|---|
| Status | `docling_tableformer`, **Default** | `docling_tableformer_v2`, per CLI-Flag, **nicht** Default |
| Architektur | ResNet-18 Backbone + Transformer-Encoder/Decoder + Zell-Bbox-Decoder (CVPR 2022) | EfficientNetV2-S + Depthwise-Spatial-Mixer + cache-aware Transformer-Decoder + Bbox-Head |
| Input | 448×448 | 448×448 |
| Output | OTSL-Token + eine Box pro Zelle | OTSL-Token + eine Box pro Zelle |
| Params | **53,2M** (accurate) / 36,4M (fast) — aus dem Safetensors-Header gezählt | ~52M (Modellkarte) |
| Checkpoint | **212,76 MB** (accurate) / 145,45 MB (fast), fp32 | **209,74 MB** |
| Lizenz Gewichte | HF-Tags `cdla-permissive-2.0` **und** `apache-2.0` (pro Ordner nicht sauber getrennt → vor Shippen prüfen) | `apache-2.0` |
| Lizenz Code | MIT (Repo) | MIT |
| Training | SynthTabNet (synthetisch, „verschiedene Appearance-Styles") + PubTabNet/FinTabNet/TableBank | nicht dokumentiert |
| Gemeldete Qualität | TEDS 95,4 simple / 90,1 complex / **93,6 alle** (IBM-Modellkarte) | keine Zahlen veröffentlicht |

Dazu im gleichen Lauf: Layout **Heron** 171,66 MB (`apache-2.0`), Table-Former-Crops brauchen
`opencv` (sonst `ModuleNotFoundError: cv2`).

**OCR-Engines:** Tesseract (CLI oder `tesserocr`), EasyOCR, RapidOCR (ONNX/OpenVINO/Paddle/Torch),
macOS Vision, SuryaOCR, Nemotron-OCR. Default ist `OcrAutoOptions`.

> **Falle 1 — Auto-OCR wählt nie Tesseract.** In einem Slim-Image, in dem nur `tesseract-ocr` liegt,
> loggt Docling „No OCR engine found" und **konvertiert ohne OCR**. Ergebnis: perfekte Struktur,
> leere Zellen, TEDS ≈ 0. Man muss `TesseractCliOcrOptions(lang=["de","en"])` explizit setzen.
> Zusätzlich sind die Tesseract-Defaultsprachen `eng, spa, fra, deu` — fehlt eine (`tesseract-ocr-deu`
> im Dockerfile!), **schlägt die Konvertierung fehl**, nicht nur die Sprache.

> **Falle 2 — EasyOCR ist auf CPU unbrauchbar.** Der Docling-Technical-Report schreibt es selbst:
> „upwards of 30 seconds per page" auf CPU. Das allein ist das Doppelte unseres 15-s-Budgets pro Tabelle.

**Laufzeit (verifiziert, fremde Hardware):**

| Setting | Wert | Quelle |
|---|---|---|
| Standard-Pipeline, Xeon E5-2690, 16 Threads, **OCR aus**, docling-parse | 0,92 pages/s ≈ **1,09 s/Seite** | Tech Report Tab. 1 |
| dito, pypdfium-Backend / Apple M3 Max 16 Threads | 1,57 / 1,34 pages/s ≈ 0,64 / 0,75 s/Seite | Tech Report Tab. 1 |
| CPU-Docs (16 vCPU, OCR aus) | 1,2 pages/s | `docs/usage/gpu.md` |
| TableFormer allein auf einem Crop | **nicht gemessen**, Schätzung 0,2–0,6 s | 53M Params, ein Encoder-Pass + autoregressive OTSL-Dekodierung |

**Docker / Offline (der unangenehme Teil):**

- `docling-serve` v1.38.2 wiegt als Image **5,04 GB** (Summe der Layer, selbst aus dem Quay-Manifest
  gelesen). Unser Submission-Image ist `python:3.12-slim` + Tesseract, also ~0,2 GB. Ein Slim-Build
  mit `docling-slim[models-local]` liegt in derselben Größenordnung (torch allein dominiert) — nicht
  gemessen, aber der Organisator vergleicht Image-Größe neben dem Score.
- `pip install docling` zieht auf Linux den **CUDA-torch** (Wheel 539,2 MB + `nvidia-*`-Pakete). Ohne
  `--extra-index-url https://download.pytorch.org/whl/cpu` bläht sich das Image um mehrere GB.
- **Modelle werden per Default beim ersten Lauf aus dem Netz geladen** („By default, models are
  downloaded automatically upon first usage"). Im Offline-Container heißt das: erster Convert
  scheitert. Lösung ist sauber und dokumentiert: `docling-tools models download` beim **Build**
  (oder `DOCLING_ARTIFACTS_PATH` / `--artifacts-path` auf ein mitkopiertes Verzeichnis). Der
  Downloader pinnt `revision=` bereits pro Repo — das erfüllt unsere Pin-Anforderung.
- Kaltstart: ~385 MB Gewichte (Layout + TableFormer) müssen von Platte in torch; im 5-Minuten-
  Startbudget unkritisch, aber es ist ein zusätzlicher Faktor, den unser aktueller Pfad nicht hat.

---

## 3. Output-Format: bekommen wir Zellen mit Spans?

**Ja — besser als bei TATR.** `TableCell` im DoclingDocument-JSON trägt direkt:

| Feld | Bedeutung | Unser Contract |
|---|---|---|
| `start_row_offset_idx` / `start_col_offset_idx` | Grid-Position (0-basiert) | `r`, `c` |
| `end_row_offset_idx` / `end_col_offset_idx` | Ende exklusiv | `r+rs`, `c+cs` |
| `row_span` / `col_span` | Spans, Default 1 | `rs`, `cs` |
| `text` | Zellinhalt (aus OCR/Textebene) | `text` |
| `column_header`, `row_header`, `row_section` | semantische Flags | ignorieren |

Damit ist die Abbildung auf unser JSON **eine Schleife ohne Geometrie-Mathematik** — der konkrete
Vorteil gegenüber TATR, wo Zellen aus `row ∩ column` minus Spanning-Subzellen hergeleitet werden
müssen und zwei Kopfzeilen flachgeklopft werden.

Leere Zellen: `table_cells` enthält nur belegte Zellen, aber die computed property `grid` füllt das
komplette `num_rows × num_cols`-Raster mit `TableCell(text="")` auf. Wer `table_cells` naiv
serialisiert, verliert die Leerzellen und damit TEDS — genau unser Fehlerbild `missing-empty-cells`.
Der HTML-Export setzt `rowspan=`/`colspan=` korrekt, ist für uns aber der Umweg.

## 4. Wo es an unseren drei Achsen scheitert

| Achse | Docling/TableFormer | Urteil |
|---|---|---|
| **Degradation (T3 Fax, T2 Foto)** | Keine Veröffentlichung. Heron (Layout) ist auf Dokument-Layout trainiert (DocLayNet-Taxonomie); ein Fax ist out-of-distribution. TableFormer sieht nur den 448×448-Crop und ist linienunabhängig — das spricht für Rauschen, aber **gemessen ist nichts**. | **unbekannt, eher riskant** — genau die Stelle, an der TATR ohne Denoise auf 0,083 einbrach |
| **Struktur (Spans, 2-Zeilen-Header, borderless)** | OTSL modelliert `fcel/ecel/lcel/ucel/xcel` inkl. Spans; Header-Semantik ist nur ein Flag, die **Geometrie** bleibt erhalten (anders als TATR). Borderless ist per Architektur kein Problem (keine Liniendetektion). Zweizeilige Header: plausibel, unbelegt. | **der einzige echte Pluspunkt**, aber ungemessen |
| **Content (deutsche Zahlen, Umlaute)** | Docling normalisiert **nichts**. `1.234,56`, `(2079)`, `–` kommen so heraus, wie die OCR sie liest. Umlaute hängen zu 100 % an der Engine (`deu`-Traineddata). | **kein Beitrag** — unsere `P1`-Dash-Normalisierung und `P0..P5` bleiben Pflicht |
| **QA-Tauglichkeit** | Ein verschobener Span killt QA genauso wie bei uns; Docling hat keine Konsistenzprüfung über Zellen. | kein Vorteil |

Der harte Vergleich in einer Zeile: unser bester Struktur-Beleg ist **0,9365 mit Oracle-Text** und
**0,6608 final mit Tesseract**. Docling müsste Struktur *und* Text gleichzeitig besser machen. Text
macht es nicht — es benutzt dieselbe OCR.

---

## 5. Integrationsskizze (falls wir es doch messen wollen)

```python
# scripts/docling_experiment.py  (Skizze, nicht implementiert)
from docling.document_converter import DocumentConverter, ImageFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions, TesseractCliOcrOptions
from docling.experimental.datamodel.table_crops_layout_options import TableCropsLayoutOptions

opts = ThreadedPdfPipelineOptions(
    layout_options=TableCropsLayoutOptions(),   # Bild IST die Tabelle → Layout überspringen
    do_table_structure=True, do_ocr=True,
    ocr_options=TesseractCliOcrOptions(lang=["de", "en"]),   # nie "auto" lassen
    artifacts_path="/models",                   # offline, beim Build gefüllt
)
conv = DocumentConverter(allowed_formats=[InputFormat.IMAGE],
                         format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)})

td = conv.convert(png).document.tables[0].data
cells = [{"text": c.text, "r": c.start_row_offset_idx, "c": c.start_col_offset_idx,
          "rs": c.row_span, "cs": c.col_span}
         for c in td.table_cells if c.text.strip()]
plus alle leeren Slots aus td.grid mit text="" nachziehen     # sonst TEDS-Verlust
```

**Zeitbudget auf unserer Kiste (16 CPUs, RTX 4070 Laptop):**

| Posten | Schätzung | Status |
|---|---|---|
| Heron-Layout | entfällt mit `TableCropsLayoutOptions` | — |
| TableFormer (53M, 448²) | 0,2–0,6 s | Schätzung, nicht gemessen |
| Tesseract pro Zelle (unser Stand) | **0,20 s** | gemessen, MEASUREMENTS.md |
| Modell-Laden (einmalig) | 1–3 s | Schätzung |
| **Summe pro Tabelle** | **≈ 1–2 s** | **Budget 15 s hält** |

Das ist die gute Nachricht: **Zeit ist nicht das Problem, Text ist es.** Wer Docling wegen Laufzeit
ablehnt, argumentiert falsch; wer es wegen 0,66 final ablehnt, argumentiert mit unserer Messung:
gleiche Strukturklasse wie TATR, viel größeres Image, null Textgewinn.
**Was ich stattdessen tun würde (2 h, wenn überhaupt):** `docling_tableformer_v2` einmal auf `data/dev`
+ `data/val` mit Oracle-Text. Nur wenn V2 > 0,9365 kommt, ist es ein Kandidat für den Struktur-Slot,
den TATR gerade hält — nicht für den Submission-Pfad.

## 6. Quellen (alle abgerufen 2026-11-21)

- Repo, README, MIT, Feature-Liste, Bild-Formate — https://github.com/docling-project/docling
- PyPI `docling` **2.151.1** (Upload 2026-11-19), `docling-ibm-models` **4.0.4** — https://pypi.org/pypi/docling/json · https://pypi.org/pypi/docling-ibm-models/json
- Modellkarten — https://huggingface.co/docling-project/docling-models (TEDS 95,4/90,1/93,6) · https://huggingface.co/docling-project/TableFormerV2 (Preview, ~52M, Apache-2.0) · https://huggingface.co/docling-project/docling-layout-heron
- Gewichte/Größen — HF-Trees der drei Repos; Parameterzahlen aus den Safetensors-Headern gezählt; Lizenzen aus `cardData.license`
- Quellcode (abgerufen 2026-11-21) — `docling/document_converter.py` (ImageFormatOption) · `docling/datamodel/pipeline_options.py` (Defaults) · `docling/models/stages/table_structure/table_structure_model.py` · `docling-core/…/items/table/table_data.py` (Spans/Grid) · `docling-core/…/serializer/html.py`
- Doku — `docs/reference/cli.md`, `docs/usage/supported_formats.md`, `docs/usage/gpu.md`, `docs/usage/advanced_options.md`, `docs/getting_started/installation.md`, `docs/examples/experimental/process_table_crops.py`
- TableFormer-Paper (ResNet-18, 448×448, SynthTabNet, TEDS 98,5/95) — https://arxiv.org/abs/2203.01017
- Docling Technical Report (CPU-Throughput Tab. 1, EasyOCR 30 s/Seite) — https://arxiv.org/abs/2408.09869
- Heron-Layout (Trainingsdaten, RT-DETR) — https://arxiv.org/abs/2509.11720
- Image-Größe `docling-serve` v1.38.2 = 5,04 GB — https://quay.io/api/v1/repository/docling-project/docling-serve
- torch-Wheel 539,2 MB — https://pypi.org/pypi/torch/json
- OmniDocBench (Docling nur als Tool gelistet, keine Leaderboard-Zeile → keine öffentliche TEDS-Zahl) — https://github.com/opendatalab/OmniDocBench
- Unsere Zahlen — [../MEASUREMENTS.md](../MEASUREMENTS.md), [../TATR_RESEARCH.md](../TATR_RESEARCH.md) §6, [../ERRORS.md](../ERRORS.md)
