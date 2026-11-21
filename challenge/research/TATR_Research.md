# Table Transformer (microsoft/table-transformer) — research note

**Retrieved:** 2026-11-21 · **Subject:** TATR / TATR-v1.0 / TATR-v1.1 (Smock & Pesala, Microsoft)
**Question:** verdient TATR einen Platz in unserer Pipeline — und wenn ja, welchen?

Read together with [../TATR_RESEARCH.md](../TATR_RESEARCH.md) (our earlier note, 2026-11-20, full
reasoning trace), [../PROBLEM.md](../PROBLEM.md) §10 open decision #1, [../DECISIONS.md](../DECISIONS.md)
ADR-001, and [../ERRORS.md](../ERRORS.md).

---

## 0. Verdict

**Kein Platz auf dem Score-Pfad. Ein Platz als *billiger Struktur-Validator* — und auch der ist
unbewiesen.** Konkret:

- Als Ersatz für den VLM: **tot gemessen**. TATR-Grid + Tesseract-Text = `final 0.6608` (dev 24) gegen
  MinerU2.5 1.2B `0.9659` raw / `0.9862` mit P0..P5. Der Grund ist nicht das Grid, sondern der Text:
  dieselbe Zelle, oracle Text 0.946 auf T3, Tesseract-Text 0.623 auf T3.
- Als Struktur-Spezialist (Fine-Tune-Basis): **nicht jetzt**. Sein eigenes Struktur-Ceiling ist
  `TEDS_w 0.9365` (oracle Text, dev 24) — *unter* dem, was der VLM mit echtem Text end-to-end liefert
  (dev `final 0.9659` raw, `0.9862` mit P0..P5). Ein Spezialist, der schlechter ist als das
  Generalmodell, löst kein Problem.
- Als Validator: **der einzige offene Slot**, aber mit einem harten Argument dagegen: ein Orakel mit
  0.937 (nur Struktur) kann ein Modell mit 0.9862 (final) nicht validieren. Das Disagreement-Signal
  wäre überwiegend TATRs eigener Fehler. Nicht gemessen ⇒ nicht behaupten. Ein einziger gezielter Test
  (§5) kann das entscheiden, ~1 Stunde.
- Als Fallback-Stufe (Entscheidung #7): **0.6608 > Tesseract-Floor 0.3410**, also formal besser als
  "nichts" — aber es kostet `torch` im Image (§4) für einen Pfad, der in 120 gemessenen Tabellen
  nie ausgelöst hat. Schlechter Trade.

Kurz: TATR ist ein gutes, kleines, deterministisches Strukturmodell, das auf unserer Verteilung an
seiner Textlosigkeit scheitert. Wir behalten es als Werkzeug, nicht als Pipeline-Stufe.

---

## 1. Claim-Tabelle

| Claim | Confidence | Basis |
|---|---|---|
| TATR ist ein **Struktur-Detektor, kein OCR**. Es liefert Zeilen/Spalten/Spans/Header-Boxen, Text muss von woanders kommen. | verified | `src/inference.py`, README; eigene Messung §2/§3 |
| Als Drop-in-Ersatz für MinerU2.5 verliert es deutlich: `final 0.6608` vs `0.9659`. | **measured (wir, dev 24)** | [../MEASUREMENTS.md](../MEASUREMENTS.md) Zeile 2026-11-20 |
| Struktur-Ceiling (TATR-Grid + GT-Text) = `TEDS_w 0.9365`, 23/24 Formen exakt. | **measured (wir, dev 24)** | ebd.; `scripts/tatr_experiment.py` |
| Der T3-Kollaps (T3 TEDS 0.083) ist **Duplikat-Detektion auf Sensorrauschen**, nicht Skew/Kapazität: 3×3-Median + NMS 0.3 → T3 0.946. | **measured (wir, dev 24)** | ebd., §6 Tabelle Zeilen A–C |
| Naives Deskew schadet: −0.21 TEDS auf T2. | **measured (wir, dev 24)** | ebd., Zeile B |
| Repo ist **archiviert** (read-only) seit **2026-09-21**. Keine Upstream-Fixes mehr. | verified | GitHub-Banner, abgerufen 2026-11-21 |
| Code **und** Gewichte sind **MIT**. Kommerziell unproblematisch. | verified | GitHub LICENSE, HF-Modelcard `license: mit` |
| Struktur-Checkpoint: **28,8 M Params, fp32, 115.434.268 Bytes** (≈110 MiB) als `model.safetensors`; zusätzlich `pytorch_model.bin` 115.509.981 B. | verified | HF-API `blobs=true`, 2026-11-21; Modelcard "28.8M params, F32" |
| v1.1-Gewichte (`v1.1-Pub`/`-Fin`/`-All`) liegen **doch** auf HF, als rohe `.pth` (115.508.750 B, MIT). | verified | HF-API `bsmock/TATR-v1.1-All`, 2026-11-21 — **Korrektur** zur älteren Notiz |
| Laufzeit **im Docker-Container auf CPU** ist ungemessen. Gemessen ist nur MPS: 0,08–0,17 s/Tabelle (Struktur allein). | unverified (Container) / measured (MPS) | ebd.; kein CPU-Lauf vorhanden |
| TATR als Validator liefert ein brauchbares Fehlersignal auf VLM-Output. | unverified | nirgends gemessen — mein Hauptzweifel, §5 |
| Fine-Tune auf `generate_dataset.py`-Labels ist technisch billig (28,8 M, VOC-XML, exakte Spans). | medium | Architektur + Datenformat, nicht ausprobiert |
| Detektions-Modell (110 MB) wäre für den Hidden-Trap "PDF / Text um die Tabelle" nützlich. | medium | PROBLEM.md §8; Modell existiert, auf unserem Fall ungemessen |

---

## 2. Modell-Fakten

**Architektur.** DETR mit ResNet-18-Backbone (kein FPN), 6 Encoder-/6 Decoder-Layer, `num_queries=125`,
"normalize before" (LayerNorm vor Attention). Kein OCR-Kopf, kein Text-Slot — Ausgabe ist eine Menge
von Objekt-Boxen mit 6 Klassen: `table`, `table column`, `table row`, `table column header`,
`table projected row header`, `table spanning cell`. Zwei getrennte Modelle, beide 110 MB:

| Modell | Aufgabe | Kennzahl (PubTables-1M test) |
|---|---|---|
| Detektion DETR-R18 | Seite → Tabellen-Boxen (+ `table rotated`, 90° CCW) | AP 0.970, AP50 0.995 |
| Struktur DETR-R18 | Tabelle → Zeilen, Spalten, Spans, Header | GriTS_Top 0.9849, Acc_Con 0.8243 |

**Zahlen, die für uns zählen:** 28,8 M Params, fp32, 110 MiB pro Checkpoint. Zwei Modelle = 220 MiB,
Struktur allein = 110 MiB — Faktor ~40 kleiner als MinerU2.5 1.2B, passt in jedes Image-Budget,
*wenn* nicht `torch` wäre (§4). **Checkpoints / Host:** Hugging Face.
- `microsoft/table-transformer-structure-recognition` — transformers-Format, safetensors, MIT,
  lastModified 2023-09-06. **Das ist der praktikable Weg** (`AutoModelForObjectDetection`), und genau
  das nutzt `scripts/tatr_experiment.py`.
- `bsmock/tatr-pubtables1m-v1.0` — Original-`.pth` (Detektion + Struktur), MIT.
- `bsmock/TATR-v1.1-Pub` / `-Fin` / `-All` — 115.508.750 B `.pth`, MIT, lastModified 2023-08-22.
  v1.1 ist laut Paper besser (bessere Annotationen, FinTabNet-c mit drin), liegt aber **nur als
  Roh-`.pth`** vor: Laden erfordert das Repo-`src/` mit gepinntem torch 1.13 + `timm`, also
  Konvertierungsarbeit. Für uns: v1.0 aus `transformers` nehmen, v1.1 nur, falls wir je fine-tunen.

**Lizenz.** Code MIT, Gewichte MIT (beide HF-Repos `license: mit`), Training-Daten PubTables-1M
**CDLA-Permissive-2.0**. Keine Copyleft-Falle im Modell. **Aber:** `src/inference.py` importiert
`fitz` (PyMuPDF, AGPL) — nur für `Rect`. In unserem Skript ersetzt, im Container irrelevant, solange
wir `src/` nicht mitkopieren.

**CPU vs. GPU.** Das Modell ist klein genug für CPU; 110 MiB fp32, 1000-px-Input, ein Forward-Pass.
Gemessen haben wir es auf Apple-MPS: 0,08–0,17 s/Tabelle. Im Container (16 vCPU, `linux/amd64`, kein
GPU-Zugriff im Submission-Image) ist es **nicht gemessen** — plausibel 0,3–1,5 s/Tabelle, aber das ist
eine Schätzung und steht deshalb nicht als Zahl in §1.

**Offline / Docker.** Vollständig offline machbar: Gewichte ins Image, `HF_HUB_OFFLINE=1`,
`local_files_only=True`. Der eigentliche Reibungspunkt ist die Laufzeit, nicht das Modell:
`torch` 2.15.0 cp312 manylinux_2_28_x86_64 ist **539,2 MB** Wheel-Download, und die Linux-Metadaten
ziehen standardmäßig `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `nvidia-nvshmem-cu13`,
`nvidia-cusparselt-cu13` und `triton` mit (PyPI-Metadaten, 2026-11-21). Nur mit
`--index-url https://download.pytorch.org/whl/cpu` bleibt es bei einem CPU-Wheel. Unser aktuelles
Image ist `python:3.12-slim` + Tesseract + `pytesseract/pillow/numpy` — TATR würde daraus ein
Multi-GB-Image machen. Bei "Runtime und Image-Größe werden neben dem Ranking gezeigt" ist das ein
echter Preis, kein Detail.

---

## 3. Wo es auf unseren drei Achsen scheitert

### Degradation (T0–T3)

| | Befund |
|---|---|
| T2 (Foto, Perspektive, Schatten) | hält mit 0.936 (oracle Text) — besser als erwartet. **Aber** naives Entzerren kostet 0,21 TEDS: das Modell braucht achsenparallele Zeilenboxen, und unser Deskew macht es schlimmer, wenn das Bild schon gerade ist. |
| T3 (Fax, binarisiert, rauschig) | ohne Patch **0,083**. Das ist der eine echte Defekt: Speckle erzeugt 2–3 überlappende Zeilenboxen pro Zeile, aus 13×7 wird 20×39. 3×3-Median + NMS 0.3 → 0,946. Beides außerhalb des Modells, beides bei uns. |
| Skew/Perspektive allgemein | Das Detektionsmodell kennt **genau eine** Rotation: 90° CCW. Beliebiges Skew ist out-of-distribution. Rectification muss davor, gegated auf einen Skew-Schätzer. |
| Auflösung | Kein Problem: unsere Bilder (500–1110 px) liegen bei `MaxResize(1000)` praktisch in Trainingsauflösung. |

### Struktur / Spans

- **Stärke:** Zeilen/Spalten sind gelernte Objekte, nicht detektierte Linien — deshalb funktioniert
  `borderless` (T2 0.936 ohne jede Linieninformation). Spans sind eine eigene Klasse, Zellen entstehen
  durch Subtraktion ⇒ `rowspan`/`colspan` kommen aus Geometrie. Leere Zellen werden konstruktiv
  emittiert (relevant gegen `missing-empty-cells`).
- **Schwäche 1 — ein einziger Header.** `align_headers()` im Original-Code sagt es selbst: keine
  Mehrzeilen-Header. Unser GT hat zweizeilige Header (`rowspan=2`), und laut
  [../ERRORS.md](../ERRORS.md) ist `collapsed-header-row` der **dominante QA-Killer** (15/19 QA-Fehler
  auf val96). Die Zellen existieren, die Header-*Klassifikation* wird plattgemacht. Das ist unser
  Post-Processing, nicht das Modell — aber es heißt: TATR bringt für genau unsere teuerste
  Fehlerklasse **keine fertige Antwort**.
- **Schwäche 2 — Spans bleiben Restfehler.** C′ (ohne Spanning-Cells) = 0.9217, C = 0.9365. Die
  verbleibenden 0,015 sind Spans, plus 1/24 Formfehler. Kein Modellproblem, aber auch kein
  Selbstläufer.
- **Schwäche 3 — HTML-Writer unbrauchbar** (`cells_to_html`: ein `<thead>` pro Header-Zeile, kein
  `<tbody>`, kein `scope`). Wir schreiben unsere Zellen selbst aus `row_nums`/`column_nums`.

### Content (Deutsch, Zahlenformate)

Hier ist TATR **null**. Es gibt keinen Textpfad, also:
- Umlaute (`Vorräte`), `€ % kg ms`, Tausenderpunkte (`79.036,73 %`), `(x)`-Negation — alles Sache des
  OCR und unseres `pipeline/postprocess.py`. Nichts davon verbessert oder verschlechtert TATR.
- Der gemessene Preis: gleiche Zellen, T3 oracle 0.946 → Tesseract 0.623. **0,32 TEDS gehen an
  Zeichen verloren, nicht an Struktur.** Das ist die klarste Aussage der ganzen Untersuchung und
  verschiebt den Aufwand dorthin, wo er hingehört: Text, nicht Grid.

---

## 4. Integration in dieses Repo

**Was heute existiert.** `scripts/tatr_experiment.py` (368 Zeilen) — Host-Experiment, kein
Container-Pfad. Verifiziert: `torch` steht **nicht** in `requirements.txt`, `.venv` hat kein `torch`,
`submission_template/Dockerfile` installiert nur Tesseract + `pytesseract/pillow/numpy`. TATR ist
also derzeit **null** Bestandteil des Submissions-Images. `pipeline/` spricht Modelle über HTTP an
(`providers.py`: `mlx`-Host, `ollama`), nicht lokal — ein lokales TATR bräuchte einen dritten
Provider-Typ.

| Variante | Datei, die ruft | Kosten/Tabelle | ersetzt |
|---|---|---|---|
| **Validator** (empfohlen, falls §5 positiv) | neu `pipeline/validate.py`, gerufen aus `pipeline/postprocess.py` nach P5 bzw. vor dem Schreiben in `submission_template/predict.py` | 110 MiB Gewicht, 1 Forward-Pass; Container-CPU-Zeit **ungemessen** (MPS 0,08–0,17 s) | nichts — nur Routing in den Fallback-Ladder (Entscheidung #7) |
| **Fallback-Stufe** | `pipeline/providers.py` + `predict.py` | dito, plus Tesseract pro Zelle | den "0 Zellen"-Fall; Score ~0.66 statt 0 |
| **Fine-Tune-Basis** | `scripts/tatr_experiment.py` erweitern; Training host-seitig, Gewichte als gepinnte Revision ins Image | Training Stunden auf einer GPU; Inferenz wie oben | TATRs eigene Struktur — nur sinnvoll, wenn Struktur bindend wird |
| **Voller Ersatz** | — | — | gestrichen: 0.6608 gemessen |

**Image-Rechnung, ehrlich:** +`torch` (≥0,5 GB Wheel, installiert mehr) + `transformers` +
`timm`-Reste für v1.1. Realistisch 1,5–3 GB, je nach CPU-Index. Gegen ein 15-GB-Limit unkritisch,
gegen den informellen "leaner is better"-Vergleich ein Nachteil, den man begründen muss.

**Zeit-Rechnung:** unkritisch, selbst 1,5 s/Tabelle läge weit unter 15 s. Aber MinerU2.5 braucht schon
2,56 s; ein Validator darf nicht der Grund sein, dass die 15 s reißen. Deshalb nur auf Tabellen laufen
lassen, die der Post-Processor schon als verdächtig markiert (`nrows`-Ausreißer) — dann kostet er fast
nichts.

---

## 5. Der eine Test, der es entscheiden würde

Frage: **erkennt TATRs Zeilenzahl den `collapsed-header-row`-Fehler des VLM?**

Warum das der einzige sinnvolle Test ist: die Fehlerklasse ist laut [../ERRORS.md](../ERRORS.md) für
**100 % der dev-QA-Fehler (5/5)** und 15/19 der val96-QA-Fehler verantwortlich. Wenn TATR bei genau
diesen Tabellen `nrows` richtig zählt und der VLM nicht, ist ein Validator goldwert. Wenn nicht, ist
die Sache erledigt und wir haben eine Stunde verloren statt eines Wochenendes.

Vorgehen, klein und messbar:
1. `dev` + `val96`, nur die Tabellen mit `nrows`-Abweichung in `challenge/ERRORS.md` (dev/table_016,
   val96/table_050, 070, 094) plus 20 zufällige Kontrollen.
2. TATR-Struktur (C-Konfiguration: 3×3-Median + NMS 0.3), nur Zeilenzahl extrahieren.
3. Metrik: Präzision/Recall von "TATR sagt mehr Zeilen als der VLM" gegen "GT hat mehr Zeilen".
   Kein TEDS nötig, reine Zählung.
4. Abbruchkriterium vorher festlegen: < 3 von 4 Fehlerfällen erkannt ⇒ Validator verworfen.

Kosten: ein Skript-Lauf auf dem Host, kein Image, kein Deployment. 5 von 5 dev-QA-Fehlern hängen an
dieser Klasse — mehr Rechtfertigung braucht TATR nicht, weniger reicht nicht.

---

## 6. Was sich seit der Notiz vom 2026-11-20 geändert hat

| Punkt | Alt (2026-11-20) | Jetzt (2026-11-21, verifiziert) |
|---|---|---|
| Repo-Status | archiviert 2026-09-21 | **bestätigt**, read-only, 2,9k Stars, 96 offene Issues, letzter Commit vor Archivierung |
| v1.1 auf HF | "HF Hub only has v1.0" | **falsch**: `bsmock/TATR-v1.1-Pub/-Fin/-All` liegen als `.pth` auf HF (MIT, 2023-08-22). Nur das *transformers*-Format fehlt für v1.1 |
| Checkpoint-Größe | 110 MB | **bestätigt**, exakt 115.434.268 B safetensors / 115.509.981 B `.bin` |
| Params | 28,8 M | **bestätigt** (Modelcard + HF-API) |
| Lizenz | MIT | **bestätigt**, Code und Gewichte; Daten CDLA-Permissive-2.0 |
| Messwerte | 0.9365 / 0.6608 | unverändert, aus [../MEASUREMENTS.md](../MEASUREMENTS.md) übernommen |
| `torch` im Repo | nicht geprüft | **nicht vorhanden** — weder `requirements.txt` noch `.venv`; TATR ist kein Teil des Images |

---

## 7. Quellen (alle abgerufen 2026-11-21)

- Repo, README, Archiv-Banner, Lizenz, Pretrained-Tabellen, v1.1-News — https://github.com/microsoft/table-transformer
- `src/inference.py` (Transforms, Klassen, Zell-Ableitung, Header-Limitierung) — https://github.com/microsoft/table-transformer/blob/main/src/inference.py
- HF-Modelcard Struktur-Modell (28.8M params, F32, MIT) — https://huggingface.co/microsoft/table-transformer-structure-recognition
- HF-API mit Dateigrößen — https://huggingface.co/api/models/microsoft/table-transformer-structure-recognition?blobs=true
- HF Modelcards v1.0 / v1.1 — https://huggingface.co/bsmock/tatr-pubtables1m-v1.0 · https://huggingface.co/bsmock/TATR-v1.1-All
- PubTables-1M, CVPR 2022 (Renders max. 1000 px, 30 px Crop-Padding) — https://arxiv.org/abs/2110.00061
- Aligning benchmark datasets for TSR, ICDAR 2023 (65→75 % ICDAR-2013, 42→65 % FinTabNet) — https://arxiv.org/abs/2303.00716
- GriTS, ICDAR 2023 — https://arxiv.org/abs/2203.12555
- PubTables-1M Datensatz, CDLA-Permissive-2.0, 117 GB — https://huggingface.co/datasets/bsmock/pubtables-1m
- PyPI `torch` 2.15.0 Metadaten (Wheel 539,2 MB cp312 x86_64; Linux-`nvidia-*`-Deps) — https://pypi.org/pypi/torch/json
- Eigene Zahlen — [../MEASUREMENTS.md](../MEASUREMENTS.md), Reproduktion `scripts/tatr_experiment.py`,
  Volltext der früheren Analyse [../TATR_RESEARCH.md](../TATR_RESEARCH.md)
