# PP-Structure / PP-StructureV3 — Research Note

**Datum:** 2026-11-21 · **Quelle:** [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache-2.0) ·
**Frage:** verdient PP-Structure einen Platz in unserer Pipeline, und wenn ja welchen?
Zusammen lesen mit [../PLAN.md](../PLAN.md) §M3 Kandidat 2, [../DECISIONS.md](../DECISIONS.md) #1/#2/#7 und
[TATR_RESEARCH.md](TATR_RESEARCH.md) (gleiche Struktur, gleiche Frage).

---
## 0. Verdict zuerst

**Es ist eine vollständige Pipeline, kein Strukturmodell.** `table_cls → structure (SLANet/SLANet_plus/
SLANeXt) → cell_det (RT-DETR-L) → text_det + text_rec (PP-OCRv5) → HTML`. Das Strukturmodell allein
(SLANet_plus, 6.9 MB, 41.8 ms CPU) ist der einzige textfreie Baustein und damit der einzige, der mit TATR
vergleichbar ist.

**Als Ersatz für MinerU2.5 (FINAL 0.9659) fällt es aus** — nicht wegen Tempo, sondern weil es dafür keinen
Beleg gibt: kein veröffentlichter End-zu-End-TEDS für PP-TableMagic, die Modul-"Accuracy" (59.5 / 63.7 /
69.7 %) steht auf einem **internen chinesischen** Set ohne dokumentierte Metrik, und unsere eigene
Baseline wurde **nie ausgeführt** (§1). Wir haben null eigene Zahlen.

**Der eine belastbare Vorteil: Laufzeit.** SLANet_plus Struktur 41.8 ms CPU, Text-Det 57.8 ms, Text-Rec
21.2 ms — eine Tabelle ist auf CPU in **~0.3–0.6 s** durch (Schätzung, §6). Das 15-s-Limit ist hier kein
Thema; bei TableMaster wäre es eines (17.4 s/Bild im Vendor-Beispiel). PP-Structure ist damit der erste
Kandidat, der nicht am Zeitbudget scheitert, sondern nur an Qualität.

| Claim | Confidence | Basis |
|---|---|---|
| PP-Structure ist eine **Pipeline** (Struktur + Zell-Detektion + OCR); SLANet_plus ist das textfreie Strukturmodell darin. | verifiziert | `table_recognition_v2`-Doku, `ppstructure/table/README.md` |
| **15 s/Table sind bei SLANet/SLANet_plus kein Problem** (Struktur 41.8 ms CPU, Sub-Pipeline grob 0.3–0.6 s). | verifiziert (Modul) / Schätzung (Summe) | Modelltabellen, §6 |
| **PP-StructureV3 als Ganzes** ist zu schwer: 3.7–8.8 s/Seite CPU, bis 17.7 GB Peak-RAM. | verifiziert | PP-StructureV3-Doku |
| Kein **End-zu-End-TEDS** für PP-TableMagic/PP-StructureV3 publiziert, **kein WTW**-Ergebnis überhaupt. | verifiziert (Abwesenheit) | Modul-/Pipeline-Dokus |
| SLANet PubTabNet **TEDS 95.89 / Acc 76.31** ist eine **Vendor-Zahl von 2022** auf sauberen PDF-Renders, nicht auf unserer Verteilung. | verifiziert als Zahl / mittel als Prognose | `ppstructure/table/README.md` |
| `SLANet_plus` ist für **wireless/complex tables** beworben — unsere `borderless`-Achse, **ohne Zahl**. | mittel | Modellbeschreibung |
| Auf **Fax/Foto** wurde SLANet nie publiziert evaluiert. Jede Zahl dazu wäre geraten. | mittel (Negativbefund) | keine Quelle |
| Umlaute sind kein Token-Problem (`ppocrv5_dict.txt` enthält `ä ö ü ß €`), Qualität ungemessen. | verifiziert (Dict) / unverifiziert (Qualität) | Dictionary-Datei |
| Unsere Repo-Baseline ist die **moderne** Variante (PP-TableMagic), nie gelaufen; `pred_paddle/` ist PaddleOCR-**VL**. | verifiziert | §1 |
| Als **CPU-Fallback** (#7) ist es der billigste Kandidat, den wir haben. | mittel | Laufzeit verifiziert, Score ungemessen |

**Empfehlung:** nicht als Submission-Pfad, aber (a) **einmal messen** (1.3 GB, ~10 min), (b) als
**CPU-Fallback** vorhalten, (c) falls Struktur je Engpass wird: **SLANet_plus fine-tunen statt TATR**
(6.9 MB statt 110 MB, 42 ms statt MPS-only, Rezept publiziert).

---
## 1. Was im Repo schon liegt — und was der Name nicht bedeutet

Die Arbeit ist zu 80 % getan, nur nicht gefahren.

| Artefakt | Was es wirklich ist | Status |
|---|---|---|
| `baselines/baseline_paddleocr.py` | `TableRecognitionPipelineV2(use_layout_detection=False)` = Pipeline `table_recognition_v2` = **PP-TableMagic**: table_cls + structure + cell_det + PP-OCRv5. Also **nicht** „plain det+rec ohne Struktur", sondern die volle Kette inkl. `rowspan`/`colspan`. | Code da, **nie ausgeführt** |
| `predictions_paddle/` | Zielverzeichnis der Baseline (Docstring) | **existiert nicht** |
| `~/.paddlex` | Modell-Cache des ersten Laufs (~1.3 GB) | **existiert nicht** |
| `pred_paddle/` | **Nicht** die PP-Structure-Baseline: `raw/paddle/*.txt` (PaddleOCR-**VL** 0.9B, OTSL) durch `vlm_to_html.py --format otsl` | 24 Dateien, VLM-Ursprung |
| `ranking.json` → `"team": "pred_paddle"` | Der **0.957-Lauf von PaddleOCR-VL 0.9B**, nicht PP-Structure | `metrics/runs.jsonl`: `paddleocr-vl` |

**Beleg der Verwechslung:** `pred_paddle/table_003.html` ist die OTSL-Dekodierung von
`raw/paddle/table_003.txt` (`<fcel>Metric<fcel>Train…` → `<td>Metric</td><td>Train</td>…`), inklusive
`colspan="2"` aus `<lcel>`-Läufen. Hätte SLANet das erzeugt, käme es aus `predict_paddle.py`, nicht aus
einem OTSL-Konverter. ⇒ Im Repo bedeutet „Paddle" zwei verschiedene Dinge; die **Struktur**-Pipeline wurde
nie gemessen, der `PLAN.md`-Erwartungswert 0.5–0.6 für Kandidat 2 ist bis heute eine Vermutung.

---
## 2. Begriffe

| Name | Was es ist | Für uns |
|---|---|---|
| **PP-Structure** (v1/v2, 2022) | Dokument-Pipeline (Layout + Tabelle + KIE + Mathe); Tabellenpfad = DB-Det + CRNN-Rec + SLANet | nur der Tabellenpfad |
| **PP-StructureV3** (2025/26) | **Seiten**-Pipeline: Layout + General-OCR + UVDoc + Tabellen-Sub-Pipeline + Formel + Siegel + Chart + Reading-Order → Markdown/JSON | zu groß (§6) |
| **PP-TableMagic / `table_recognition_v2`** | Tabellen-**Sub-Pipeline**: table_cls + Struktur **oder** Zell-Detektion, e2e-Modus wired/wireless, plus OCR → HTML | **unser Kandidat** |
| **SLANet / SLANet_plus** | autoregressive HTML-Token-Modelle mit Zell-Box-Regression, 488/512 px, PP-LCNet-Backbone | Struktur-Baustein |
| **SLANeXt_wired/-wireless** | getrennte Gewichte liniert/liniendlos, 351 MB | zu schwer (§3) |
| **TableMaster** | PubTabNet-Sieger 2021 (ResNet-Extra + MASTER-Decoder) | Laufzeit-Killer |
| **RT-DETR-L_*_table_cell_det** | Zell-Detektor, liefert Boxen **ohne** Spans; HTML entsteht geometrisch | interessant, weil span-frei |

---
## 3. Modell-Inventar (Vendor-Zahlen)

Vendor-Testumgebung: Intel Xeon Gold 6271C @2.60 GHz, 8 Threads, FP32, `paddle_static`; GPU = Tesla T4 —
also ein *langsamerer* CPU als unsere 16-Kern-Laptop-CPU ⇒ unsere Zeiten dürften besser sein. Alle
Apache-2.0 (Repo-`LICENSE` gelesen).

| Modell | Aufgabe | Checkpoint | Params | GPU ms | CPU ms | ONNX |
|---|---|---|---|---|---|---|
| SLANet | Struktur → HTML-Tokens + Zellboxen | 6.9 MB | nicht publiziert (fp32 ⇒ ~1.7 M) | 23.96 | **43.12** | ja |
| SLANet_plus | dito, robuster, wireless-tauglich | 6.9 MB | nicht publiziert | 23.43 | **41.80** | ja (e2e 23.42 ms, Gerät unklar) |
| SLANeXt_wired / _wireless | Struktur, getrennte Gewichte | 351 MB | ~88 M (Größenrechnung) | 85.92 | **501.66** | ja |
| TableMaster | Struktur | ~200 MB-Klasse | nicht publiziert | – | Vendor-Beispiel **17.4 s/Bild** | nein |
| RT-DETR-L_*_table_cell_det | Zellboxen | 124 MB | ~31 M (Größenrechnung) | 33.47 | **402.55** (HPM 256.6) | ja (12.07 ms) |
| PP-LCNet_x1_0_table_cls | liniert vs. liniendlos | 6.6 MB | – | 2.62 | **3.17** | ja |
| PP-OCRv5_mobile_det | Textdetektion | 4.7 MB | – | 10.67 | **57.77** | ja |
| PP-OCRv5_mobile_rec | Textrekognition | 16 MB | – | 5.43 | **21.20** | ja |
| PP-OCRv5_server_rec | dito, genauer | 81 MB | – | 8.46 | 31.21 | ja |
| UVDoc | Foto-Entzerrung | 30.3 MB | – | 19.05 | **869.82** | ja |

Packaging: `paddlepaddle` CPU-Wheel **194.8 MB** (PyPI 3.3.1, linux x86_64/cp312), `paddleocr`/`paddlex`
reines Python, Modelle ~1.3 GB im Cache ⇒ **Image-Schätzung 2.5–3.5 GB** (nicht gemessen; Limit 15 GB).
Preis dafür: eine eigene Runtime plus `opencv`-Kette im Image, die wir sonst nicht hätten.

---
## 4. Veröffentlichte Zahlen (Vendor, **nicht unsere**)

| Modell / Set | Metrik | Wert | Quelle | Einordnung |
|---|---|---|---|---|
| SLANet (v2) | PubTabNet **TEDS** | **95.89** | `ppstructure/table/README.md` §2 | eigener Det+Rec davor; saubere PDF-Renders |
| SLANet (v2) | PubTabNet **Acc** (1 falsches Token = falsch) | **76.31** | dito | die ehrliche Zahl: ~jede 4. Tabelle strukturell falsch |
| TableRec-RARE (v2) | PubTabNet TEDS / Acc | 93.88 / 71.73 | dito | Vorgänger |
| SLANet (v2) | Speed CPU+MKL | **766 ms/Bild** | dito | ganze Pipeline, PubTabNet-Größe |
| SLANet-LCNetV2 / TableMaster | PubTabNet Acc | 76.67 / **77.47** | `algorithm_table_*.md` | TableMaster: „langsam, GPU empfohlen", 17.4 s/Bild |
| SLANet / _plus / SLANeXt | „Accuracy" auf **internem chinesischem** Set | 59.52 / 63.69 / 69.65 | Modultabellen v3 | **Metrik nicht dokumentiert** — nicht als TEDS lesen |
| PP-StructureV3 | OmniDocBench **Table Edit** ↓ EN/ZH | 0.159 / 0.109 | Algorithmen-Seite | Edit-Distanz, **kein** TEDS, Seiten-Level |
| PP-StructureV3 lightweight | s/Seite CPU (Intel 8350C) | 3.74 | Benchmark-Tabelle | entspricht ~ unserem Laptop-CPU-Niveau |

**Was fehlt und nicht ersetzt werden kann:** kein **WTW**-Ergebnis für irgendein Paddle-Modell; kein
End-zu-End-TEDS für PP-TableMagic; **keine** Evaluation auf Fotos, Scans, Faxen, Schräglage oder Schatten.
Die 95.89 ist eine T0-artige Zahl. Wer sie auf T3 überträgt, rät.

---
## 5. Wo es auf unseren drei Achsen bricht

**Degradation (T1–T3).** Strukturell blind für Perspektive: 488/512-px-Resize, achsenparallele Zellboxen,
keine Rotationsklasse; UVDoc ist optional und kostet **870 ms CPU**. **Fax** ist der wahrscheinlichste
Totalausfall — Binärisierung + Speckle erzeugt genau die Duplikat-Strukturen, die TATR auf T3 von 0.946 auf
0.083 gedrückt haben (doppelte `<tr>`/`<td>`-Tokens); bei SLANet ungemessen, Mechanismus identisch.
Positiv: SLANet_plus ist laut Vendor offset-unempfindlich (relevant, wenn der Crop die Tabelle nicht sauber
umschließt, T2), SLANet nicht. Zu Rauschen, Unschärfe, JPEG und Durchscheinen ist **nichts** belegt.

**Struktur.** Spans sind nativ (Token-Sequenz trägt `rowspan`/`colspan`) — kein Span-Repair wie bei TATR.
Leere Zellen sind Tokens (`<td></td>`) ⇒ unsere `missing-empty-cells`-Klasse sollte seltener auftreten
([../ERRORS.md](../ERRORS.md)), ungemessen. Liniendlos: SLANet_plus ist dafür beworben, **ohne Zahl** — und
dass Paddle einen `table_cls`-Vorschalter (94.2 % Top-1) braucht, zeigt, dass liniert/liniendlos dort zwei
Verteilungen sind. `<thead>` ist nicht garantiert: Header-Zeilen werden zu normalen Zeilen, für TEDS
harmlos, für unsere QA (Header in Zeile 1–3) teuer — gleiches Risiko wie TATR. Autoregressiv heißt fragil:
ein falsches Token verschiebt den Rest; Acc 76.3 % bei TEDS 95.9 bedeutet, dass strukturell falsche
Tabellen für TEDS billig und für QA (`collapsed-header-row`) tödlich sind.

**Content.** Umlaute sind auf Token-Ebene kein Problem — `ppocrv5_dict.txt` enthält `ä ö ü Ä Ö Ü ß €`
(geprüft) — aber `en_PP-OCRv4_mobile_rec` ist ein englisches Modell: wer darauf schaltet, verliert sie.
**PP-OCRv5_rec nehmen, nicht `en_*`.** Zahlenformate sind reine Zeichenfolgen und werden weder normalisiert
noch zerstört (`(4.269,69)`, `62.929,65 kg` kommen so heraus wie gedruckt) — die Arbeit bleibt in
`pipeline/postprocess.py`. Text wird per Box-Overlap zugeordnet (wie TATR), nicht generativ: bei
mehrzeiligen Zellen mit engen Zeilenabständen der klassische Fehlerort, bei unseren kurzen Zellen
überwiegend unkritisch. Trainiert wurde auf chinesischen Tabellen: Topologie ist sprachunabhängig,
Zellgrößen und Leserichtung nicht — der `borderless`-Stil mit deutschen Zahlenkolonnen ist nicht deren
Verteilung.

---
## 6. Integration — konkret

| Slot | Ersetzt | Seam | Aufwand | Erwartung |
|---|---|---|---|---|
| **A. CPU-Fallback** (#7) | nichts; füllt „VLM OOM/slow ⇒ trotzdem Zellen" | `submission_template/predict.py` + `pipeline/ppstructure_provider.py` | ~1 Tag, Image +1.5 GB | **0.3–0.6 s/Table, 0 VRAM** |
| **B. Struktur-Fine-Tune** | TATR als Basis | neues `scripts/`, Generator-Labels → SLANet-Format | 1–2 Tage + GPU-Stunden | 6.9 MB / 42 ms, kleiner+schneller als TATR (110 MB) |
| **C. Validator** | TATRs Validator-Rolle | `pipeline/` VLM-HTML vs. SLANet-HTML | 0.5 Tag | teurer als TATR (0.1 s), liefert aber Text mit |
| **D. Volle Ablösung** | MinerU2.5 | – | – | **nein** (§0) |

**Der Verdrahtungspfad existiert.** `pipeline/convert.py` kennt `html` als Dialekt (`first_html_table`,
`convert(fmt="auto")`); PP-Structure gibt HTML aus ⇒ Anschluss ist eine Registry-Zeile, kein neuer Parser.
Vorverarbeitung, die es braucht (aus §5, nicht aus der Doku): **Deskew/Perspektive gated** auf einen
Schätzwert (naive Entzerrung hat TATR −0.21 TEDS auf T2 gekostet, TATR §6), **3×3-Median bei T3** (der
einzige belegte Gewinn: T3 0.083 → 0.946, gleicher Duplikat-Mechanismus), **nicht binarisieren**
(PP-OCRv5 und SLANet erwarten Graustufen/Farbe), **Crop-Padding ~10 px**.

Erwartete Laufzeit CPU (Schätzung, Addition der Vendor-Modulzeiten, **nicht gemessen**):

```
table_cls 3 ms + SLANet_plus 42 ms + OCR-det 58 ms + OCR-rec 30–60 ms (5 Zeilen, gebatcht) + Postproc 10 ms
≈ 0.15–0.25 s/Table + 0.1–0.4 s Python/Image-Overhead  ⇒  0.3–0.6 s
```

Das ist **6–8× schneller als MinerU2.5 (2.56 s)** und lässt ~14 s Reserve. Wer den Zell-Detektor statt
SLANet nimmt, zahlt 257–403 ms extra und verliert die Spans. **Fallback-Kette, die ich bauen würde:**
`MinerU2.5 (GPU) → PP-TableMagic (CPU) → Tesseract-Floor` — der mittlere Schritt schließt genau die Lücke,
die heute zu 0 Zellen führt, wenn der VLM nicht lädt; deterministisch, offline, debugbar.

**Was ich *nicht* nehmen würde:** PP-StructureV3 als Ganzes. Es ist eine **Seiten**-Pipeline (3.74–8.83 s/
Seite CPU, 17.7 GB Peak-RAM, Formel-/Chart-/Siegel-Module, die wir nie brauchen) und wir bekommen bereits
**einen Tabellen-Crop pro Bild** — der Layout-Schritt wäre reine Kosten.

---
## 7. Unverifiziert — was eine Messung sofort klärt

| Offene Frage | Warum sie zählt | Billigster Test |
|---|---|---|
| Was misst „Accuracy 63.69" bei SLANet_plus? | entscheidet, ob es TATRs 0.9365 (Oracle-Text) schlägt | Doku-Metrik suchen; sonst selbst messen |
| PP-TableMagic **End-zu-End auf `data/dev`** | ersetzt den `PLAN.md`-Erwartungswert 0.5–0.6 durch eine Zahl | `baselines/baseline_paddleocr.py` + `score_all.py` |
| SLANet_plus auf **T3/Fax** und auf **borderless** | Ausfallpunkt bzw. die einzige Vendor-Behauptung, die für uns zählt | derselbe Lauf, Tier-/Style-Split |
| Umlaut-Genauigkeit (`Mär`, `ß`) | Dict enthält sie, Qualität unbekannt | 20 Zell-Crops, Zeichen-F1 |
| ORT-Zeiten (23.4 ms) auf **CPU**, Image-Größe real | entscheidet Image-Aufbau und was das Ranking sieht | `engine="onnxruntime"` timen, `docker images` |

**Ehrlich:** diese Notiz enthält **keine einzige eigene Zahl** für PP-Structure — alles Vendor-Material
plus Repo-Fakten. Der Grund ist nicht Recherche-Faulheit, sondern dass der Lauf nie gemacht wurde; er ist
der nächste Schritt, nicht ein weiterer Artikel.

---
## 8. Quellen (abgerufen 2026-11-21)

- Repo, Apache-2.0, Code — https://github.com/PaddlePaddle/PaddleOCR (`LICENSE` gelesen)
- v2-Tabellen-Pipeline, TEDS 95.89 / Acc 76.31 / 766 ms CPU — https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/table/README.md
- SLANet-LCNetV2 (Acc 76.67) / TableMaster (Acc 77.47, „slow, GPU recommended", 17.4 s/Bild) —
  `docs/version2.x/algorithm/table_recognition/algorithm_table_slanet.md` und `…_table_master.en.md`
- Struktur-Modul v3 (59.52 / 63.69 / 69.65, Größen, CPU-Zeiten, `engine="onnxruntime"`) —
  https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/table_structure_recognition.en.md
- Zell-Detektion (RT-DETR-L, mAP 82.7, 124 MB, 402.55 ms CPU) —
  https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/table_cells_detection.en.md
- **PP-TableMagic = General Table Recognition v2** (`TableRecognitionPipelineV2`, wired/wireless-Schalter) —
  https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/table_recognition_v2.en.md
- PP-StructureV3 (Komposition, Table-Edit 0.159/0.109, 3.74–8.83 s/Seite CPU, 17.7 GB Peak-RAM) —
  https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-StructureV3/PP-StructureV3.en.md
- OmniDocBench (Herkunft der Table-Edit-Metrik) — https://arxiv.org/abs/2412.07626
- PP-OCRv5-Dictionary mit `ä ö ü ß €` — https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/ppocrv5_dict.txt
- `paddlepaddle` 3.3.1 CPU-Wheel 194.8 MB — https://pypi.org/pypi/paddlepaddle/json
- Repo-intern: `baselines/baseline_paddleocr.py`, `pred_paddle/`, `raw/paddle/`, `ranking.json`,
  `metrics/runs.jsonl`, `pipeline/convert.py`, [../MEASUREMENTS.md](../MEASUREMENTS.md),
  Referenzzahlen [../../metrics/LEADERBOARD.md](../../metrics/LEADERBOARD.md)
