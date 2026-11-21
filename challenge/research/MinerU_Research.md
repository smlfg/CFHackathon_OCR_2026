# MinerU (opendatalab) — Recherchenotiz

**Datum:** 2026-11-21 · **Frage:** Warum ist MinerU2.5 1.2B unsere beste Linie, was davon ist Modell
und was unsere Schicht, und was müsste passieren, damit etwas anderes sie ersetzt?
Zusammen lesen mit [../MEASUREMENTS.md](../MEASUREMENTS.md), [../ERRORS.md](../ERRORS.md),
[../TATR_RESEARCH.md](../TATR_RESEARCH.md) (Vorlage), [../../metrics/LEADERBOARD.md](../../metrics/LEADERBOARD.md).
Zahlen ohne Quellenangabe sind **von uns gemessen**; Fremdzahlen sind `publiziert` oder `Bericht`.
Vertrag unverändert: Foto/Scan/Fax → Zellen `[{text,r,c,rs,cs}]`, offline, Laptop, Docker,
15 s/Tabelle, `final = 0.85·TEDS_w + 0.15·QA`.

---

## 0. Verdikt

**MinerU2.5 1.2B ist nicht wegen eines einzelnen Tricks unsere beste Linie, sondern weil drei Dinge
zusammenfallen:** (a) das Modell gibt **OTSL** aus — eine Matrixsprache mit fünf Struktur-Tokens statt
HTML mit 28 —, Spans (`lcel`/`ucel`/`xcel`) sind also billig; (b) es läuft bei uns in **Stadium II** des
Herstellers (native Auflösung, `max_edge=0`, Prompt `Table Recognition:`), weil unsere Eingabe bereits
**eine** zugeschnittene Tabelle ist — Stadium I (Layout) fällt samt seiner Fehler weg; (c) unser
Postprocessing repariert genau die vier Klassen, die das Modell offen lässt.

Der Anteil ist messbar ungleich verteilt: **Modell ≈ 0.9659 (dev) / 0.9621 (val96)**, **Postprocessing
+0.0203 / +0.0187**. Von den verbleibenden 0.0138 (dev) bzw. 0.0192 (val96) sind rund 45 % bzw. 31 %
QA-Verlust, und davon 79 % (15/19) **eine** Ursache: zusammengeklappte zweizeilige Header.

**Empfehlung:** Linie behalten, nicht weiter am Modell drehen. Offen sind Header-Struktur (QA) und —
wichtiger — **eine MinerU-Zahl im Ziel-Docker**, die es bis heute nicht gibt.

---

## 1. Modell vs. unsere Schicht — die Zerlegung

Drei Schichten, die man nicht verwechseln darf. „Roh" heißt: Modelltext **plus** unser OTSL-Parser
(`pipeline/convert.py:otsl_to_cells`) — also nicht reines Modell.

| Schicht | Was sie tut | Was sie wert ist | Beleg |
|---|---|---|---|
| **Modell** (1.2B VLM) | Bild → OTSL-Text; Struktur als Tokenfolge, Text als Zeichen | T0 0.959 · T1 0.992 · T2 0.956 · T3 0.966 (dev, roh) | `metrics.sqlite` `mineru2.5-1.2b__dev__9e485ad5` |
| **Konverter** (unser Code) | `<fcel>/<lcel>/<ucel>/<xcel>` → Zellen mit `r/c/rs/cs`, HTML | nicht separat gemessen — im „roh"-Wert enthalten | `pipeline/convert.py` |
| **Postprocessing** P0–P5 | Phantomspalten (P0), Gedankenstrich (P1), Zahlenformat (P2), numerische `cs=2` per Tintenprojektion (P3), Header-Ecke (P4), geteilte Platzhalter (P5) | **+0.0203 dev / +0.0187 val96** | `pipeline/postprocess.py`, `scripts/ablate_postprocess.py` |

Zwei Konsequenzen: **(1) „Roh" ist schon halb wir** — der Sprung von Modelltext zu bewertbaren Zellen
(Spans auflösen, Grid-Positionen footprinten) ist unsere Arbeit; wer 0.9659 als „Modellleistung"
zitiert, zitiert unsere Fehler mit. **(2) Der größte Einzelgewinn ist eine Textnormalisierung** —
P1 allein `+0.0124` dev, `+0.0085` val96, also 61 % bzw. 45 % des gesamten Postprocessing-Gewinns,
für ein `ASCII "-" → "–"/"—"`. Kein Modellproblem, ein Zielformat-Problem.

---

## 2. Gemessene Zahlen (Stand 2026-11-20)

Kopiert aus [../MEASUREMENTS.md](../MEASUREMENTS.md), MinerU-Zeilen mit unveränderten Werten
(Beschriftung gekürzt); Spalten wie dort.

| Datum | Konfiguration | Datensatz | T0 | T1 | T2 | T3 | QA | Final | s/Tabelle |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-20 | MinerU2.5 1.2B, bf16, **roh** (MLX) | dev (24) | 0.959 | 0.992 | 0.956 | 0.966 | 0.958 | **0.9659** | 2.56 |
| 2026-11-20 | MinerU2.5 1.2B, bf16, **roh** (MLX) | val96 (96, seed=2) | 0.986 | 0.975 | 0.945 | 0.963 | 0.960 | **0.9621** | 2.33 |
| 2026-11-20 | + T3 `median3`-Preproc (nur T3; verworfen: kostet CellF1/QA) | val96 T3 (24) / dev T3 (6) | — | — | — | 0.956 / 0.956 | 0.850 / 1.000 | — | 2.08 / 2.19 |
| 2026-11-20 | + T3 `despeckle1`-Preproc (nur T3; **roh+pp gewinnt Ende-zu-Ende**) | val96 T3 (24) / dev T3 (6) | — | — | — | 0.970 / 0.966 | 1.000 / 1.000 | — | 2.21 / 2.29 |
| 2026-11-20 | **+ Postprocessing P0..P5** | dev (24) | 0.979 | 1.000 | 0.977 | **1.0000** | 0.958 | **0.9862** | 2.56+pp |
| 2026-11-20 | **+ Postprocessing P0..P5** | val96 (96, seed=2) | 0.995 | 0.983 | 0.975 | **0.9891** | 0.960 | **0.9808** | 2.33+pp |

**Die entscheidende Zeile ist die letzte.** Preprocessing auf T3 hilft dem *rohen* Modell
(`despeckle1` hebt T3 von 0.963 auf 0.970), aber **Ende-zu-Ende verliert es**: val96 T3 mit PP
`0.9891` roh vs. `0.9881` despeckle1 vs. `0.9740` median3 ([../ERRORS.md](../ERRORS.md) §3). Preproc
und PP reparieren dieselben Stellen, PP besser, und `median3` kostet zusätzlich QA (0.850).
**Regel: Bildvorbereitung nur, wenn sie gegen `roh + P0..P5` gewinnt, nicht gegen `roh`.**

Rohdaten-Nebenbeobachtungen (gleiche Läufe, `raw/mineru2.5-1.2b/meta.json`): s/Tabelle **Median 2.02
(dev) / 2.17 (val96)**, p90 2.90 / 3.38, Max 4.08 / 4.87 · Tokens 15.944+8.875 (dev) bzw.
68.529+38.602 (val96) · `empty_rate` 0.029 / 0.022 · Zellen Median 48 / 50.5, max 87 / 107 ·
kalter Load 61.0 s, warmer 1.5 s.

Dass **T3 roh (0.966) über T0 roh (0.959)** liegt, ist bei 6 Tabellen pro Tier Rauschen: dev ist pro
Tier mit ±0.03 unsicher. Entscheidungsmenge ist val96 (96 Tabellen), nicht dev.

---

## 3. Claims

| Claim | Confidence | Basis |
|---|---|---|
| MinerU2.5 1.2B ist unsere beste Linie: 0.9659 dev roh, 0.9862 dev mit P0..P5 | **gemessen** (n=24) | [../MEASUREMENTS.md](../MEASUREMENTS.md), 2026-11-20 |
| Der Rang hält auf 96 Tabellen: 0.9621 roh, 0.9808 mit P0..P5 | **gemessen** (n=96) | dito, val96 seed=2 |
| Postprocessing trägt +0.0203 dev / +0.0187 val96 bei; P1 allein +0.0124 / +0.0085 | **gemessen** | MEASUREMENTS + ERRORS §Top-3 |
| T3-Preprocessing (`median3`, `despeckle1`) ist gegen `roh+pp` ein Rückschritt | **gemessen** | ERRORS §3 (0.9891 / 0.9881 / 0.9740) |
| Median 2.17 s/Tabelle auf M3 Max (MLX, bf16), p90 3.38 s | **gemessen** | `raw/val96/mineru2.5-1.2b/meta.json` |
| Vorsprung auf Platz 2/3 ist statistisch **nicht** gesichert: +0.002 (Chandra), +0.009 (PaddleOCR-VL), KI95 schließt 0 | **gemessen** (gepaarter Bootstrap auf dev) | `autoresearch/UPSTREAM_SEARCH.md` §0 |
| Auf val96 ist **nur MinerU** gelaufen — die Konkurrenz nur auf dev-24 | **gemessen** (Abwesenheit) | `metrics/LEADERBOARD_val96.md`: 1 Modell |
| Läuft im Ziel-Docker (linux/amd64, CUDA, offline, 15 s) | **nicht gemessen** | — (siehe §8) |
| Architektur: 0.5B Qwen2-Instruct (M-RoPE) + 675M NaViT (Qwen2-VL), 2×2 pixel-unshuffle → 1.2B | publiziert | arXiv 2509.22186 §3.1 |
| Zielsprache OTSL: 5 Struktur-Tokens statt 28, ~50 % kürzere Sequenzen | publiziert | arXiv 2509.22186 §4.2.3 |
| OmniDocBench: Overall 90.67, Table TEDS 88.22, TEDS-S 92.38 | publiziert (Vendor-Benchmark, nicht unsere Verteilung) | arXiv 2509.22186 §5.1 |
| Elementebene: PubTabNet 89.07/93.11, FinTabNet 95.97/97.61, In-house 71.48/82.83 | publiziert | dito, Table 10 |
| 2.12 Seiten/s auf 1× A100 mit vLLM; 0.95 Seiten/s ohne Deployment-Tuning | publiziert, **andere Hardware** | dito §3.4 |
| Gewichte: **AGPL-3.0**; bf16-MLX-Build 2.31 GB (mlx-vlm 0.3.3) | publiziert | HF-Cards `opendatalab/…` und `mlx-community/…-bf16` |
| CPU-only ≈ 47 s/Seite (llama.cpp) | **Bericht** (aus zweiter Hand, nicht von uns geprüft) | UPSTREAM_SEARCH Q2-Nachtrag |
| 1.x-Pipeline (modular) Table TEDS 70.90 / TEDS-S 79.11 | publiziert | arXiv 2509.22186, Table 5 |

---

## 4. Modellfakten

| Feld | Wert | Quelle |
|---|---|---|
| Parameter | 1.2B total (675M Vision + 0.5B LLM), dense | Paper §3.1 |
| Architektur | Qwen2-VL-artig; nativer Resolution-Encoder (NaViT), M-RoPE, 2×2-Pixel-Unshuffle | Paper §3.1 |
| Zwei Stufen | I: Layout auf herunterskaliertem Bild · II: Inhalt auf Crops in Originalauflösung | Paper §3.2 |
| Ausgabe (Tabelle) | OTSL: `<fcel> <lcel> <ucel> <xcel> <ecel> <nl>`, danach HTML | Paper §4.2.3; `pipeline/convert.py` |
| Quantisierung bei uns | **bf16** (MLX), `temperature=0`, `max_tokens=4096`, `max_edge=0` | `registry/models.json`, `metrics.sqlite` |
| Gewichte | MLX bf16 **2.31 GB**; Original BF16-safetensors (Größe hier nicht geprüft) | HF-Cards |
| Host | `scripts/mlx_server.py` auf dem Mac (Metal), HTTP an den Container | `pipeline/README.md` |
| Lizenz | **AGPL-3.0** (Gewichte + Repo) | HF-Card |
| Release | 2025-09-26 (arXiv v1), Gewichte `-2509-` | arXiv 2509.22186 |
| GPU nötig? | Nein im Prinzip (ONNX/llama.cpp-Pfad existiert), **ja für 15 s** | Paper §3.4 + Bericht |

**Laufzeit auf diesem Laptop:** 2.02 s Median (dev) / 2.17 s (val96), p90 2.90 / 3.38, Max 4.08 / 4.87 —
gemessen als HTTP-Roundtrip inkl. Bildkodierung, ohne Load. Das ist Metal, **nicht** die Ziel-GPU.

**Docker-/Offline-Story — der ehrliche Stand.** Unser Messaufbau ist `Host-MLX-Server + Container über
HTTP`. Im offiziellen Lauf gibt es keinen Host-Prozess; `--network none` killt genau diesen Pfad. Für
die Abgabe muss dasselbe Modell **im** Image laufen (vLLM, transformers oder llama.cpp, `linux/amd64`,
Gewichte beim Build gepinnt). Plausibel machbar — 2.31 GB bf16 passen in 8 GB VRAM, die Herstellerdoku
nennt 8 GB+ VRAM und ~2–3 GB Modelldownload, `mineru-vl-utils` unterstützt transformers/vLLM/SGLang/
llama.cpp —, aber **gemessen ist es nicht**. Die Image-Größe ist bei 2.3 GB Gewicht unkritisch.

**Was 15 s/Tabelle an Headroom lassen** (gemessen auf Metal, Ziel-GPU unbekannt): bei 2.17 / 3.38 / 4.87 s
(Median / p90 / max) sind das **6.9× / 4.4× / 3.1×** Reserve; der 5-Minuten-Start ist mit 61 s kaltem
Load ~4× gedeckt. Das reicht für **einen** Zusatzschritt pro Tabelle — Wiederholung bei leerer Ausgabe
(+2.2 s), ein zweiter VLM-Pass (+2.2 s) oder ein Struktur-Validator (TATR, 0.1 s) —, für beides zusammen
wird es eng. Die Postprocessing-Wanduhr ist `unmeasured` (steckt in „2.56+pp"). Auf reiner CPU ist der
Headroom **negativ** (~47 s/Seite, Bericht).

---

## 5. Die ältere Linie: MinerU 1.x und 2.0 — was tablespezifisch anders war

| Generation | Aufbau | Tabellenverhalten | Verlust gegen 2.5 |
|---|---|---|---|
| **MinerU 1.x** (2024-09, PDF-Extract-Kit) | Kette: Layout-Erkennung → Tabellenstrukturmodell (RapidTable/StructEqTable) → PaddleOCR je Zelle → Formelmodell → Regelwerk | Struktur aus einem **eigenen** Modell, Text aus OCR: zwei Fehlerquellen, die sich multiplizieren. Spans geometrisch, leere Zellen nur, wenn das Strukturmodell sie erzeugt | Table TEDS **70.90** (publiziert) |
| **MinerU 2.0** (2025-05/06, `MinerU2.0-2505-0.9B`) | ein 0.9B-VLM ersetzt OCR+Formel, Layout bleibt separat | ein Modell für Struktur+Text, HTML-Ziel → lange Sequenzen, Span-Fehler bei großen Tabellen | Table TEDS **83.54** / TEDS-S 87.66 (publiziert) |
| **MinerU2.5** (2025-09, 1.2B) | ein Modell, zwei Stufen; native Crops; OTSL-Ziel; Tabellen zusätzlich geometrisch entzerrt (BBox+Rotation) | OTSL macht Spans billig; Rotation/rahmenlos sind Trainingsziel | Table TEDS **88.22** / TEDS-S 92.38 |

Drei Punkte, die für uns zählen:

1. **Der Sprung 2.0 → 2.5 ist auf Tabellen der größte** (+4.7 TEDS publiziert): OTSL statt HTML plus
   native Auflösung; 1.x → 2.0 (+12.6) war der Wechsel von OCR zu VLM.
2. **Die 1.x-Kette würde an unseren T3 scheitern, nicht an der Struktur.** Analog gemessen:
   TATR-Struktur auf T3 = 0.946 mit Orakeltext, aber 0.623 mit Tesseract-Text
   ([../TATR_RESEARCH.md](../TATR_RESEARCH.md) §6). MinerU 1.x selbst ist auf unseren Daten
   **`unmeasured`**; Experiment: `mineru` 2.2.2 (letzte Version mit 2.0-Modell) auf `data/val96`.
3. **Die 2-Stufen-Struktur ist für uns ein Glücksfall** — Stufe I brauchen wir nicht, wir rufen
   Stufe II direkt auf. Aber: **wenn je „Text um die Tabelle" kommt** (PROBLEM.md §8), fehlt uns
   genau die Stufe, die das löst; dann ist MinerU als Ganzes zu prüfen, nicht nur Stufe II.

---

## 6. Fehleranalyse: was übrig bleibt

Aus [../ERRORS.md](../ERRORS.md), gemessen auf 120 Tabellen (dev+val96). Frage: **welche Fehler
überleben P0..P5?** (Vendor-Kontext: schwächster der fünf Hersteller-Tabellenbenchmarks ist die eigene
In-house-Tabelle mit 71.48 TEDS — Vendor-Zahlen decken unsere Verteilung nicht ab.)

| Klasse | Was PP dagegen tut | Rest nach PP | Kosten danach |
|---|---|---|---|
| `unicode-dash` | P1 normalisiert auf `–`/`—` | **weg** (Hauptgewinn +0.0124 dev) | — |
| `split-placeholder` (`—`+`""`) | P5 merged nur bei Tinte über der Spaltengrenze | Rest in Grenzfällen | klein |
| `illegal-numeric-colspan` | P3 trennt per Tinten-Projektion | **weg** auf T3 (0.9891 val96, 1.0000 dev) | — |
| `missing-corner-rowspan2` | P4 repariert leere `(0,0)`-Ecke | weitgehend weg | +0.0014 val96 vorher |
| **`collapsed-header-row`** | **nichts** — 2 Headerzeilen zu 1 fusioniert (`nrows` um 1 falsch) | **bleibt** | **QA: 15/19 val96-Fehler, 5/5 dev** |
| `number-format` (Leerzeichen-Tausender) | P2 nur bei ≥80 % Stimmen einer Formatfamilie | 2/96 val96, beide T3 | klein |
| `text` (Umlaute, `Item`→`ltem`, `6`→`8`) | nichts | 9/96 Textfehler, 1/96 Ziffernfehler (val96) | klein |

**Was T3 (Fax) konkret tut:** Es bricht die Struktur **nicht** — T3 roh 0.9632, nach PP 0.9891, der
zweithöchste Tier-Wert. Fax kostet (a) Platzhalter-Trennung (6/24 T3-Tabellen), (b) Leerzeichen in
Tausendergruppen (2/96, beide T3), (c) Gedankenstriche (11/24): **Text- und Formatklassen**, kein
Grid-Kollaps — Gegensatz zu TATR, wo T3 von 0.083 auf 0.946 sprang, als doppelte Zeilen/Spalten weg waren.

**Was für QA übrig bleibt — und wie viel das noch wert ist** (Arithmetik auf gemessenen Zeilen):

| Posten | dev (24) | val96 (96) |
|---|---|---|
| Verlust aus TEDS_w | 0.0090 → **0.0077 final** | 0.0158 → **0.0134 final** |
| Verlust aus QA | 0.042 → **0.0063 final** | 0.040 → **0.0060 final** |
| davon allein „collapsed header" | 0.0063 (100 % der QA-Fehler) | 0.0047 (79 %) |

Heißt: Auf val96 liegen noch **0.0192 Final** auf dem Tisch, 0.0134 in der Struktur (T2/T3
`hlines`/`borderless`) und 0.0060 in QA — und die QA-Hälfte hat **eine** Ursache. Ein Header-Fix ist
der billigste verbleibende Gewinn, aber auf ~0.005 gedeckelt.

---

## 7. Wann MinerU ersetzt werden darf — Schwellen, keine Gefühle

Zwei Fragen, zwei Schwellen. **A) Ersatz *innerhalb* unserer Kette** (Modell tauschen, P0..P5 behalten):

| Kriterium | Schwelle | Warum diese Zahl |
|---|---|---|
| `final` auf **val96** | ≥ **0.9908** (Δ ≥ +0.010) | +0.010 ist die Hälfte der gemessenen dev-Bootstrap-Breite (±0.02); kleinere Δ sind auf 96 Tabellen nicht belegbar |
| Gepaarter Bootstrap | 95 %-Untergrenze **> 0** auf denselben Tabellen | Punkt-Δ allein reicht nicht; MinerU−PaddleOCR-VL = +0.009 mit KI [−0.017; +0.038] ist der Präzedenzfall |
| T3 (40 % Gewicht) | ≥ 0.9891 nach eigenem PP, **kein Tier −0.02** | T3 dominiert die Gewichtung; ein T0-Sieg bei T3-Verlust ist kein Sieg |
| Laufzeit | p90 ≤ **5 s/Tabelle**, Kaltstart ≤ **300 s**, im **Ziel-Container** | 3× Reserve zum 15-s-Gate; MLX-Zahlen zählen hier nicht |
| Ressourcen | ≤ 8 GB VRAM bei gewählter Quantisierung, Image ≤ 15 GB | hartes Zielgerät |
| Lizenz | darf im Image mitgeliefert werden; AGPL-Frage vorher geklärt | Abgabe = Gewichte im Image |

**B) Ersatz *unserer ganzen Linie*** (Modell, das P0..P5 überflüssig macht). Dann gilt zusätzlich:

> Ein Modell, das unser Postprocessing nicht braucht, muss **roh** ≥ **0.9808 (val96)** bzw.
> **0.9862 (dev)** erreichen — also das Niveau, das wir *nach* P0..P5 haben. Alles darunter ist ein
> Zulieferer, kein Ersatz.

Zusatzbedingung: Der Kandidat muss auf **val96** laufen, bevor er diskutiert wird. Heute ist dort nur
MinerU gemessen; Platz 2–4 der dev-Rangliste beruhen auf 24 Tabellen und einem nicht-signifikanten
Abstand. **Ein Vergleich auf dev-24 ist keine Entscheidungsgrundlage.**

---

## 8. Offene Messungen (was in dieser Notiz `unmeasured` ist)

| Offen | Das eine Experiment, das es füllt |
|---|---|
| MinerU im Ziel-Container (amd64, CUDA, offline) | `docker buildx --platform linux/amd64` + Lauf auf `flat/` mit `--network none`, eine Zeile in MEASUREMENTS |
| Postprocessing-Wanduhr | `score_all.py`/`ablate_postprocess.py` mit `time.perf_counter` um `postprocess_html`, p50/p90 über val96 |
| Per-Pass-Attribution (P0/P2–P5 einzeln) | `scripts/ablate_postprocess.py --model mineru2.5-1.2b --datasets dev val96` (die `cum P0..Pk`-Zeilen) |
| Quantisierung (8-bit/4-bit) auf unseren Daten | derselbe Lauf mit `mlx-community/...-8bit`, Δfinal gegen bf16 |
| MinerU 1.x/2.0 auf unseren Daten | `mineru` 2.2.2 (letzte 2.0-Version) auf `data/val96`, gleiche Auswertung |
| CPU-Laufzeit im Container | derselbe Docker-Lauf ohne `--gpus`, eine Tabelle, Wanduhr |

---

## 9. Quellen (abgerufen 2026-11-21)

- Repo: https://github.com/opendatalab/MinerU — Versionen, `pipeline`- vs. `vlm`-Backend, 4.x-Tiers
- HF-Gewichte: https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B — Lizenz **agpl-3.0**, BF16, `qwen2_vl`
- HF-MLX-Build: https://huggingface.co/mlx-community/MinerU2.5-2509-1.2B-bf16 — **2.31 GB**, mlx-vlm 0.3.3
- Technischer Bericht: https://arxiv.org/abs/2509.22186 (v2, 2025-09-29) — Architektur §3.1, zwei Stufen §3.2, Deployment §3.4, OTSL §4.2.3, OmniDocBench §5.1, Tabellen-Benchmarks Table 10, Prompts Anhang 8.4
- Vorgänger: https://arxiv.org/abs/2409.18839 (MinerU 1.x, PDF-Extract-Kit) · OmniDocBench: https://github.com/opendatalab/OmniDocBench
- Eigene Zahlen: [../MEASUREMENTS.md](../MEASUREMENTS.md), [../ERRORS.md](../ERRORS.md), [../../metrics/LEADERBOARD.md](../../metrics/LEADERBOARD.md), `metrics/metrics.sqlite` (`mineru2.5-1.2b__dev__9e485ad5`, `__val96__9e485ad5`), `raw/*/mineru2.5-1.2b/meta.json`
- Eigene Vorarbeit: [../TATR_RESEARCH.md](../TATR_RESEARCH.md), `challenge/autoresearch/UPSTREAM_SEARCH.md` (Bootstrap-KIs, Docker-Lücke)
