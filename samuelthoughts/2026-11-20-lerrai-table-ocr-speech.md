# Robust Table OCR for LLM Consumption — Talk (lerrai Challenge)

**Date:** 2026-11-20
**Context:** lerrai Hackathon, GRAVITY Heilbronn, Campus Founders Batch #9
**Format:** English, spoken, ~200 words

---

## Technical core (English)

The challenge is called *Robust Table OCR for LLM Consumption*. Sounds like an OCR problem. It isn't.

Recognizing the characters is the easy part. Keeping the structure is the hard part. A tax form has merged cells, multi-level headers, numbers that belong to exactly one row and one column. Flatten that into a stream of text and the numbers are all still there — but the model no longer knows which number belongs to which label. One wrong cell assignment and the answer is confidently wrong.

It's the same problem I hit with my Algebra agent. The model reads every symbol correctly and still loses the meaning, because the meaning lives in the layout: which step belongs to which line, what lines up with the equals sign. Two dimensions squeezed into one is lossy.

So the architecture I'd argue for is: image, recognition, a structured intermediate representation with coordinates and cell relationships, validation — and only then the LLM.

Concretely: I don't want to train a new OCR model. I'd run two existing pipelines — Docling, Surya, PaddleOCR — on identical test tables, measure cell-assignment and number errors, and build a validation layer that flags uncertain extractions instead of guessing.

And one question decides the whole design: is the real bottleneck OCR quality, table structure, or LLM reliability? Those are three different engineering problems.

---

## Deutsche Fassung (Kurzfassung)

lerrai ist ein junges Mannheimer KI-Startup (2026) für Steuerkanzleien: TaxTech, Document AI. Das Produkt SteuerSync/Karl Otto gleicht Steuerrechtsänderungen mit Mandantenakten ab. Die Challenge „Robust Table OCR for LLM Consumption" läuft 20.–22.11.2026 im GRAVITY Heilbronn (Campus Founders Batch #9).

Der eigentliche Engpass ist nicht die Texterkennung, sondern die Strukturrekonstruktion: Aus mehrspaltigen Tabellen mit verbundenen Zellen darf kein linearer Textstrom werden, sonst verliert das LLM die Zuordnung von Zahl zu Zeile und Spalte. Also nicht Bild → OCR → Text → LLM, sondern Bild → Erkennung → strukturierte Zwischenrepräsentation → Validierung → LLM.

Das ist dieselbe Problemklasse wie beim Algebra-Agenten: Die Zeichen sind korrekt erkannt, die Beziehung zwischen den Zeilen ist weg. 2D → 1D ist verlustbehaftet.

Mein Ansatz: kein neues OCR-Modell, sondern zwei bestehende Pipelines (Docling, Surya, PaddleOCR) auf identischen Testtabellen vergleichen, Fehler bei Zellzuordnung und Zahlen messbar machen und eine Validierungsschicht bauen, die unsichere Extraktionen markiert.

Offen vor der Implementierung: Eingabeformate, Qualitätsmetriken, Deliverables. Kernfrage ans Team: Ist der Engpass OCR-Qualität, Tabellenstruktur oder die Verlässlichkeit der LLM-Antworten? Drei verschiedene Engineering-Probleme.
