# The problem, structured

One page you can hold against every decision. Source: kick-off transcript
([TRANSCRIPT.md](TRANSCRIPT.md)) + `README.md` in the repo root. Where the two disagree, the README
wins and the difference is flagged.

## 1. In one sentence

Given a photo, scan or fax of a **table**, produce a machine-readable table — every cell with its
text, grid position and spans — accurate enough that a program can query it, on a **laptop, offline,
inside Docker**.

## 2. Why it exists (business)

Lerai builds an AI platform for **tax consultants**. Their daily work is going over client documents
(invoices, purchases, hires, office purchases) and producing quarterly reports for finance and the
Finanzamt, then re-reading the Finanzamt's answer, producing another report, and later citing court
decisions back into tax law. Thousands of clients × tens of thousands of documents each.

The bottleneck this challenge attacks: German tax offices are still paper-heavy. Documents arrive as
scanned PDFs, phone photos, faxes — skewed, shadowed, wrinkled. Today the fallback is "send the whole
image to an LLM", which reads well but is **too expensive** for a 50-page PDF answering one question.
So: cheap, local, document-structure-aware extraction, so the expensive model is only asked the
semantic question.

## 3. What makes it hard

Three independent failure axes. Each one alone can ruin the table.

| Axis | What it is | Why it hurts |
|---|---|---|
| **Degradation** | skew, perspective, shadow, blur, noise, JPEG, fax binarization, bleed-through | text is still legible to a human, so every error is the pipeline's fault |
| **Structure** | merged cells (rowspan/colspan), two-row headers with spanning group labels, borderless layouts, empty cells | one wrong span or a shifted column silently attaches a number to the wrong row — the downstream consumer never notices |
| **Content** | German/English text with umlauts, units (`% € kg ms`), one number format per table (`1,234.56` / `1.234,56` / `12 345`, negatives as `-x` or `(x)`) | a single misread separator changes the value; format must be normalized, but signs and units kept |

Prior work that isn't good enough: **Tesseract** (the company baseline, CV from the 70s) — fine on
clean grid tables, collapses on photos and faxes (final 0.348). Sending whole page images to an LLM
works but is uneconomical. That gap is the task.

## 4. Input and output contract

- **In:** `/data/images/<table_id>.png|.jpg` — one image per table (flat folder, no subdirectories).
- **Out:** `/data/predictions/<table_id>.json` (or `.html`, JSON wins if both) — a list of cells:
  ```json
  [{"text": "Position", "r": 0, "c": 0, "rs": 2, "cs": 1},
   {"text": "Delta",    "r": 0, "c": 1, "rs": 1, "cs": 3},
   {"text": "Jan",      "r": 1, "c": 1}]
  ```
  - `r`, `c` = zero-based top-left grid position; `rs`, `cs` = spans, default 1.
  - `text` = exactly as printed (number format, sign, unit). Whitespace normalized by the scorer.
  - **Write empty cells too** (`"text": ""`) — omitted empty cells cost TEDS.
  - Header vs body is not distinguished.
  - Unparseable file = 0 for that table.
- **Not:** a new PDF, a summary, prose. Structure preserved, text-based.
- The **format** is ours to choose within that contract; the benchmark stops at "cells out".

## 5. How it is scored

```
final = 0.85 × tier-weighted TEDS + 0.15 × QA accuracy
tier-weighted TEDS = 0.1·T0 + 0.2·T1 + 0.3·T2 + 0.4·T3
tie-breaker: cell F1 (exact text at exact position)
```

- **TEDS** = tree edit distance between our HTML tree and the truth. Missing row / wrong span costs
  a lot; one wrong character in a long cell costs little.
- **QA accuracy** = ~5 questions per table (lookup + aggregate) answered **only from our cells**:
  row label matched in column 1, column header matched anywhere in the first three rows. Labels
  match case/space-insensitively; numbers match by value (`1,234.56` = `1.234,56` = `1234.56`).
  One shifted column or one misread label = wrong answer.
- **Degraded tiers dominate on purpose** (0.7 of the weight is T2+T3). Perfect on clean images only
  ⇒ ceiling ≈ 0.3.

## 6. Hard constraints

| | |
|---|---|
| Inference | **offline**, no network, **open-weight models only** (training/synthesis beforehand free) |
| Packaging | Docker image, built from the **repo root** with `-f submission_template/Dockerfile .`; entrypoint `<entry> /data/images /data/predictions` |
| Weights | inside the image, downloaded at build time, pinned revision, not committed to git |
| Machine | RTX 4070 Laptop **8 GB VRAM**, 16 CPUs, 24 GB RAM, `linux/amd64` |
| Time | 5 min startup + **15 s per table** (~29 min for ~96 tables); write each prediction as soon as it exists |
| Image | ≤ 15 GB, build ≤ 60 min |
| Rules | don't detect the eval environment, read nothing but `/data/images`, only the container's output counts |

## 7. What actually wins

Score is mechanical, but the organizer said twice that the **difference between teams is the
approach**:

- "try to understand the idea, come up with your own creativity, and **document that**"
- "the difference will be how you approached the problem and how the performance is"
- "I expect you all to fix everything for the others" — he wants to see *how* problems get solved.
- Explicitly asked for **intermediate results** ("show me some intermediate results"), a
  **reasoning trail** — Samuel's phrasing "architectural decision record" was not contradicted.
- Explicitly against blind delegation: "solve each problem one by one instead of just AI going and
  fixing this"; "minimal vibe coding".
- Runtime and image size are shown next to the ranking; among similar scores the leaner pipeline
  wins the informal comparison.

⇒ Deliverables: score **and** a readable decision log (what failed, why, with numbers), error
analysis, and a container that builds first try.

## 8. Hidden-set traps

From the transcript (README says ~96 hidden tables, same generator, private seed, all tiers/styles
equally; dev = 24):

- Don't overfit the 24 dev tables — the hidden set is new tables from the same generator.
- "Some other surprises": at least one input may be a **PDF**, and at least one may have **text
  around the table** — so page/region detection may be needed, and the input reader must not assume
  "PNG/JPG containing only a table".
- Content is text-only tables: no graphics, no paragraphs as content — but real-world-ish layouts,
  variable column structure, parent/group columns.
- Number format must be normalized; format is consistent *within* a table (usable for post-correction).

## 9. Open questions / to verify in the repo

1. Are hidden inputs really PDFs, or is that only "in one example"? (`predict.py` must handle both
   without crashing — cheap insurance.)
2. Does "text around the table" mean extra cells must be excluded? TEDS compares one table only.
3. Exact hand-in: private repo from template, invite `metinferhat@outlook.com`, register team name.
4. Codex/credits promised "tomorrow" — confirm we actually have them.
5. Team composition still open (an "Ahmed" did not show up; decision was pending within the hour).
6. Multi-page PDFs: one table per file, or several? Affects the output contract.

## 10. Decisions we have to make (and record)

1. **Structure strategy:** classic CV line/whitespace detection vs. a table-structure model
   (Table Transformer / PaddleOCR table pipeline) vs. hybrid (detect structure, OCR cell crops).
2. **Text strategy:** Tesseract vs. PaddleOCR vs. a small document VLM — within 8 GB VRAM and 15 s/table.
3. **Pre-processing:** how much deskew/perspective/illumination correction is worth it before OCR.
4. **Post-processing:** cross-cell consistency — one number format per table, span repair, empty-cell
   completion.
5. **Training:** fine-tune on generated data (`generate_dataset.py`, unlimited and exact) or stay
   zero-shot; if fine-tune, where the weights are pinned.
6. **Output:** JSON only (simpler, same score) — default unless a reason appears.
7. **Fallback ladder:** what the container does when a model is slow or an image is unreadable
   (must never produce zero over a whole table if a cheap fallback exists).

Each decision → one ADR line in [PLAN.md](PLAN.md) §Decision log: what we chose, what we rejected,
what the number was.
