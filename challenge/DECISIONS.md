# Decisions (ADRs)

Short by design: context, options, choice, evidence, revisit-if. One file so a reviewer can read the
reasoning in one place — the organizer explicitly asked for it.

Template:

```
## ADR-NNN — <title>                    (YYYY-MM-DD, status: proposed|accepted|superseded)
Context:   what forced the decision
Options:   A / B / C, each with what it would cost
Choice:    X
Evidence:  numbers (dev/val, s/table), error classes fixed, or "measured, no gain"
Revisit:   the condition under which this becomes wrong
```

## Open decisions (from [PROBLEM.md](PROBLEM.md) §10)

| # | Question | Leaning | Status |
|---|---|---|---|
| 1 | Structure: classic CV vs. table-structure model vs. hybrid | measure candidate 2 before deciding | open |
| 2 | Text: Tesseract vs. PaddleOCR vs. small VLM | OCR before VLM (time budget) | open |
| 3 | Preprocessing: how much correction is worth it | on/off module, measure | open |
| 4 | Post-processing: per-table number format, span repair, empty cells | do it — cheapest score | open |
| 5 | Fine-tune on generated data? | only after a real baseline number | open |
| 6 | Output JSON vs HTML | JSON | open |
| 7 | Fallback ladder when a model is slow/unreadable | never zero cells | open |

## Accepted
