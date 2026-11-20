# Challenge preparation — TableSalt / Lerai

Our own folder for the challenge. Everything that is *thinking* lives here; the repo root stays the
organizer's kit (generator, scorer, baselines, template).

| File | What it is |
|---|---|
| [PROBLEM.md](PROBLEM.md) | the problem, structured: business context, the three difficulty axes, I/O contract, scoring, constraints, traps, open questions, decisions |
| [TRANSCRIPT.md](TRANSCRIPT.md) | cleaned kick-off transcript (organizer + Lerai co-founder), with a table of likely mistranscriptions |
| [transcript.raw.txt](transcript.raw.txt) | untouched copy of the original `Transribe` file |
| [PLAN.md](PLAN.md) | milestone plan M0–M5, candidate architectures, commands, risk register, definition of done |
| [MEASUREMENTS.md](MEASUREMENTS.md) | every run with numbers (fill in from M0 on) |
| [ERRORS.md](ERRORS.md) | error classes per tier/style with examples (fill in from M1 on) |
| [DECISIONS.md](DECISIONS.md) | ADRs — one per real decision, with evidence |

## The short version

Build `table image → cells (text, r, c, rs, cs)`, offline, in Docker, on a 4070 Laptop with 8 GB
VRAM and 15 s per table. Score = 0.85 × tier-weighted TEDS (T2+T3 = 0.7 of it) + 0.15 × QA.

Floor is the shipped Tesseract baseline at **0.348** (dev). Everything above that has to be earned
on degraded images, on structure (spans, empty cells) and on number formats — and it has to be
documented, because the organizer said the reasoning is what separates the teams.

## Reading order for a new teammate

1. [PROBLEM.md](PROBLEM.md) — 5 minutes, tells you what is being asked and what is not.
2. `../README.md` §5–§8 — the exact output format, scoring, submission contract.
3. [PLAN.md](PLAN.md) — where we are and what the next step is.
4. [TRANSCRIPT.md](TRANSCRIPT.md) — only if you want the atmosphere and the unstated expectations
   (intermediate results, ADRs, "minimal vibe coding", hidden-set surprises).
