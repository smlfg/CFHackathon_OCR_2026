# External real-data evaluation (ADR-017)

This folder asks one question: how does the submission pipeline (`tablesalt/`, ~0.997 on the
synthetic dev set) do on **real** public tables? The answer is far worse. Samuel's target was
**≥ 0.99 mean TEDS on five new real datasets**, and that target is **open**: 0 of 4 text-GT
datasets reach it.

The numbers come from `main` @ `973cc2d`, CPU Tesseract, 2 workers, measured 2026-11-22. The
same code is on this branch: the branch does not change any prediction (see the PR). Scores
are unweighted means from the repo's own `evaluate.py`. The challenge's T0–T3 weights do not
exist on external data.

| Dataset | n | Source | License | Domain | TEDS | TEDS-S | Cell-F1 |
|---|---:|---|---|---|---:|---:|---:|
| SciTSR-PD | 30 | [bevaya/SciTSR-pd](https://huggingface.co/datasets/bevaya/SciTSR-pd) | CC0-1.0 | arXiv tables, PDF renders | 0.5166 | 0.6529 | 0.1756 |
| FinTabNet (OTSL) | 30 | [docling-project/FinTabNet_OTSL](https://huggingface.co/datasets/docling-project/FinTabNet_OTSL) | orig. CDLA-Permissive-1.0 (HF copy tagged `other`) | financial PDF tables | 0.4703 | 0.5801 | 0.1343 |
| ICDAR-2013.c | 30 | [bsmock/ICDAR-2013.c](https://huggingface.co/datasets/bsmock/ICDAR-2013.c) | CDLA-Permissive-2.0 | EU/US government documents | 0.3192 | 0.4493 | 0.0737 |
| PubTabNet (OTSL) | 30 | [docling-project/PubTabNet_OTSL](https://huggingface.co/datasets/docling-project/PubTabNet_OTSL) | orig. CDLA-Sharing-1.0 (share-alike) | PubMed tables, PDF renders | 0.2590 | 0.4548 | 0.0675 |
| ICDAR 2019 cTDaR B.1 (historical) | 18 | [cndplab-founder/ICDAR2019_cTDaR](https://github.com/cndplab-founder/ICDAR2019_cTDaR) | **not clearly permissive** (© PKU Founder Group, all rights reserved) | degraded archival scans | n/a* | **0.2458** | n/a* |

\* cTDaR has structure ground truth but **no cell text**, so only TEDS-S is meaningful. Because of
its license, **no cTDaR data and no cTDaR download code are in this repository**. Only the score
row is kept. It was measured once on a local sample that was not redistributed.

**Speed:** about 1.1–1.4 s/table (152 s for the 120 text-GT tables), so speed is not the problem;
accuracy is. **Best single tables:**

- FinTabNet `HAL.2014.page_42.pdf_125091`: TEDS 0.9516.
- SciTSR-PD `1606.09371v1.5`: TEDS 1.0000.
- ICDAR-2013.c `us-gov-dataset_us-004_table_0`: TEDS 0.9810.

These are easy individual grids. The dataset means stay between 0.26 and 0.52.

## Why the gap

The pipeline is built on generator knowledge: a 20 px font, right-aligned numbers, fixed word
lists for lexicon snapping, and a fixed number format per table (ADR-005, ADR-009). Real tables
break each of those assumptions:

- denser typography, multi-row headers, indented financial labels, scientific notation;
- scanned official documents (ICDAR-2013) and degraded archival pages (cTDaR);
- Cell-F1 needs exact text at the exact grid position, so one OCR typo or one shifted column
  zeroes the match. TEDS-S (structure alone) is only 0.45–0.65, so structure is already wrong
  before text is considered.

Headline numbers: [reports/summary.json](reports/summary.json). Per-table reports and predict logs
are regenerated locally in `reports/` and are git-ignored.

## Re-run (downloads are not committed)

```bash
pip install -r requirements.txt datasets huggingface_hub   # + tesseract-ocr (eng, deu)
cd eval/real
python prepare_all.py --n 30            # streams the 4 permissive sets into samples/ (git-ignored)
python run_real_eval.py --workers 2     # predicts with this repo's pipeline, writes reports/
python run_real_eval.py --skip-predict  # re-score existing predictions only
```

`prepare_all.py` filters each stream to tables with roughly ≤ 16 rows, ≤ 10 columns and ≤ 80
cells, taking the first acceptable ones. That is **not a random sample**, and it leaves out the
hardest, largest tables.

- PubTabNet/FinTabNet: text is rebuilt from OTSL tokens.
- ICDAR-2013.c: text is rebuilt by placing the official word boxes onto the grid.
- SciTSR: annotations are generated automatically from LaTeX and can be noisy on spans.

## Not included

- **TabRecSet**: its images are a 2.1 GB archive or a gated copy.
- **WTW**: no text ground truth.
- **PubTables-1M**: same domain as PubTabNet.
- **Full SciTSR**: arXiv license; we used only the CC0 subset.
- **A German Finanzamt scan set** with structure and text ground truth: none found.
