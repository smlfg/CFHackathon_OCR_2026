#!/usr/bin/env python3
"""Download small real-table samples and convert them to the TableSalt cell format.

Output lives under eval/real/samples/<dataset>/ (git-ignored, never committed).
ICDAR 2019 cTDaR was evaluated once locally but is not downloadable from here: its license is
not clearly permissive (see README.md).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

# Local imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from loaders.common import (  # noqa: E402
    grid_stats,
    html_tokens_to_cells,
    icdar2013_voc_to_cells,
    safe_id,
    scitsr_to_cells,
    write_manifest,
    write_sample,
    write_source,
)

SAMPLES = ROOT / "samples"
RAW = ROOT / "raw"
os.environ.setdefault("HF_DATASETS_CACHE", str(RAW / "hf_cache"))
os.environ.setdefault("HF_HOME", str(RAW / "hf_home"))

# Keep tables small enough for a CPU Tesseract pass.
MAX_ROWS = 16
MAX_COLS = 10
MAX_CELLS = 80
MIN_CELLS = 4
MAX_SIDE_IN = 2500  # skip huge uncropped pages before we resize


def _accept_cells(cells: list[dict], img_size=None) -> bool:
    st = grid_stats(cells)
    if st["n_cells"] < MIN_CELLS or st["n_cells"] > MAX_CELLS:
        return False
    if st["n_rows"] < 2 or st["n_rows"] > MAX_ROWS:
        return False
    if st["n_cols"] < 2 or st["n_cols"] > MAX_COLS:
        return False
    if img_size is not None:
        w, h = img_size
        if max(w, h) > 8000:
            return False
    return True


def _pil_from_hf(image):
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode not in ("RGB", "L") else image
    return Image.open(io.BytesIO(image)).convert("RGB")


def prepare_otsl(name: str, repo: str, split: str, n: int, style: str, source: dict) -> int:
    """Stream a Docling OTSL dataset and keep up to n acceptable tables."""
    from datasets import load_dataset

    out = SAMPLES / name
    if (out / "manifest.json").exists() and len(
        json.loads((out / "manifest.json").read_text())
    ) >= min(n, 5):
        k = len(json.loads((out / "manifest.json").read_text()))
        print(f"[skip] {name}: already has {k} samples")
        return k

    print(f"[{name}] streaming {repo} split={split} ...")
    ds = load_dataset(repo, split=split, streaming=True)
    entries = []
    seen = 0
    kept_span = 0
    t0 = time.time()
    try:
        for row in ds:
            seen += 1
            try:
                rows, cols = int(row["rows"]), int(row["cols"])
                if not (2 <= rows <= MAX_ROWS and 2 <= cols <= MAX_COLS):
                    continue
                img = _pil_from_hf(row["image"])
                cells = html_tokens_to_cells(row["html"], row["cells"])
                if not _accept_cells(cells, img.size):
                    continue
                # Prefer a mix: first half any, later prefer spanning if we have few.
                st = grid_stats(cells)
                if kept_span < max(3, n // 4) and st["n_spans"] == 0 and len(entries) >= n // 2:
                    # keep looking a bit for spans, but don't stall forever
                    if seen < 400 or len(entries) < n:
                        if st["n_spans"] == 0 and len(entries) >= n:
                            continue
                tid = safe_id(Path(str(row.get("filename") or f"{name}_{seen}")).stem, maxlen=70)
                if any(e["id"] == tid for e in entries):
                    tid = f"{tid}_{seen}"
                extra = {
                    "source_filename": row.get("filename"),
                    "hf_split": split,
                    "imgid": row.get("imgid"),
                }
                entry = write_sample(out, tid, img, cells, style=style, extra=extra)
                entries.append(entry)
                if entry["has_spans"]:
                    kept_span += 1
                print(
                    f"  kept {len(entries)}/{n}  {tid}  {entry['n_rows']}x{entry['n_cols']} "
                    f"spans={entry['has_spans']}  (seen {seen})"
                )
                if len(entries) >= n:
                    break
            except Exception as exc:
                print(f"  skip row {seen}: {exc}")
            if seen >= 800:
                print(f"  stop searching after {seen} rows, kept {len(entries)}")
                break
    finally:
        del ds

    write_manifest(out, entries)
    source = dict(source)
    source.update(
        {
            "split": split,
            "requested_n": n,
            "kept": len(entries),
            "rows_scanned": seen,
            "seconds": round(time.time() - t0, 1),
            "with_spans": kept_span,
        }
    )
    write_source(out, source)
    print(f"[{name}] wrote {len(entries)} samples in {time.time() - t0:.1f}s")
    return len(entries)


def prepare_scitsr(n: int) -> int:
    from datasets import load_dataset

    out = SAMPLES / "scitsr_pd"
    print("[scitsr_pd] loading local parquet...")
    data_files = {
        "train": str(RAW / "SciTSR-pd/data/train-00000-of-00001.parquet"),
        "test": str(RAW / "SciTSR-pd/data/test-00000-of-00001.parquet"),
    }
    ds = load_dataset("parquet", data_files=data_files)
    entries = []
    # Prefer the official test split first, then fill from train.
    for split in ("test", "train"):
        for row in ds[split]:
            try:
                cells = scitsr_to_cells(row["cells"])
                img = _pil_from_hf(row["image"])
                if not _accept_cells(cells, img.size):
                    continue
                tid = safe_id(row["table_id"], maxlen=70)
                extra = {
                    "source_table_id": row["table_id"],
                    "paper_id": row.get("paper_id"),
                    "paper_license": row.get("paper_license"),
                    "is_comp": bool(row.get("is_comp")),
                    "hf_split": split,
                }
                entries.append(
                    write_sample(out, tid, img, cells, style="scientific_latex", extra=extra)
                )
                print(
                    f"  kept {len(entries)}/{n}  {tid}  {entries[-1]['n_rows']}x{entries[-1]['n_cols']}"
                )
                if len(entries) >= n:
                    break
            except Exception as exc:
                print(f"  skip {row.get('table_id')}: {exc}")
        if len(entries) >= n:
            break
    write_manifest(out, entries)
    write_source(
        out,
        {
            "name": "SciTSR-PD",
            "link": "https://huggingface.co/datasets/bevaya/SciTSR-pd",
            "original": "https://github.com/Academic-Hammer/SciTSR",
            "paper": "https://arxiv.org/abs/1908.04729",
            "license": "CC0-1.0 (subset of papers with CC0 / public-domain dedication)",
            "domain": "scientific tables rendered from arXiv LaTeX/PDF",
            "real_or_degraded": "real digital-born scientific tables (PDF renders, not photos)",
            "notes": "Annotations come from the original SciTSR LaTeX pipeline and can be noisy on spanning cells.",
            "kept": len(entries),
            "requested_n": n,
        },
    )
    print(f"[scitsr_pd] wrote {len(entries)} samples")
    return len(entries)


def prepare_icdar2013(n: int) -> int:
    import json as _json

    from PIL import Image

    base = RAW / "ICDAR-2013.c/extracted/ICDAR-2013.c-Structure"
    out = SAMPLES / "icdar2013"
    entries = []
    # val first (official-ish split in the .c release), then test if needed
    for split in ("val", "test"):
        xmls = sorted((base / split).glob("*.xml"))
        for xml_path in xmls:
            stem = xml_path.stem
            img_path = base / "images" / f"{stem}.jpg"
            words_path = base / "words" / f"{stem}_words.json"
            if not img_path.exists() or not words_path.exists():
                continue
            try:
                words = _json.loads(words_path.read_text(encoding="utf-8"))
                if isinstance(words, dict) and "words" in words:
                    words = words["words"]
                cells = icdar2013_voc_to_cells(xml_path, words)
                img = Image.open(img_path)
                if not _accept_cells(cells, img.size):
                    continue
                tid = safe_id(stem, maxlen=70)
                extra = {"source_split": split, "source_file": stem}
                entries.append(write_sample(out, tid, img, cells, style="scanned_gov", extra=extra))
                print(
                    f"  kept {len(entries)}/{n}  {tid}  {entries[-1]['n_rows']}x{entries[-1]['n_cols']} "
                    f"spans={entries[-1]['has_spans']}"
                )
                if len(entries) >= n:
                    break
            except Exception as exc:
                print(f"  skip {stem}: {exc}")
        if len(entries) >= n:
            break
    write_manifest(out, entries)
    write_source(
        out,
        {
            "name": "ICDAR-2013.c",
            "link": "https://huggingface.co/datasets/bsmock/ICDAR-2013.c",
            "original": "ICDAR 2013 Table Competition (EU + US government documents)",
            "paper": "https://arxiv.org/abs/2303.00716",
            "license": "CDLA-Permissive-2.0 for this corrected release; original documents are described as public-domain (EU/US government)",
            "domain": "scanned / born-digital official government reports (EU and US)",
            "real_or_degraded": "real scanned or official documents; not synthetic",
            "notes": "Cell text reconstructed by assigning word boxes to the canonical row/column/span grid.",
            "kept": len(entries),
            "requested_n": n,
        },
    )
    print(f"[icdar2013] wrote {len(entries)} samples")
    return len(entries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="target sample size per dataset")
    ap.add_argument(
        "--only", nargs="*", default=None, help="subset: pubtabnet fintabnet icdar2013 scitsr"
    )
    args = ap.parse_args()
    n = args.n
    want = set(args.only) if args.only else {"pubtabnet", "fintabnet", "icdar2013", "scitsr"}

    counts = {}
    if "pubtabnet" in want:
        counts["pubtabnet"] = prepare_otsl(
            "pubtabnet",
            "docling-project/PubTabNet_OTSL",
            "val",
            n,
            style="scientific_pdf",
            source={
                "name": "PubTabNet (OTSL conversion)",
                "link": "https://huggingface.co/datasets/docling-project/PubTabNet_OTSL",
                "original": "https://huggingface.co/datasets/ajimeno/PubTabNet",
                "paper": "https://arxiv.org/abs/1911.10683",
                "license": "Original PubTabNet: CDLA-Sharing-1.0. OTSL conversion on HF is tagged license=other.",
                "domain": "scientific journal tables from PubMed Central OA",
                "real_or_degraded": "real digital-born PDF table crops (clean renders, not camera photos)",
            },
        )
    if "fintabnet" in want:
        counts["fintabnet"] = prepare_otsl(
            "fintabnet",
            "docling-project/FinTabNet_OTSL",
            "test",
            n,
            style="financial_pdf",
            source={
                "name": "FinTabNet (OTSL conversion)",
                "link": "https://huggingface.co/datasets/docling-project/FinTabNet_OTSL",
                "original": "https://developer.ibm.com/exchanges/data/all/fintabnet/",
                "paper": "https://arxiv.org/abs/2305.03393 (OTSL); original FinTabNet IBM Research",
                "license": "Original FinTabNet: CDLA-Permissive-1.0. OTSL conversion on HF is tagged license=other.",
                "domain": "financial reports / SEC-style tables",
                "real_or_degraded": "real digital-born financial PDF table crops (not photos)",
            },
        )
    if "icdar2013" in want:
        counts["icdar2013"] = prepare_icdar2013(n)
    if "scitsr" in want:
        counts["scitsr"] = prepare_scitsr(n)

    print("=== summary ===")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
