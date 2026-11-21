"""CLI and orchestration.

    python -m tablesalt.run --images DIR --out DIR [--workers N] [--debug DIR]

Every table is written as soon as it is done.  Per table: full pipeline -> on exception or
an implausible/invalid result the Tesseract floor baseline -> at worst ``[]``.  A crash in
one table never stops the run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_THREAD_LIMIT", "1")

from . import ocr, postprocess, preprocess, spans, structure, validate  # noqa: E402
from .config import CFG, Config  # noqa: E402
from .io_utils import list_images, load_gray, write_json_atomic  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------
# one table
# ----------------------------------------------------------------------------

def predict(img_path: Path, cfg: Config = CFG, info: dict | None = None) -> list[dict]:
    """Image -> cell list.  Raises on failure (the caller falls back)."""
    t0 = time.perf_counter()
    info = info if info is not None else {}
    prep = preprocess.prepare(load_gray(img_path), cfg)
    lay = structure.analyze(prep.binary, cfg)
    if not (cfg.min_cols <= lay.grid.n_cols <= cfg.max_cols):
        raise ValueError(f"implausible column count {lay.grid.n_cols}")
    rules = (lay.hline_mask > 0) | (lay.vline_mask > 0)
    cands = ocr.read_variants(prep.gray, lay.blobs, lay.h, rules_mask=rules, cfg=cfg)
    best = [max(c, key=lambda tc: tc[1]) for c in cands]
    texts = [t for t, _ in best]
    confs = [c for _, c in best]

    cells, hr = spans.build_cells(lay, texts, cfg)
    for c in cells:
        c.neg_hint = any(lay.blobs[i].lead_dash for i in c.blobs)

    # second pass: numeric whitelist for body value blobs whose first read looks numeric
    if cfg.second_pass and time.perf_counter() - t0 < cfg.table_budget_s:
        cand = sorted({i for c in cells if c.r >= hr and c.c >= 1 and c.cs == 1
                       for i in c.blobs if any(ch.isdigit() for ch in texts[i])})
        if cand:
            m, md, psm = cfg.ocr_variants[-1]
            res2 = ocr.read_blobs(prep.gray, lay.blobs, lay.h, idx=cand,
                                  whitelist=cfg.num_whitelist, rules_mask=rules,
                                  cfg=replace(cfg, ocr_margin=m, ocr_median=md, tess_psm=psm))
            for i, r2 in zip(cand, res2):      # one more candidate per numeric blob
                cands[i] = cands[i] + [r2]
            for i in range(len(cands)):        # keep the candidate lists rectangular
                if len(cands[i]) < len(cfg.ocr_variants) + 1:
                    cands[i] = cands[i] + [cands[i][0]]

    cells = postprocess.fix_table(cells, hr, lay.grid.n_cols, cfg, cands=cands,
                                  blob_x=[b.x0 for b in lay.blobs])
    out = [c.to_json() for c in sorted(cells, key=lambda c: (c.r, c.c))]
    problems = validate.check(out)
    if problems:
        raise ValueError("invalid cells: " + "; ".join(problems[:3]))
    info.update(style=lay.extra.get("style"), rows=lay.grid.n_rows, cols=lay.grid.n_cols,
                hr=hr, angle=prep.angle, homography=prep.homography,
                col_src=lay.extra.get("col_src"))
    return out


# ----------------------------------------------------------------------------
# fallback: the Tesseract floor baseline
# ----------------------------------------------------------------------------

def _baseline():
    p = ROOT / "baselines" / "baseline_tesseract.py"
    spec = importlib.util.spec_from_file_location("baseline_tesseract", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def baseline_cells(img_path: Path) -> list[dict]:
    import html as _html
    import re
    table = _baseline().to_html(img_path)
    cells = []
    for r, row in enumerate(re.findall(r"<tr>(.*?)</tr>", table, flags=re.S)):
        for c, td in enumerate(re.findall(r"<td>(.*?)</td>", row, flags=re.S)):
            cells.append({"text": _html.unescape(td), "r": r, "c": c, "rs": 1, "cs": 1})
    return cells


# ----------------------------------------------------------------------------
# batch
# ----------------------------------------------------------------------------

def process_one(args: tuple[str, str, bool]) -> dict:
    img, out_dir, verbose = args
    img_path, out_dir = Path(img), Path(out_dir)
    t = time.perf_counter()
    info: dict = {"id": img_path.stem}
    try:
        cells = predict(img_path, CFG, info)
        info["mode"] = "pipeline"
    except Exception as e:                      # noqa: BLE001 - never let one table kill the run
        info["error"] = f"{type(e).__name__}: {e}"
        if verbose:
            traceback.print_exc()
        try:
            cells = baseline_cells(img_path)
            info["mode"] = "baseline"
        except Exception as e2:                 # noqa: BLE001
            info["error"] += f" | baseline: {type(e2).__name__}: {e2}"
            cells, info["mode"] = [], "empty"
    write_json_atomic(out_dir / f"{img_path.stem}.json", cells)
    info["seconds"] = round(time.perf_counter() - t, 3)
    return info


def main(images_dir, out_dir, workers: int | None = None, verbose: bool = False,
         log: str | None = None) -> list[dict]:
    images = list_images(Path(images_dir))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, min(len(images), (os.cpu_count() or 2)))
    jobs = [(str(p), str(out_dir), verbose) for p in images]
    infos = []
    t0 = time.perf_counter()
    if workers == 1:
        it = map(process_one, jobs)
        for info in it:
            infos.append(_report(info))
    else:
        with Pool(workers) as pool:
            for info in pool.imap_unordered(process_one, jobs):
                infos.append(_report(info))
    total = time.perf_counter() - t0
    n_fb = sum(1 for i in infos if i.get("mode") != "pipeline")
    print(f"{len(infos)} tables in {total:.1f}s ({total / max(1, len(infos)):.2f}s/table wall, "
          f"{workers} workers), fallbacks: {n_fb}", flush=True)
    if log:
        Path(log).write_text(json.dumps(sorted(infos, key=lambda i: i["id"]), indent=1),
                             encoding="utf-8")
    return infos


def _report(info: dict) -> dict:
    flag = "" if info.get("mode") == "pipeline" else f"  [{info.get('mode')}] {info.get('error')}"
    print(f"{info['id']} done {info['seconds']:.2f}s{flag}", flush=True)
    return info


def cli(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--log", default=None, help="write per-table info (mode, time, ...) as JSON")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    main(a.images, a.out, a.workers, a.verbose, a.log)


if __name__ == "__main__":
    sys.exit(cli())
