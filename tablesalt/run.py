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
import tempfile
import time
import traceback
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_THREAD_LIMIT", "1")

from . import ocr, postprocess, preprocess, selfcheck, spans, structure, validate
from .config import CFG, Config
from .io_utils import is_pdf, list_images, load_gray, render_pdf_page, write_json_atomic

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
        cand = sorted(
            {
                i
                for c in cells
                if c.r >= hr and c.c >= 1 and c.cs == 1
                for i in c.blobs
                if any(ch.isdigit() for ch in texts[i])
            }
        )
        if cand:
            m, md, psm = cfg.ocr_variants[-1]
            res2 = ocr.read_blobs(
                prep.gray,
                lay.blobs,
                lay.h,
                idx=cand,
                whitelist=cfg.num_whitelist,
                rules_mask=rules,
                cfg=replace(cfg, ocr_margin=m, ocr_median=md, tess_psm=psm),
            )
            for i, r2 in zip(cand, res2):  # one more candidate per numeric blob
                cands[i] = cands[i] + [r2]
            for i in range(len(cands)):  # keep the candidate lists rectangular
                if len(cands[i]) < len(cfg.ocr_variants) + 1:
                    cands[i] = cands[i] + [cands[i][0]]

    cells = postprocess.fix_table(
        cells, hr, lay.grid.n_cols, cfg, cands=cands, blob_x=[b.x0 for b in lay.blobs]
    )
    out = [c.to_json() for c in sorted(cells, key=lambda c: (c.r, c.c))]
    problems = validate.check(out)
    if problems:
        raise ValueError("invalid cells: " + "; ".join(problems[:3]))
    info.update(
        style=lay.extra.get("style"),
        rows=lay.grid.n_rows,
        cols=lay.grid.n_cols,
        hr=hr,
        angle=prep.angle,
        homography=prep.homography,
        col_src=lay.extra.get("col_src"),
    )
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

# second opinion: only tables the self-check flags are read again, on the CPU, with
# other settings; the read with fewer problems wins.  Costs nothing for clean tables.
SECOND_OPINIONS = [  # layout settings; tuned on fresh tables, never on dev
    dict(row_split=0.7),  # looser rows: keeps a two-row header together
    dict(edge_support=0.5),  # stricter column edges: splits headers like "Q1 Q2"
    dict(blob_join_x=1.1),
    dict(ocr_cap_height=48, second_pass=True),
]


def predict_checked(img_path: Path, cfg: Config = CFG, info: dict | None = None) -> list[dict]:
    info = info if info is not None else {}
    t0 = time.perf_counter()
    best, err = None, None
    try:
        best = predict(img_path, cfg, info)
    except Exception as e:
        err = e
    probs = selfcheck.check(best) if best is not None else ["failed"]
    info["check"] = probs[:3]
    if not probs or os.environ.get("TABLESALT_SECOND_OPINION", "1") == "0":
        if best is None:
            raise err
        return best
    one = time.perf_counter() - t0  # what one read costs on this machine right now
    deadline = float(os.environ.get("TABLESALT_DEADLINE_S", "12"))  # hard limit is 15 s
    for k, alt in enumerate(SECOND_OPINIONS):
        if time.perf_counter() - t0 + 1.3 * one > deadline:
            info["second_opinion_skipped"] = len(SECOND_OPINIONS) - k
            break
        try:
            cand = predict(img_path, replace(cfg, **alt), {})
        except Exception:
            continue
        p2 = selfcheck.check(cand)
        if len(p2) < len(probs):
            best, probs = cand, p2
            info["second_opinion"] = k
        if not probs:
            break
    info["check_after"] = probs[:3]
    if best is None:
        raise err
    return best


# Written when the pipeline *and* the Tesseract floor both fail.  One empty cell instead of
# ``[]``: the file still parses as a table and is never an empty prediction (ADR-008).
LAST_RESORT: list[dict] = [{"text": "", "r": 0, "c": 0, "rs": 1, "cs": 1}]


def process_one(args: tuple[str, str, bool]) -> dict:
    """Predict one input file and write ``<out_dir>/<stem>.json``; never raises."""
    img, out_dir, verbose = args
    src, out_dir = Path(img), Path(out_dir)
    t = time.perf_counter()
    info: dict = {"id": src.stem}
    with tempfile.TemporaryDirectory(prefix="tablesalt_") as tmp:
        cells = _predict_with_fallbacks(src, Path(tmp), info, verbose)
    if ocr.TESS_ERRORS:
        info["tesseract_errors"] = len(ocr.TESS_ERRORS)
        ocr.TESS_ERRORS.clear()
    write_json_atomic(out_dir / f"{src.stem}.json", cells)
    info["seconds"] = round(time.perf_counter() - t, 3)
    return info


def _predict_with_fallbacks(src: Path, tmp: Path, info: dict, verbose: bool) -> list[dict]:
    """Fallback ladder: pipeline -> Tesseract floor -> one empty cell."""
    img_path = src
    try:
        if is_pdf(src):
            img_path = render_pdf_page(src, tmp / f"{src.stem}.png")
            info["pdf"] = True
        cells = predict_checked(img_path, CFG, info)
        info["mode"] = "pipeline"
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        if verbose:
            traceback.print_exc()
        try:
            cells = baseline_cells(img_path)
            info["mode"] = "baseline"
        except Exception as e2:
            info["error"] += f" | baseline: {type(e2).__name__}: {e2}"
            cells = []
        if not cells:
            cells, info["mode"] = [dict(c) for c in LAST_RESORT], "empty"
    return cells


def main(
    images_dir, out_dir, workers: int | None = None, verbose: bool = False, log: str | None = None
) -> list[dict]:
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
    print(
        f"{len(infos)} tables in {total:.1f}s ({total / max(1, len(infos)):.2f}s/table wall, "
        f"{workers} workers), fallbacks: {n_fb}, flagged: {sum(1 for i in infos if i.get('check'))}, "
        f"fixed by 2nd read: {sum(1 for i in infos if i.get('check') and not i.get('check_after'))}, "
        f"tesseract errors: {sum(i.get('tesseract_errors', 0) for i in infos)}",
        flush=True,
    )
    if log:
        Path(log).write_text(
            json.dumps(sorted(infos, key=lambda i: i["id"]), indent=1), encoding="utf-8"
        )
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
