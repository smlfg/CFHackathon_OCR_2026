"""Leaderboard output: LEADERBOARD.md, metrics/leaderboard.html, per-run detail pages."""

from __future__ import annotations

import html as htmllib
import json
from pathlib import Path

from . import config, dataset, store

COLUMNS = ["final", "teds_weighted", "qa_accuracy", "cell_f1", "t0", "t1", "t2", "t3"]


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render(dataset_name: str | None = None) -> dict:
    sets = (
        [dataset_name]
        if dataset_name
        else [s for s in ("dev", "val") if config.dataset_dir(s).exists()]
    )
    written = {}
    for name in sets:
        all_rows = store.runs(name)
        rows = [r for r in all_rows if r["status"] == "ok" and not r.get("adapter_flag")]
        skipped = [r for r in all_rows if r.get("adapter_flag")]
        unrun = [r for r in all_rows if r["status"] not in ("ok",)]
        written[name] = {
            "md": _write_md(name, rows, skipped, unrun),
            "html": _write_html(name, rows, skipped, unrun),
        }
        for row in all_rows:
            if row["status"] == "ok":
                _write_detail(row)
    return written


def _write_md(
    name: str, rows: list[dict], skipped: list[dict], unrun: list[dict] | None = None
) -> Path:
    lines = [
        f"# TableSalt OCR leaderboard — {name}",
        "",
        f"{len(rows)} scored models · dataset `{name}` · official scorer "
        "(`score_all.py`), final = 0.85 × tier-weighted TEDS + 0.15 × QA.",
        "",
        "| # | Model | Params | Quant | fmt | T0 | T1 | T2 | T3 | QA | CellF1 | FINAL | s/table |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(rows, 1):
        lines.append(
            f"| {i} | {row['model_name']} | {_fmt(row['params_b'], 2)}B | {row['quant'] or '–'} "
            f"| {json.loads(row['meta'] or '{}').get('fmt') or '–'} "
            f"| {_fmt(row['t0'], 3)} | {_fmt(row['t1'], 3)} | {_fmt(row['t2'], 3)} "
            f"| {_fmt(row['t3'], 3)} | {_fmt(row['qa_accuracy'], 3)} "
            f"| {_fmt(row['cell_f1'], 3)} | **{_fmt(row['final'], 4)}** "
            f"| {_fmt(row['s_per_table'], 1)} |"
        )
    if skipped:
        lines += ["", "## Excluded (adapter output did not parse)", ""]
        for row in skipped:
            lines.append(
                f"- {row['model_name']} — {row['adapter_flag']}, "
                f"empty_rate {_fmt(row['empty_rate'], 2)} "
                f"([detail](runs/{row['run_id']}.html))"
            )
    if unrun:
        lines += ["", "## Not scored", ""]
        for row in unrun:
            note = (json.loads(row["meta"] or "{}").get("note") or "").strip()
            lines.append(f"- {row['model_name']} — {row['status']}" + (f": {note}" if note else ""))
    path = config.METRICS / "LEADERBOARD.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_html(
    name: str, rows: list[dict], skipped: list[dict], unrun: list[dict] | None = None
) -> Path:
    body = []
    for i, row in enumerate(rows, 1):
        meta = json.loads(row["meta"] or "{}")
        body.append(
            "<tr>"
            f"<td class=r>{i}</td><td><a href='runs/{row['run_id']}.html'>"
            f"{htmllib.escape(row['model_name'])}</a></td>"
            f"<td class=r>{_fmt(row['params_b'], 2)}</td><td>{row['quant'] or '–'}</td>"
            f"<td>{meta.get('fmt') or '–'}</td>"
            f"<td class=r>{_fmt(row['t0'], 3)}</td><td class=r>{_fmt(row['t1'], 3)}</td>"
            f"<td class=r>{_fmt(row['t2'], 3)}</td><td class=r>{_fmt(row['t3'], 3)}</td>"
            f"<td class=r>{_fmt(row['qa_accuracy'], 3)}</td>"
            f"<td class=r>{_fmt(row['cell_f1'], 3)}</td>"
            f"<td class='r strong'>{_fmt(row['final'], 4)}</td>"
            f"<td class=r>{_fmt(row['s_per_table'], 1)}</td>"
            "</tr>"
        )
    excluded = "".join(
        f"<li>{htmllib.escape(r['model_name'])} — {r['adapter_flag']}, "
        f"empty_rate {_fmt(r['empty_rate'], 2)} "
        f"(<a href='runs/{r['run_id']}.html'>detail</a>)</li>"
        for r in skipped
    )
    not_scored = "".join(
        f"<li>{htmllib.escape(r['model_name'])} — {r['status']}"
        + (
            f": {htmllib.escape(json.loads(r['meta'] or '{}').get('note') or '')}"
            if json.loads(r["meta"] or "{}").get("note")
            else ""
        )
        + "</li>"
        for r in (unrun or [])
    )
    html = f"""<!doctype html><meta charset=utf-8><title>TableSalt OCR leaderboard</title>
<style>
 body{{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:2rem;color:#111;background:#fff}}
 h1{{margin:0 0 .25rem;font-size:1.5rem}} p.sub{{margin:0 0 1.5rem;color:#666}}
 table{{border-collapse:collapse;width:100%}} th,td{{padding:.4rem .5rem;border-bottom:1px solid #e6e6e6;text-align:left}}
 th{{font-size:.75rem;text-transform:uppercase;letter-spacing:.04em;color:#666;cursor:pointer;user-select:none}}
 td.r,th.r{{text-align:right;font-variant-numeric:tabular-nums}} .strong{{font-weight:600}}
 tr:hover td{{background:#fafafa}} a{{color:#0b62d6;text-decoration:none}} a:hover{{text-decoration:underline}}
 @media (prefers-color-scheme: dark){{body{{background:#111;color:#eee}} th,td{{border-color:#2a2a2a}} tr:hover td{{background:#1a1a1a}} a{{color:#6cb6ff}}}}
</style>
<h1>TableSalt OCR leaderboard</h1>
<p class=sub>{len(rows)} models · dataset <code>{htmllib.escape(name)}</code> · final = 0.85 × tier-weighted TEDS + 0.15 × QA · click a model for per-table detail</p>
<table id=t><thead><tr><th>#</th><th>Model</th><th class=r>B</th><th>Quant</th><th>fmt</th>
<th class=r>T0</th><th class=r>T1</th><th class=r>T2</th><th class=r>T3</th><th class=r>QA</th>
<th class=r>CellF1</th><th class=r>FINAL</th><th class=r>s/table</th></tr></thead>
<tbody>{"".join(body)}</tbody></table>
{"<h2>Excluded</h2><ul>" + excluded + "</ul>" if excluded else ""}
{"<h2>Not scored</h2><ul>" + not_scored + "</ul>" if not_scored else ""}
<script>
document.querySelectorAll('#t th').forEach((th,i)=>th.onclick=()=>{{
 const tb=document.querySelector('#t tbody');const rows=[...tb.rows];
 const num=!/Model|Quant|fmt/.test(th.textContent);
 rows.sort((a,b)=>{{const x=a.cells[i].textContent.trim(),y=b.cells[i].textContent.trim();
  return num?(parseFloat(y)||0)-(parseFloat(x)||0):x.localeCompare(y);}});
 tb.append(...rows);}});
</script>"""
    path = config.METRICS / "leaderboard.html"
    path.write_text(html, encoding="utf-8")
    return path


def _write_detail(row: dict) -> None:
    out = config.METRICS / "runs" / f"{row['run_id']}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        ds = dataset.load(row["dataset"])
    except SystemExit:
        return
    gt_dir = ds.path
    pred_dir = config.PRED / row["model_id"]
    parts = []
    for table in store.table_rows(row["run_id"]):
        entry = next((e for e in ds.manifest if e["id"] == table["tid"]), None)
        if entry is None:
            continue
        image = (gt_dir / entry["image"]).relative_to(config.ROOT)
        gt_html = (gt_dir / table["tid"] / "gt" / "table.html").read_text(encoding="utf-8")
        pred_path = pred_dir / f"{table['tid']}.html"
        pred_html = (
            pred_path.read_text(encoding="utf-8") if pred_path.exists() else "<p><em>empty</em></p>"
        )
        parts.append(f"""
<section>
 <h3>{table["tid"]} <small>{table["tier"]} · {table["style"]} · TEDS {table["teds"]:.4f} · CellF1 {table["cell_f1"]:.3f}</small></h3>
 <div class=grid>
  <figure><img src="../../{image}" loading=lazy><figcaption>image</figcaption></figure>
  <figure><div class=scroll>{gt_html}</div><figcaption>ground truth</figcaption></figure>
  <figure><div class=scroll>{pred_html}</div><figcaption>prediction</figcaption></figure>
 </div>
</section>""")
    html = f"""<!doctype html><meta charset=utf-8><title>{htmllib.escape(row["model_name"])} — {row["dataset"]}</title>
<style>
 body{{font:14px/1.5 -apple-system,BlinkMacSystemFont,sans-serif;margin:2rem;background:#fff;color:#111}}
 h1{{font-size:1.4rem;margin:0 0 .25rem}} h2{{font-size:1rem;color:#555;margin:.25rem 0 1.5rem}}
 h3{{font-size:.95rem;margin:2rem 0 .5rem}} small{{font-weight:400;color:#666}}
 .grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;align-items:start}}
 figure{{margin:0}} img{{max-width:100%;border:1px solid #ddd;border-radius:6px}}
 .scroll{{max-height:22rem;overflow:auto;border:1px solid #ddd;border-radius:6px;padding:.5rem}}
 figcaption{{font-size:.75rem;color:#666;margin-top:.25rem}}
 table td,table th{{border:1px solid #bbb;padding:.2rem .4rem;font-size:.75rem}}
 table{{border-collapse:collapse}}
 @media (prefers-color-scheme: dark){{body{{background:#111;color:#eee}} .scroll,img{{border-color:#333}} figcaption,h2,small{{color:#aaa}} table td,table th{{border-color:#444}}}}
</style>
<h1>{htmllib.escape(row["model_name"])}</h1>
<h2>{row["hf_repo"]} · {row["quant"]} · {row["dataset"]} · FINAL {row["final"]:.4f} ·
TEDS {row["teds_weighted"]:.4f} · QA {row["qa_accuracy"]:.4f} · CellF1 {row["cell_f1"]:.4f} ·
{row["s_per_table"] or 0:.1f}s/table · sorted worst TEDS first</h2>
{"".join(parts)}"""
    out.write_text(html, encoding="utf-8")
