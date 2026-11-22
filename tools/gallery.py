#!/usr/bin/env python3
# ruff: noqa: E501
"""Build a single self-contained gallery.html from a TableSalt dataset.

Usage:
    python tools/gallery.py                                  # ground truth only
    python tools/gallery.py --pred predictions_tess          # GT next to a prediction
    python tools/gallery.py --gt data/dev --pred preds --out gallery.html

No server and no dependencies beyond the stdlib: images are embedded as data
URIs, so the file opens straight from disk (file://). Metrics are computed with
the official functions in evaluate.py, so what the gallery shows matches the
scorer - when its dependencies (requirements.txt) are importable.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import sys
from html.parser import HTMLParser
from pathlib import Path


# --------------------------------------------------------------------------- #
# Official metrics, imported lazily so the gallery still opens without deps.
# --------------------------------------------------------------------------- #
def load_scorer():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: official scorer
    try:
        import evaluate as ev
    except Exception:
        return None
    return ev


def esc(value) -> str:
    return html.escape(str(value), quote=True)


# --------------------------------------------------------------------------- #
# Parsing: cells.json and arbitrary prediction HTML -> one grid representation
# --------------------------------------------------------------------------- #
class _TableParser(HTMLParser):
    """First <table> in a document -> cells with r/c/rs/cs/header/text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cells: list[dict] = []
        self._tables = 0
        self._depth = 0
        self._r = -1
        self._c = 0
        self._cur: dict | None = None
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._tables += 1
            self._depth += 1
        elif self._tables != 1:
            return
        elif tag == "tr":
            self._r += 1
            self._c = 0
        elif tag in ("td", "th"):
            a = dict(attrs)
            self._cur = {
                "text": "",
                "r": self._r,
                "c": self._c,
                "rs": _span(a.get("rowspan")),
                "cs": _span(a.get("colspan")),
                "header": tag == "th",
            }
            self._buf = []
        elif tag == "br" and self._cur is not None:
            self._buf.append(" ")

    def handle_endtag(self, tag):
        if tag == "table":
            self._depth -= 1
        elif (tag in ("td", "th") and self._cur is not None) or (
            tag == "tr" and self._cur is not None
        ):
            self._cur["text"] = " ".join("".join(self._buf).split())
            self.cells.append(self._cur)
            self._c += self._cur["cs"]
            self._cur = None

    def handle_data(self, data):
        if self._cur is not None:
            self._buf.append(data)


def _span(value) -> int:
    try:
        return max(1, int(value or 1))
    except (TypeError, ValueError):
        return 1


def parse_html_table(html_str: str) -> list[dict]:
    parser = _TableParser()
    try:
        parser.feed(html_str)
        parser.close()
    except Exception:
        pass
    return parser.cells


def load_pred_cells(pred_dir: Path, tid: str) -> list[dict] | None:
    """Prediction as cells: <tid>.json (cells schema) or <tid>.html."""
    pj, ph = pred_dir / f"{tid}.json", pred_dir / f"{tid}.html"
    if pj.exists():
        try:
            return json.loads(pj.read_text(encoding="utf-8"))
        except Exception:
            return None
    if ph.exists():
        return parse_html_table(ph.read_text(encoding="utf-8", errors="replace"))
    return None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def footprint(cells: list[dict]) -> dict[tuple[int, int], dict]:
    """Map every covered (r, c) to the cell covering it."""
    out = {}
    for cell in cells:
        for dr in range(cell.get("rs", 1)):
            for dc in range(cell.get("cs", 1)):
                out[(cell.get("r", 0) + dr, cell.get("c", 0) + dc)] = cell
    return out


def diff_cells(gt_cells: list[dict], pred_cells: list[dict]) -> dict[tuple[int, int], str]:
    """Per prediction cell origin: 'ok' if the GT has the same text at that position."""
    gt = footprint(gt_cells)
    status = {}
    for cell in pred_cells:
        key = (cell.get("r", 0), cell.get("c", 0))
        hit = gt.get(key)
        status[key] = "ok" if hit and norm(hit.get("text")) == norm(cell.get("text")) else "bad"
    return status


def norm(text) -> str:
    return " ".join(str(text or "").split()).casefold()


def render_table(cells: list[dict], status: dict[tuple[int, int], str] | None = None) -> str:
    """Render cells (cells.json schema) as an HTML table, optionally diff-tinted."""
    if not cells:
        return "<p class=empty>no cells</p>"

    grid = {(c.get("r", 0), c.get("c", 0)): c for c in cells}
    n_rows = max(c.get("r", 0) + c.get("rs", 1) for c in cells)
    n_cols = max(c.get("c", 0) + c.get("cs", 1) for c in cells)

    out = []
    for r in range(n_rows):
        tds = []
        for c in range(n_cols):
            cell = grid.get((r, c))
            if cell is None:
                continue  # covered by a span
            tag = "th" if cell.get("header") else "td"
            attrs = ""
            if cell.get("rs", 1) > 1:
                attrs += f' rowspan="{cell["rs"]}"'
            if cell.get("cs", 1) > 1:
                attrs += f' colspan="{cell["cs"]}"'
            cls = []
            if cell.get("rs", 1) > 1 or cell.get("cs", 1) > 1:
                cls.append("span")
            if status is not None:
                cls.append(status.get((r, c), "bad"))
            if cls:
                attrs += f' class="{" ".join(cls)}"'
            text = esc(cell.get("text", ""))
            tds.append(f"<{tag}{attrs}>{text or '<span class=nil>&middot;</span>'}</{tag}>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    return "<table>" + "".join(out) + "</table>"


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def qa_block(qa: list[dict]) -> str:
    if not qa:
        return ""
    items = []
    for item in qa:
        answer = item.get("answer")
        num = item.get("answer_numeric")
        num_html = (
            f"<span class=num>{esc(num)}</span>"
            if isinstance(num, (int, float)) and str(num) != str(answer)
            else ""
        )
        items.append(
            f"<li><span class=qtype>{esc(item.get('type', 'qa'))}</span>"
            f"<span class=q>{esc(item.get('question', ''))}</span>"
            f"<span class=a>{esc(answer)}{num_html}</span></li>"
        )
    return f"<ol class=qa>{''.join(items)}</ol>"


def metric_badges(teds: float | None, teds_s: float | None, f1: float | None) -> str:
    if teds is None:
        return ""

    def tone(v: float) -> str:
        return "good" if v >= 0.8 else ("mid" if v >= 0.5 else "weak")

    return (
        f'<span class="tag score {tone(teds)}">TEDS {teds:.3f}</span>'
        f'<span class="tag score {tone(teds_s)}">struct {teds_s:.3f}</span>'
        f'<span class="tag score {tone(f1)}">cells {f1:.3f}</span>'
    )


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(gt: Path, out: Path, pred: Path | None = None) -> Path:
    manifest = json.loads((gt / "manifest.json").read_text(encoding="utf-8"))
    ev = load_scorer() if pred else None

    cards, tiers, styles = [], set(), set()
    scored = []

    for entry in manifest:
        tid = entry["id"]
        t_dir = gt / tid
        tiers.add(entry.get("tier", "?"))
        styles.add(entry.get("style", "?"))

        image = t_dir / Path(entry["image"]).name
        cells_file = t_dir / "gt" / "cells.json"
        qa_file = t_dir / "gt" / "qa.json"
        cells = json.loads(cells_file.read_text(encoding="utf-8")) if cells_file.exists() else []
        qa = json.loads(qa_file.read_text(encoding="utf-8")) if qa_file.exists() else []

        pred_cells = load_pred_cells(pred, tid) if pred else None
        teds = teds_s = f1 = None
        pred_panel = ""

        if pred is not None:
            if pred_cells:
                diff_cells(cells, pred_cells)
                pred_panel = (
                    "<div class=panel><h3>prediction "
                    f"<span class=src>{esc(pred.name)}</span></h3>"
                    f"{render_table(pred_cells, diff_cells(cells, pred_cells))}</div>"
                )
            else:
                exists = (pred / f"{tid}.html").exists() or (pred / f"{tid}.json").exists()
                note = (
                    "prediction is an empty table - nothing recognised"
                    if exists
                    else "no prediction file"
                )
                pred_panel = (
                    f"<div class=panel><h3>prediction <span class=src>{esc(pred.name)}</span></h3>"
                    f"<p class=empty>{note}</p></div>"
                )

            if ev is not None:
                gt_html = ev.cells_json_to_html(cells)
                pred_html = ev.load_prediction(pred, tid)
                if pred_html:
                    teds = ev.teds(gt_html, pred_html)
                    teds_s = ev.teds(gt_html, pred_html, structure_only=True)
                    f1 = ev.cell_f1(gt_html, pred_html)
                    scored.append({"tier": entry.get("tier"), "teds": teds, "f1": f1})

        meta = f"{entry.get('n_rows', '?')} &times; {entry.get('n_cols', '?')}"
        gt_panel = "<div class=panel><h3>ground truth</h3>" + render_table(cells) + "</div>"
        cards.append(
            f'<article class=card data-tier="{esc(entry.get("tier"))}" '
            f'data-style="{esc(entry.get("style"))}" '
            f'data-teds="{teds if teds is not None else ""}" '
            f'id="{esc(tid)}"><header><h2>{esc(tid)}</h2><div class=tags>'
            f'<span class="tag tier {esc(entry.get("tier"))}">{esc(entry.get("tier"))}</span>'
            f"<span class=tag>{esc(entry.get('style'))}</span>"
            f"{'<span class=tag>spans</span>' if entry.get('has_spans') else ''}"
            f"<span class=tag>{meta}</span>{metric_badges(teds, teds_s, f1)}"
            f"</div></header>"
            f'<div class="body{" three" if pred is not None else ""}">'
            f'<figure><img loading=lazy src="{data_uri(image)}" alt="{esc(tid)}"></figure>'
            f"{gt_panel}{pred_panel}</div>{qa_block(qa)}</article>"
        )

    tier_buttons = "".join(
        f'<button data-f="tier" data-v="{esc(t)}">{esc(t)}</button>' for t in sorted(tiers)
    )
    style_buttons = "".join(
        f'<button data-f="style" data-v="{esc(s)}">{esc(s)}</button>' for s in sorted(styles)
    )

    headline = ""
    if scored:
        by_tier: dict[str, list[float]] = {}
        for row in scored:
            by_tier.setdefault(row["tier"], []).append(row["teds"])
        teds_mean = sum(row["teds"] for row in scored) / len(scored)
        f1_mean = sum(row["f1"] for row in scored) / len(scored)
        weighted = (
            ev.tier_weighted({t: sum(v) / len(v) for t, v in by_tier.items()})
            if ev is not None
            else teds_mean
        )
        headline = (
            f"<p class=sub><b>{esc(pred.name)}</b>: mean TEDS {teds_mean:.3f} &middot; "
            f"tier-weighted TEDS (the scored part) {weighted:.3f} &middot; "
            f"mean cell F1 {f1_mean:.3f} &middot; "
            f"<span class=legend><span class=sw ok></span>cell matches GT</span> "
            f"<span class=legend><span class=sw bad></span>wrong / missing</span></p>"
        )
    elif pred is not None:
        headline = (
            f"<p class=sub><b>{esc(pred.name)}</b>: metrics unavailable - run with the "
            "venv that has requirements.txt installed for TEDS and cell F1.</p>"
        )
    else:
        headline = (
            "<p class=sub>Ground truth only. Add <code>--pred &lt;dir&gt;</code> "
            "to compare a prediction cell by cell.</p>"
        )

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TableSalt gallery - {esc(gt.name)}</title>
<style>
  :root {{ --bg:#f6f5f2; --fg:#1b1b1b; --mut:#6b6b6b; --line:#d8d5ce; --card:#fff; --acc:#2f6f4f;
           --ok:#2f6f4f; --bad:#b3352f; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg:#141514; --fg:#eceae5; --mut:#9a9a94; --line:#33352f; --card:#1c1e1c;
           --acc:#7fc9a1; --ok:#6fbf8f; --bad:#e0716a; }} }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:24px; background:var(--bg); color:var(--fg);
         font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }}
  h1 {{ margin:0 0 4px; font-size:22px; letter-spacing:-.01em; }}
  code {{ font-size:12px; background:var(--bg); border:1px solid var(--line); padding:1px 5px; border-radius:4px; }}
  .sub {{ color:var(--mut); margin:0 0 20px; font-size:13px; }}
  .bar {{ position:sticky; top:0; z-index:5; display:flex; flex-wrap:wrap; gap:16px;
          padding:10px 0; margin-bottom:16px; background:var(--bg); border-bottom:1px solid var(--line); }}
  .group {{ display:flex; gap:6px; align-items:center; font-size:12px; color:var(--mut); }}
  button {{ font:inherit; font-size:12px; padding:4px 10px; border-radius:999px; cursor:pointer;
            border:1px solid var(--line); background:var(--card); color:var(--fg); }}
  button.on {{ border-color:var(--acc); color:var(--acc); font-weight:600; }}
  .grid {{ display:grid; gap:18px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; }}
  .card.hide {{ display:none; }}
  .card>header {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:space-between; margin-bottom:12px; }}
  h2 {{ margin:0; font-size:15px; font-variant-numeric:tabular-nums; }}
  h3 {{ margin:0 0 6px; font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:var(--mut); font-weight:600; }}
  .src {{ font-weight:400; text-transform:none; letter-spacing:0; }}
  .tags {{ display:flex; gap:6px; flex-wrap:wrap; }}
  .tag {{ font-size:11px; color:var(--mut); border:1px solid var(--line); border-radius:999px; padding:2px 8px; }}
  .tag.tier.T0 {{ color:#2f6f4f; border-color:#2f6f4f; }}
  .tag.tier.T1 {{ color:#8a6a12; border-color:#8a6a12; }}
  .tag.tier.T2 {{ color:#a4561c; border-color:#a4561c; }}
  .tag.tier.T3 {{ color:#a33; border-color:#a33; }}
  .tag.score {{ font-variant-numeric:tabular-nums; }}
  .tag.score.good {{ color:var(--ok); border-color:var(--ok); }}
  .tag.score.mid  {{ color:#8a6a12; border-color:#8a6a12; }}
  .tag.score.weak {{ color:var(--bad); border-color:var(--bad); }}
  .body {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:16px; align-items:start; }}
  .body.three {{ grid-template-columns:minmax(0,1.1fr) minmax(0,1fr) minmax(0,1fr); }}
  @media (max-width:1100px) {{ .body, .body.three {{ grid-template-columns:1fr; }} }}
  figure {{ margin:0; background:repeating-conic-gradient(var(--line) 0 25%,transparent 0 50%) 0 0/16px 16px;
            border:1px solid var(--line); border-radius:6px; padding:6px; }}
  figure img {{ display:block; width:100%; height:auto; border-radius:3px; }}
  .panel {{ overflow-x:auto; }}
  table {{ border-collapse:collapse; font-size:12px; width:100%; }}
  th,td {{ border:1px solid var(--line); padding:4px 7px; text-align:left; vertical-align:top; }}
  th {{ background:color-mix(in srgb, var(--acc) 10%, transparent); font-weight:600; }}
  td.span,th.span {{ outline:1px dashed var(--acc); outline-offset:-3px; }}
  th.ok,td.ok {{ background:rgba(47,111,79,.16); }}
  th.bad,td.bad {{ background:rgba(179,53,47,.18); }}
  .nil {{ color:var(--line); }}
  .qa {{ margin:14px 0 0; padding-left:22px; font-size:13px; }}
  .qa li {{ margin:3px 0; }}
  .qtype {{ font-size:10px; text-transform:uppercase; letter-spacing:.06em; color:var(--acc); margin-right:8px; }}
  .q {{ color:var(--mut); margin-right:8px; }}
  .a {{ font-weight:600; font-variant-numeric:tabular-nums; }}
  .num {{ font-weight:400; color:var(--mut); margin-left:6px; font-size:11px; }}
  .empty {{ color:var(--mut); }}
  .legend {{ display:inline-flex; align-items:center; gap:4px; margin-left:10px; }}
  .sw {{ width:10px; height:10px; border-radius:2px; display:inline-block; border:1px solid var(--line); }}
  .sw.ok {{ background:rgba(47,111,79,.55); }}
  .sw.bad {{ background:rgba(179,53,47,.55); }}
</style></head><body>
<h1>TableSalt gallery</h1>
{headline}
<div class=bar>
  <div class=group>tier <button data-f="tier" data-v="" class=on>all</button>{tier_buttons}</div>
  <div class=group>style <button data-f="style" data-v="" class=on>all</button>{style_buttons}</div>
  {'<div class=group>score <button data-f="score" data-v="weak">TEDS &lt; 0.5</button><button data-f="score" data-v="all" class=on>any</button></div>' if scored else ""}
</div>
<div class=grid>{"".join(cards)}</div>
<script>
  const state = {{ tier:"", style:"", score:"" }};
  const cards = [...document.querySelectorAll(".card")];
  const apply = () => cards.forEach(c => {{
    const t = parseFloat(c.dataset.teds);
    c.classList.toggle("hide",
      (state.tier && c.dataset.tier !== state.tier) ||
      (state.style && c.dataset.style !== state.style) ||
      (state.score === "weak" && !(isNaN(t) || t < 0.5)));
  }});
  document.querySelectorAll("button[data-f]").forEach(btn => btn.onclick = () => {{
    const f = btn.dataset.f;
    state[f] = btn.dataset.v;
    document.querySelectorAll(`button[data-f="${{f}}"]`).forEach(b => b.classList.toggle("on", b === btn));
    apply();
  }});
</script>
</body></html>
"""
    out.write_text(doc, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", default="data/dev", help="dataset root (contains manifest.json)")
    ap.add_argument("--pred", default=None, help="predictions dir with <table_id>.json or .html")
    ap.add_argument("--out", default="gallery.html")
    args = ap.parse_args()
    path = build(Path(args.gt), Path(args.out), Path(args.pred) if args.pred else None)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
