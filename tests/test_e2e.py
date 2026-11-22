"""End to end: one generated table per tier runs through the full pipeline without fallback."""

import importlib.util
import random
import shutil
from pathlib import Path

import pytest

from tablesalt import run, validate

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    shutil.which(run.CFG.tess_cmd) is None, reason="tesseract not installed"
)


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen", ROOT / "generate_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "tier, style, seed",
    [("T0", "grid", 11), ("T1", "hlines", 12), ("T2", "borderless", 13), ("T3", "grid", 14)],
)
def test_tier(tmp_path, gen, tier, style, seed):
    rng = random.Random(seed)
    spec = gen.make_table_spec(rng)
    img = gen.degrade(gen.render_table(spec, style, rng), tier, rng)
    path = tmp_path / "table.png"
    img.save(path)
    info = {}
    cells = run.predict(path, info=info)
    assert validate.is_valid(cells)
    assert info["rows"] == spec["n_rows"]
    assert info["cols"] == spec["n_cols"]
    assert info["hr"] == spec["header_rows"]


def test_process_one_never_crashes(tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")
    info = run.process_one((str(bad), str(tmp_path / "out"), False))
    assert info["mode"] in ("baseline", "empty")
    assert (tmp_path / "out" / "broken.json").exists()
