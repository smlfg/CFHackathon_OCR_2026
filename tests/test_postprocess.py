import pytest

from tablesalt import lexicon as lx
from tablesalt.postprocess import Fmt, detect_format, fix_table, parse, render
from tablesalt.types import Cell


@pytest.mark.parametrize(
    "text, digits, neg",
    [
        ("1,234.56", "123456", False),
        ("1.234,56", "123456", False),
        ("12 345", "12345", False),
        ("(1,234.56)", "123456", True),
        ("-12,345", "12345", True),
        ("79.036,73 %", "7903673", False),
    ],
)
def test_parse(text, digits, neg):
    n = parse(text)
    assert n is not None and n.digits == digits and n.neg == neg


def test_parse_rejects_labels():
    assert parse("n/a") is None
    assert parse("see note") is None
    assert parse("") is None


@pytest.mark.parametrize(
    "cells, style, sep, parens",
    [
        (["1,234.56", "-9,876.54", "12.30", "45,678.90"], "en", ",", False),
        (["1.234,56", "(9.876,54)", "12,30", "45.678,90"], "de", ",", True),
        (["12 345", "-1 234", "98 765", "4 321"], "int", " ", False),
        (["12,345", "(1,234)", "98,765", "4,321"], "int", ",", True),
    ],
)
def test_detect_format(cells, style, sep, parens):
    f = detect_format([parse(t) for t in cells])
    assert f.style == style
    assert f.parens == parens
    if style == "int":
        assert f.int_sep == sep


@pytest.mark.parametrize(
    "ocr, fmt, want",
    [
        ("1,234,56", Fmt("de"), "1.234,56"),  # wrong group separator repaired
        ("1234.56", Fmt("en"), "1,234.56"),  # missing thousands separator
        ("123456", Fmt("en"), "1,234.56"),  # all separators lost
        ("-1.234,56", Fmt("de", parens=True), "(1.234,56)"),
        ("(12 345)", Fmt("int", " ", False), "-12 345"),
        ("12345 kg", Fmt("int", ",", False, "kg"), "12,345 kg"),
        ("12,345.67%", Fmt("en", unit="%"), "12,345.67 %"),
        ("O.54", Fmt("en"), "0.54"),
    ],
)
def test_render(ocr, fmt, want):
    assert render(parse(ocr), fmt) == want


def _table(header, labels, values, hr=1):
    cells = [Cell(text=t, r=0, c=c) for c, t in enumerate(header)]
    for r, (lab, vals) in enumerate(zip(labels, values, strict=False), start=hr):
        cells.append(Cell(text=lab, r=r, c=0))
        cells += [Cell(text=v, r=r, c=c) for c, v in enumerate(vals, start=1)]
    return cells


def test_fix_table_snaps_labels_and_numbers():
    cells = _table(
        ["Positon", "Q1", "O2", "Q3"],
        ["Revenu", "Cost of saies", "EBlT"],
        [["1,234.56", "-2,345.67", "n/a"], ["3,456.78", "4.567,89", "Wa"], ["12.34", "–", ""]],
    )
    fix_table(cells, hr=1, n_cols=4)
    got = {(c.r, c.c): c.text for c in cells}
    assert [got[(0, c)] for c in range(4)] == ["Position", "Q1", "Q2", "Q3"]
    assert [got[(r, 0)] for r in (1, 2, 3)] == ["Revenue", "Cost of sales", "EBIT"]
    assert got[(2, 2)] == "4,567.89"
    assert got[(2, 3)] == "n/a"
    assert got[(3, 2)] == lx.EN_DASH
    assert got[(3, 3)] == ""
