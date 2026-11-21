from tablesalt.validate import check, is_valid


def cell(r, c, rs=1, cs=1, text="x"):
    return {"text": text, "r": r, "c": c, "rs": rs, "cs": cs}


def test_full_grid_is_valid():
    cells = [cell(r, c) for r in range(3) for c in range(4)]
    assert is_valid(cells)


def test_spans_cover_grid():
    cells = [cell(0, 0, rs=2), cell(0, 1, cs=3), cell(1, 1), cell(1, 2), cell(1, 3)]
    assert check(cells) == []


def test_hole_detected():
    cells = [cell(0, 0), cell(0, 1), cell(1, 0)]
    assert any("hole" in p for p in check(cells))


def test_overlap_detected():
    cells = [cell(0, 0, cs=2), cell(0, 1)]
    assert any("overlap" in p for p in check(cells))


def test_bad_span_and_empty():
    assert check([]) == ["no cells"]
    assert any("bad" in p for p in check([cell(0, 0, rs=0)]))
