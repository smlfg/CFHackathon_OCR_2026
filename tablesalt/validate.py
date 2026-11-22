"""Check a cell list: every (r, c) grid position covered exactly once, spans inside the grid."""

from __future__ import annotations


def check(cells: list[dict]) -> list[str]:
    """-> list of problems (empty = valid)."""
    problems: list[str] = []
    if not cells:
        return ["no cells"]
    try:
        n_rows = max(int(c["r"]) + int(c.get("rs", 1)) for c in cells)
        n_cols = max(int(c["c"]) + int(c.get("cs", 1)) for c in cells)
    except (KeyError, TypeError, ValueError) as e:
        return [f"malformed cell: {e}"]
    seen: dict[tuple[int, int], int] = {}
    for k, c in enumerate(cells):
        if not isinstance(c.get("text", ""), str):
            problems.append(f"cell {k}: text is not a string")
        r, cc = int(c["r"]), int(c["c"])
        rs, cs = int(c.get("rs", 1)), int(c.get("cs", 1))
        if r < 0 or cc < 0 or rs < 1 or cs < 1:
            problems.append(f"cell {k}: bad position/span {r},{cc},{rs},{cs}")
            continue
        for dr in range(rs):
            for dc in range(cs):
                p = (r + dr, cc + dc)
                if p in seen:
                    problems.append(f"overlap at {p}: cells {seen[p]} and {k}")
                seen[p] = k
    for r in range(n_rows):
        for cc in range(n_cols):
            if (r, cc) not in seen:
                problems.append(f"hole at {(r, cc)}")
    return problems


def is_valid(cells: list[dict]) -> bool:
    """True if the cell list covers the grid exactly once."""
    return not check(cells)
