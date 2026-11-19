"""Geometry of label placement along one axis: spreading boxes so that none
overlaps, and packing band rectangles with their labels into rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _spread(centres: np.ndarray, widths: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Positions as close as possible to the (sorted) ``centres`` at which boxes of ``widths``
    neither overlap nor leave ``[lo, hi]``: overlapping neighbours are merged into blocks
    that are centred on their members' wishes."""
    blocks: list[dict[str, Any]] = []
    for i in range(len(centres)):
        blocks.append({"idx": [i], "width": float(widths[i])})
        while True:
            block = blocks[-1]
            offset, wish = 0.0, 0.0
            for k in block["idx"]:
                wish += centres[k] - widths[k] / 2.0 - offset
                offset += widths[k]
            block["left"] = min(max(wish / len(block["idx"]), lo), max(lo, hi - block["width"]))
            if len(blocks) > 1 and blocks[-2]["left"] + blocks[-2]["width"] > block["left"]:
                previous = blocks.pop(-2)
                block["idx"] = previous["idx"] + block["idx"]
                block["width"] += previous["width"]
                continue
            break
    out = np.empty(len(centres))
    for block in blocks:
        x = block["left"]
        for k in block["idx"]:
            out[k] = x + widths[k] / 2.0
            x += widths[k]
    return out


def _declutter(wish: np.ndarray, widths: np.ndarray, lo: float, hi: float, *, max_shift: float,
               droppable: np.ndarray | None = None, priority: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Spread labels along one axis; where there is no room, drop the most crowded ones.

    Returns ``(positions, kept)`` in the input order. A label is dropped
    when the labels do not fit at all, or when fitting them would move one
    further than ``max_shift`` from its feature; the one with the nearest
    neighbour goes first (among ``droppable``, while any is left) -- with a
    ``priority`` per label, the lowest-priority one among the most crowded.
    """
    n = len(wish)
    kept = np.ones(n, dtype=bool)
    positions = np.array(wish, dtype=float)
    droppable = np.ones(n, dtype=bool) if droppable is None else np.asarray(droppable, dtype=bool)
    while kept.any():
        idx = np.nonzero(kept)[0]
        idx = idx[np.argsort(wish[idx], kind="stable")]
        fits = widths[idx].sum() <= hi - lo
        if fits:
            placed = _spread(wish[idx], widths[idx], lo, hi)
            if np.max(np.abs(placed - wish[idx])) <= max_shift or len(idx) == 1:
                positions[idx] = placed
                break
        if len(idx) == 1:
            positions[idx] = np.clip(wish[idx], lo, hi)
            break
        gaps = np.diff(wish[idx])
        nearest = np.minimum(np.r_[gaps, np.inf], np.r_[np.inf, gaps])
        pool = np.nonzero(droppable[idx])[0]
        if not pool.size:
            pool = np.arange(len(idx))
        if priority is None:
            kept[idx[pool[np.argmin(nearest[pool])]]] = False
        else:   # among the most crowded labels, the one that matters least goes
            crowded = pool[nearest[pool] <= 1.5 * nearest[pool].min() + 1e-9]
            kept[idx[crowded[np.argmin(np.asarray(priority, dtype=float)[idx[crowded]])]]] = False
    return positions, kept


def _pack_bands(entries: list[dict[str, Any]], to_px: Any, measure: Any, lo: float, hi: float, px: float) -> int:
    """Pack band rectangles and their labels into rows, **every species in one row**.

    All bands of a species (all regions of one ice) share a row; species
    whose rectangles and labels do not meet may share it too. A label goes
    to the right of its rectangle, else to its left, where the row has room;
    a rectangle without room keeps its mark and loses its label (its
    species is named at another of its bands). Sets ``row``, ``side`` and
    ``label_px`` on every entry and returns the number of rows.
    """
    gap = 3.0 * px
    rows: list[list[tuple[float, float]]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for e in entries:
        groups.setdefault((e["group"], e["species"]), []).append(e)

    def free(taken: list[tuple[float, float]], left: float, right: float) -> bool:
        return all(right < t0 or left > t1 for t0, t1 in taken)

    for members in groups.values():
        rects = []
        for e in members:
            a, b = to_px([e["lo"], e["hi"]])
            if b - a < 2.0 * px:
                a, b = 0.5 * (a + b) - px, 0.5 * (a + b) + px
            e["label_px"] = measure(e["text"])[0]
            rects.append((a, b))
        for r in range(len(rows) + 1):
            if r == len(rows):
                rows.append([])
            if not all(free(rows[r], a - px, b + px) for a, b in rects):
                continue
            taken = rows[r] + [(a - px, b + px) for a, b in rects]
            sides: list[str | None] = []
            for e, (a, b) in zip(members, rects):
                width, side = e["label_px"], None
                for name, left, right in (("right", b + gap - px, b + gap + width + px),
                                          ("left", a - gap - width - px, a - gap + px)):
                    if left >= lo and right <= hi and free(taken, left, right):
                        side = name
                        taken.append((left, right))
                        break
                sides.append(side)
            if any(sides) or not rows[r]:        # named somewhere in the row -- or the row is empty anyway
                rows[r] = taken
                for e, side in zip(members, sides):
                    e.update(row=r, side=side)
                break
    return len(rows)


__all__ = [
]
