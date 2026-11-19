"""Cube planes on another spatial grid: copied where the lattices agree, else by exact pixel overlap.

A cube that already lies on the pixel lattice of the target grid is *placed*:
its values are copied to the pixels they belong to and nothing is computed.
Any other cube is resampled with the only weights that conserve flux for
surface-brightness data -- the areas over which each of its pixels overlaps
each target pixel: a target pixel takes the area-weighted mean of the source
pixels under it, so every source pixel hands its flux to the target pixels in
proportion to the area it shares with them. Nothing is interpolated between
pixels and no value is invented: where the target pixels are smaller, each
shows the source pixel it lies in.

That resampling is exact and conserves flux, but it is a second resampling of
data the cube builder already resampled once, and it costs resolution: a
source pixel is spread over the target pixels it straddles (up to four of
them for a same-scale grid shifted by half a pixel), and coarser pixels stay
visible as blocks. Building the cubes on the target grid in the first place
(``cube_build_matched``) avoids both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .grid import SpatialGrid

#: DQ of a spaxel without data: what ``cube_build`` writes there (DO_NOT_USE | NON_SCIENCE).
NO_DATA_DQ = 513
#: DQ bits that only say "no data"; every other bit of a contributing pixel is passed on.
_NO_DATA_BITS = np.uint32(NO_DATA_DQ)
#: Planes resampled at a time (bounds the memory of the intermediate arrays).
_CHUNK = 256
#: Overlaps below this fraction of a target pixel are rounding, not geometry, and are dropped.
_MIN_OVERLAP = 1e-9
#: A DQ bit is passed on -- and a missing error makes the error of the result unknown -- when the
#: spaxels concerned make up at least this much of the valid area under a target spaxel (a sliver
#: of a flagged neighbour flags nothing).
_DQ_SHARE = 0.01

IDENTITY = "identity"
REGRIDDED = "regridded"


# --------------------------------------------------------------------------- polygon overlap
def _clip(polygon: list[tuple[float, float]], axis: int, bound: float, keep_below: bool) -> list[tuple[float, float]]:
    """A convex polygon clipped to the half-plane ``coordinate[axis] <= bound`` (or ``>=``): one
    pass of Sutherland-Hodgman."""
    out: list[tuple[float, float]] = []
    n = len(polygon)
    for k in range(n):
        p, q = polygon[k], polygon[(k + 1) % n]
        p_in = p[axis] <= bound if keep_below else p[axis] >= bound
        q_in = q[axis] <= bound if keep_below else q[axis] >= bound
        if p_in:
            out.append(p)
        if p_in != q_in:
            t = (bound - p[axis]) / (q[axis] - p[axis])
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def _area(polygon: list[tuple[float, float]]) -> float:
    """Area of a simple polygon (shoelace formula)."""
    n = len(polygon)
    if n < 3:
        return 0.0
    twice = 0.0
    for k in range(n):
        x0, y0 = polygon[k]
        x1, y1 = polygon[(k + 1) % n]
        twice += x0 * y1 - x1 * y0
    return 0.5 * abs(twice)


def overlap_matrix(source: SpatialGrid, target: SpatialGrid, *, pixels: np.ndarray | None = None) -> Any:
    """Overlap areas between the pixels of two grids, as a sparse matrix.

    Element ``[j, i]`` is the area that source pixel ``i`` shares with target
    pixel ``j``, in units of the target pixel area (pixels numbered row by
    row, ``i = y * nx + x``). The outline of every source pixel is carried to
    the target grid through the sky -- so any offset, scale ratio and rotation
    between the two is taken as it is -- and clipped against the target pixels
    it reaches. ``pixels`` (a boolean image on the source grid) restricts the
    work to the source pixels that carry data; the others get no entries.

    A column sums to the area of its source pixel in target pixels (as far as
    the pixel lies inside the target grid), a row to the fraction of its
    target pixel that the source grid covers.
    """
    from scipy import sparse

    sy, sx = source.shape
    ty, tx = target.shape
    corner_y, corner_x = np.mgrid[0:sy + 1, 0:sx + 1]
    px, py = target.pixels_of(source, corner_x - 0.5, corner_y - 0.5)     # the corner lattice of the source pixels
    wanted = np.ones((sy, sx), dtype=bool) if pixels is None else np.asarray(pixels, dtype=bool)
    rows: list[int] = []
    cols: list[int] = []
    areas: list[float] = []
    for y, x in zip(*np.nonzero(wanted)):
        quad = [(float(px[y, x]), float(py[y, x])), (float(px[y, x + 1]), float(py[y, x + 1])),
                (float(px[y + 1, x + 1]), float(py[y + 1, x + 1])), (float(px[y + 1, x]), float(py[y + 1, x]))]
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        # target pixel k spans [k - 0.5, k + 0.5]
        x_lo, x_hi = max(0, int(np.floor(min(xs) + 0.5))), min(tx - 1, int(np.floor(max(xs) + 0.5)))
        y_lo, y_hi = max(0, int(np.floor(min(ys) + 0.5))), min(ty - 1, int(np.floor(max(ys) + 0.5)))
        for row in range(y_lo, y_hi + 1):
            strip = _clip(_clip(quad, 1, row - 0.5, False), 1, row + 0.5, True)
            if len(strip) < 3:
                continue
            for col in range(x_lo, x_hi + 1):
                area = _area(_clip(_clip(strip, 0, col - 0.5, False), 0, col + 0.5, True))
                if area > _MIN_OVERLAP:
                    rows.append(row * tx + col)
                    cols.append(int(y) * sx + int(x))
                    areas.append(area)
    return sparse.csr_matrix((areas, (rows, cols)), shape=(ty * tx, sy * sx))


# --------------------------------------------------------------------------- a cube on a grid
@dataclass
class Placement:
    """How a cube gets onto a target grid, and what that does to it.

    ``mode`` is :data:`IDENTITY` (shared pixel lattice: values are copied,
    ``offset`` is the target pixel of the cube's pixel (0, 0)) or
    :data:`REGRIDDED` (exact-overlap resampling with ``matrix``).
    ``scale_ratio`` is the source spaxel scale over the target one and
    ``phase`` the sub-pixel offset of the source lattice (target pixels).
    ``correlation`` is the effective number of target spaxels one source
    spaxel is spread over (the square of its area in target spaxels over the
    sum of its squared overlaps): after resampling, neighbouring target
    spaxels share source pixels, so the variance of a sum over many of them
    is ``correlation`` times their variances added up -- its error
    ``sqrt(correlation)`` times their errors in quadrature (1 for an
    identity placement).
    """

    source: SpatialGrid
    target: SpatialGrid
    mode: str
    offset: tuple[int, int] | None
    matrix: Any
    scale_ratio: float
    phase: tuple[float, float]
    correlation: float

    @classmethod
    def plan(cls, source: SpatialGrid, target: SpatialGrid, *, pixels: np.ndarray | None = None) -> Placement:
        """Decide how cubes on ``source`` get onto ``target`` (``pixels``: the source spaxels with data)."""
        ratio = source.scale / target.scale
        offset = target.lattice_offset(source)
        if offset is not None:
            return cls(source, target, IDENTITY, offset, None, ratio, (0.0, 0.0), 1.0)
        matrix = overlap_matrix(source, target, pixels=pixels)
        # per source pixel: (sum of its overlaps)^2 / sum of their squares; the mean over the pixels in use
        column = np.asarray(matrix.sum(axis=0)).ravel()
        squares = np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel()
        used = squares > 0
        correlation = float(np.mean(column[used] ** 2 / squares[used])) if used.any() else 1.0
        return cls(source, target, REGRIDDED, None, matrix, ratio, target.phase_of(source), max(correlation, 1.0))

    # ------------------------------------------------------------------ images
    def image(self, image: np.ndarray, *, min_coverage: float = 0.5) -> np.ndarray:
        """A 2-D image of surface-brightness-like values on the target grid (NaN where there is none)."""
        return self.cube(np.asarray(image, dtype=float)[None], min_coverage=min_coverage)["data"][0]

    def mask(self, mask: np.ndarray, *, complete: float = 0.999) -> np.ndarray:
        """A boolean spaxel mask on the target grid: the target spaxels that lie (to the fraction
        ``complete`` of their area) inside the masked source spaxels."""
        ty, tx = self.target.shape
        mask = np.asarray(mask, dtype=bool)
        if self.mode == IDENTITY:
            out = np.zeros((ty, tx), dtype=bool)
            (ty0, ty1, tx0, tx1), (sy0, sy1, sx0, sx1) = self._windows()
            out[ty0:ty1, tx0:tx1] = mask[sy0:sy1, sx0:sx1]
            return out
        return (self.matrix @ mask.ravel().astype(float)).reshape(ty, tx) >= complete

    # ------------------------------------------------------------------ cubes
    def cube(self, data: np.ndarray, err: np.ndarray | None = None, dq: np.ndarray | None = None,
             wmap: np.ndarray | None = None, *, min_coverage: float = 0.5) -> dict[str, np.ndarray]:
        """Planes ``(n, ny, nx)`` of a cube on the target grid.

        Returns ``data`` (NaN where the cube holds nothing), ``err``, ``dq``
        and ``wmap`` (those of the inputs that were given; ``dq`` always).
        Identity placement copies. Resampling takes, per target spaxel, the
        mean of the valid source spaxels under it weighted by overlap area;
        the spaxel is kept when valid source spaxels cover at least
        ``min_coverage`` of its area -- at the rim of a footprint a spaxel
        that is mostly covered shows what covers it rather than being
        dropped; such a spaxel carries the mean of the part that is covered,
        so flux is conserved in the interior of a footprint and only there.
        Errors are propagated as for a weighted mean of independent values
        (see :attr:`correlation` for sums; NaN where a value without an error
        contributes a hundredth of the valid area or more), the weight map is averaged
        with the same weights, and the DQ bits of the spaxels that contribute
        at least a hundredth of the valid area are combined; spaxels without
        data get :data:`NO_DATA_DQ`.
        """
        data = np.asarray(data)
        n = data.shape[0]
        ty, tx = self.target.shape
        out: dict[str, np.ndarray] = {"data": np.full((n, ty, tx), np.nan, dtype=np.float32),
                                      "dq": np.full((n, ty, tx), NO_DATA_DQ, dtype=np.uint32)}
        if err is not None:
            out["err"] = np.full((n, ty, tx), np.nan, dtype=np.float32)
        if wmap is not None:
            out["wmap"] = np.zeros((n, ty, tx), dtype=np.float32)
        valid = np.isfinite(data)
        if dq is not None:
            valid &= (np.asarray(dq).astype(np.uint32) & np.uint32(1)) == 0
        if self.mode == IDENTITY:
            (ty0, ty1, tx0, tx1), (sy0, sy1, sx0, sx1) = self._windows()
            target = (slice(None), slice(ty0, ty1), slice(tx0, tx1))
            source = (slice(None), slice(sy0, sy1), slice(sx0, sx1))
            good = valid[source]
            out["data"][target] = np.where(good, data[source], np.nan)
            out["dq"][target] = np.where(good, 0 if dq is None else np.asarray(dq)[source].astype(np.uint32), NO_DATA_DQ)
            if err is not None:
                out["err"][target] = np.where(good, np.asarray(err)[source], np.nan)
            if wmap is not None:
                out["wmap"][target] = np.where(good, np.asarray(wmap)[source], 0.0)
            return out

        matrix = self.matrix
        squared = matrix.multiply(matrix).tocsr()
        other_bits = 0
        if dq is not None:
            present = np.bitwise_or.reduce(np.asarray(dq).astype(np.uint32)[valid]) if valid.any() else np.uint32(0)
            other_bits = int(present & ~_NO_DATA_BITS)
        for start in range(0, n, _CHUNK):
            stop = min(n, start + _CHUNK)
            good = valid[start:stop].reshape(stop - start, -1)
            values = np.where(good, data[start:stop].reshape(stop - start, -1), 0.0).astype(np.float64)
            covered = (matrix @ good.T.astype(np.float64)).T                 # area fraction under valid source spaxels
            keep = covered >= max(float(min_coverage), 1e-9)
            norm = np.where(keep, covered, 1.0)
            flux = (matrix @ values.T).T
            out["data"][start:stop] = np.where(keep, flux / norm, np.nan).reshape(stop - start, ty, tx)
            dq_chunk = np.where(keep, 0, NO_DATA_DQ).astype(np.uint32)
            if other_bits:
                source_dq = np.asarray(dq[start:stop]).astype(np.uint32).reshape(stop - start, -1)
                for bit in (1 << b for b in range(32) if other_bits >> b & 1):
                    flagged = (good & ((source_dq & np.uint32(bit)) != 0)).T.astype(np.float64)
                    dq_chunk |= np.where(keep & ((matrix @ flagged).T >= _DQ_SHARE * norm), np.uint32(bit), np.uint32(0))
            out["dq"][start:stop] = dq_chunk.reshape(stop - start, ty, tx)
            if err is not None:
                variance = np.where(good, np.asarray(err[start:stop], dtype=np.float64).reshape(stop - start, -1) ** 2, 0.0)
                unknown = good & ~np.isfinite(variance)              # a value without an error: its share is not zero
                variance = np.where(np.isfinite(variance), variance, 0.0)
                known = keep & ~((matrix @ unknown.T.astype(np.float64)).T >= _DQ_SHARE * norm) if unknown.any() else keep
                out["err"][start:stop] = np.where(known, np.sqrt((squared @ variance.T).T) / norm, np.nan).reshape(stop - start, ty, tx)
            if wmap is not None:
                weights = np.where(good, np.asarray(wmap[start:stop], dtype=np.float64).reshape(stop - start, -1), 0.0)
                out["wmap"][start:stop] = np.where(keep, (matrix @ weights.T).T / norm, 0.0).reshape(stop - start, ty, tx)
        return out

    def _windows(self) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        """Identity placement: the target window ``(y0, y1, x0, x1)`` and the source window that
        lands in it (the part of the source that lies inside the target grid)."""
        x0, y0 = self.offset or (0, 0)
        sy, sx = self.source.shape
        ty, tx = self.target.shape
        tx0, tx1 = max(0, x0), min(tx, x0 + sx)
        ty0, ty1 = max(0, y0), min(ty, y0 + sy)
        return (ty0, max(ty0, ty1), tx0, max(tx0, tx1)), (ty0 - y0, max(ty0, ty1) - y0, tx0 - x0, max(tx0, tx1) - x0)
