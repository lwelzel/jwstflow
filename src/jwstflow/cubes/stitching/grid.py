"""The spatial grid of an IFU cube, and the one grid several cubes are joined on.

A :class:`SpatialGrid` is the celestial pixel grid of a cube: a tangent point, the
pixel it sits on, the pixel-to-sky matrix and the array size. :func:`joint_grid`
chooses the grid of a joint cube from the grids of its segments -- the finest spaxel
scale among them over the union of their fields -- and
:meth:`SpatialGrid.cube_build_parameters` hands that grid to the official
``cube_build``, so band cubes can be built on it straight from the detector data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from ..wcs import celestial_wcs_from_header

#: Two grids share a pixel lattice -- so that data move between them by copying, without
#: any resampling -- when their pixel centres coincide to this fraction of a pixel.
LATTICE_TOLERANCE_PIX = 1e-3
#: Spaxel scales closer than this (relative) count as equal when the finest is chosen.
SCALE_TOLERANCE = 1e-4
ARCSEC2_PER_SR = (180.0 * 3600.0 / math.pi) ** 2


@dataclass(frozen=True)
class SpatialGrid:
    """The celestial pixel grid of an IFU cube (tangent projection).

    ``crval`` is the tangent point (RA, Dec) [deg], ``crpix`` the pixel it
    sits on (1-based, FITS), ``cd`` the matrix from pixel offsets to
    tangent-plane offsets [deg per pixel; rows: RA-like, Dec-like] and
    ``shape`` the array size ``(ny, nx)``. Pixel coordinates taken and
    returned by the methods are 0-based, pixel centres at integers.
    """

    crval: tuple[float, float]
    crpix: tuple[float, float]
    cd: tuple[tuple[float, float], tuple[float, float]]
    shape: tuple[int, int]

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_header(cls, header: Any, shape: tuple[int, ...] | None = None) -> SpatialGrid:
        """From the SCI header of a cube (the spectral axis is ignored)."""
        wcs = celestial_wcs_from_header(header)
        if shape is None:
            shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
        cd = np.asarray(wcs.pixel_scale_matrix, dtype=float)
        return cls((float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1])), (float(wcs.wcs.crpix[0]), float(wcs.wcs.crpix[1])),
                   ((float(cd[0, 0]), float(cd[0, 1])), (float(cd[1, 0]), float(cd[1, 1]))),
                   (int(shape[-2]), int(shape[-1])))

    @classmethod
    def from_fits(cls, path: Path | str, ext: str = "SCI") -> SpatialGrid:
        from astropy.io import fits

        with fits.open(path) as hdul:
            return cls.from_header(hdul[ext].header)

    # ------------------------------------------------------------------ geometry
    @cached_property
    def wcs(self) -> Any:
        """The grid as a celestial ``astropy.wcs.WCS``."""
        from astropy.wcs import WCS

        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.cunit = ["deg", "deg"]
        w.wcs.crval = list(self.crval)
        w.wcs.crpix = list(self.crpix)
        w.wcs.cd = np.asarray(self.cd, dtype=float)
        w.wcs.set()
        return w

    @property
    def scale(self) -> float:
        """Spaxel scale [arcsec]: the square root of the pixel area."""
        (a, b), (c, d) = self.cd
        return math.sqrt(abs(a * d - b * c)) * 3600.0

    @property
    def pixel_area_arcsec2(self) -> float:
        return self.scale**2

    @property
    def pixel_area_sr(self) -> float:
        return self.scale**2 / ARCSEC2_PER_SR

    @property
    def rotation(self) -> float | None:
        """Rotation angle [deg] in the convention of jwst cubes (``PC = [[-cos, sin], [sin, cos]]``
        with positive, equal ``CDELT``; 0 for north up, east left) -- None for a grid that is not
        of that form (skewed, mirrored, non-square spaxels)."""
        s = self.scale / 3600.0
        (a, b), (c, d) = self.cd
        angle = math.atan2(c, d)
        expected = np.array([[-math.cos(angle), math.sin(angle)], [math.sin(angle), math.cos(angle)]]) * s
        if not np.allclose(expected, np.array([[a, b], [c, d]]), rtol=0.0, atol=1e-6 * s):
            return None
        return math.degrees(angle)

    def pixel_to_sky(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        """(RA, Dec) [deg] of 0-based pixel coordinates."""
        ra, dec = self.wcs.wcs_pix2world(np.asarray(x, dtype=float), np.asarray(y, dtype=float), 0)
        return np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)

    def sky_to_pixel(self, ra: Any, dec: Any) -> tuple[np.ndarray, np.ndarray]:
        """0-based pixel coordinates of (RA, Dec) [deg]."""
        x, y = self.wcs.wcs_world2pix(np.asarray(ra, dtype=float), np.asarray(dec, dtype=float), 0)
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def pixels_of(self, other: SpatialGrid, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates in this grid of the pixel coordinates ``(x, y)`` of ``other``."""
        return self.sky_to_pixel(*other.pixel_to_sky(x, y))

    def lattice_offset(self, other: SpatialGrid, *, tol: float = LATTICE_TOLERANCE_PIX) -> tuple[int, int] | None:
        """Where ``other`` sits in this grid when both share one pixel lattice.

        ``(x0, y0)``: the pixel of this grid that pixel (0, 0) of ``other``
        coincides with -- every pixel of ``other`` then *is* a pixel of this
        grid, and its data move over by copying. None when the lattices
        differ (another scale or orientation, or the same ones shifted by a
        fraction of a pixel) by more than ``tol`` pixels anywhere on ``other``.
        """
        ny, nx = other.shape
        xs = np.array([0.0, nx - 1.0, 0.0, nx - 1.0])
        ys = np.array([0.0, 0.0, ny - 1.0, ny - 1.0])
        px, py = self.pixels_of(other, xs, ys)
        x0, y0 = round(float(px[0])), round(float(py[0]))
        if np.max(np.abs(px - (x0 + xs))) > tol or np.max(np.abs(py - (y0 + ys))) > tol:
            return None
        return int(x0), int(y0)

    def phase_of(self, other: SpatialGrid) -> tuple[float, float]:
        """Offset of the lattice of ``other`` from this one, in pixels of this grid and within
        (-0.5, 0.5]: where the centre of pixel (0, 0) of ``other`` falls inside a pixel of this
        grid. (0, 0) for a shared lattice; for another scale it only says where that one pixel lies."""
        px, py = self.pixels_of(other, 0.0, 0.0)
        return float(px - round(float(px))), float(py - round(float(py)))

    # ------------------------------------------------------------------ output
    def header_cards(self) -> dict[str, Any]:
        """The celestial WCS keywords of this grid, in the form jwst cubes carry them (positive
        ``CDELT`` in degrees and a ``PC`` matrix; a grid not of the jwst form keeps unit ``CDELT``
        of one arcsec and puts everything into ``PC``)."""
        rotation = self.rotation
        if rotation is not None:
            scale = self.scale / 3600.0
            angle = math.radians(rotation)
            pc = ((-math.cos(angle), math.sin(angle)), (math.sin(angle), math.cos(angle)))
            if rotation == 0.0:
                pc = ((-1.0, 0.0), (0.0, 1.0))
        else:
            scale = 1.0 / 3600.0
            pc = tuple(tuple(v / scale for v in row) for row in self.cd)   # type: ignore[assignment]
        return {"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CUNIT1": "deg", "CUNIT2": "deg",
                "CRVAL1": self.crval[0], "CRVAL2": self.crval[1], "CRPIX1": self.crpix[0], "CRPIX2": self.crpix[1],
                "CDELT1": scale, "CDELT2": scale,
                "PC1_1": pc[0][0], "PC1_2": pc[0][1], "PC2_1": pc[1][0], "PC2_2": pc[1][1]}

    def cube_build_parameters(self) -> dict[str, Any]:
        """This grid as parameters of the official ``cube_build`` step.

        ``cube_build`` lays a cube out around its tangent point: square
        spaxels of ``scalexy`` arcsec, an odd number of them along each axis
        (``nspax_x``, ``nspax_y``), the tangent point (``ra_center``,
        ``dec_center``) on the central one, rotated by ``cube_pa`` when not
        north-up. A grid of that form -- every grid :func:`joint_grid`
        returns from jwst cubes is -- maps onto these parameters exactly, so
        cubes built with them share this grid pixel for pixel. Raises for any
        other grid.
        """
        ny, nx = self.shape
        rotation = self.rotation
        if rotation is None:
            raise ValueError("cube_build needs square spaxels in the jwst orientation convention; this grid has "
                             f"the pixel matrix {self.cd}")
        if nx % 2 == 0 or ny % 2 == 0 or abs(self.crpix[0] - (nx + 1) / 2) > 1e-6 or abs(self.crpix[1] - (ny + 1) / 2) > 1e-6:
            raise ValueError("cube_build needs an odd number of spaxels with the tangent point on the central one; "
                             f"this grid is {nx} x {ny} with CRPIX {self.crpix}")
        pars: dict[str, Any] = {"scalexy": self.scale, "ra_center": self.crval[0], "dec_center": self.crval[1],
                                "nspax_x": nx, "nspax_y": ny}
        if abs(rotation) > 1e-9:
            pars["cube_pa"] = rotation
        return pars

    def describe(self) -> str:
        ny, nx = self.shape
        rotation = self.rotation
        return (f"{nx} x {ny} spaxels of {self.scale:.4f}\" ({nx * self.scale:.2f}\" x {ny * self.scale:.2f}\"), tangent point "
                f"{self.crval[0]:.7f} {self.crval[1]:+.7f} on pixel ({self.crpix[0]:g}, {self.crpix[1]:g})"
                + (f", rotated {rotation:.3f} deg" if rotation else ""))


def data_bounds(valid: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box ``(x_min, x_max, y_min, y_max)`` (inclusive, 0-based) of the spaxels of a
    cube that carry data in any plane (``valid``: 2-D, or 3-D and then collapsed); None when
    none does."""
    covered = np.asarray(valid, dtype=bool)
    if covered.ndim == 3:
        covered = covered.any(axis=0)
    ys, xs = np.nonzero(covered)
    if not len(xs):
        return None
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def reference_index(grids: Sequence[SpatialGrid]) -> int:
    """Which of the grids a joint grid is built on: the one with the finest spaxels -- the first
    of them, in the order given, when several share that scale (to :data:`SCALE_TOLERANCE`)."""
    if not grids:
        raise ValueError("a joint grid needs at least one grid")
    scales = [g.scale for g in grids]
    finest = min(scales)
    return next(i for i, s in enumerate(scales) if s <= finest * (1.0 + SCALE_TOLERANCE))


def joint_grid(grids: Sequence[SpatialGrid], bounds: Sequence[tuple[int, int, int, int] | None] | None = None) -> SpatialGrid:
    """The grid of a joint cube: the finest spaxel scale over the union of the fields.

    The joint grid is the pixel lattice of the *reference* grid
    (:func:`reference_index`: the one with the finest spaxels) -- its tangent
    point, orientation, spaxel scale and the positions of its pixel centres
    -- extended until it holds the footprint of every grid (``bounds``: the
    bounding box of the spaxels with data in each, from :func:`data_bounds`;
    the whole array when None). So the best-sampled segment is never
    resampled: its pixels are pixels of the joint grid, and a cube that
    already lies on the joint grid stays on it.

    When the tangent point of the reference sits on a pixel centre -- as in
    every cube ``cube_build`` makes -- the grid is laid out the way
    ``cube_build`` lays cubes out: an odd number of spaxels along each axis,
    the tangent point on the central one. That makes it expressible as
    ``cube_build`` parameters (:meth:`SpatialGrid.cube_build_parameters`);
    it is the smallest grid of that form that holds every footprint. For a
    reference laid out otherwise the grid is the smallest window of its
    lattice that holds them (and cannot be handed to ``cube_build``).
    """
    reference = grids[reference_index(grids)]
    x_min = y_min = math.inf
    x_max = y_max = -math.inf
    for i, grid in enumerate(grids):
        box = bounds[i] if bounds is not None else None
        if box is None:
            if bounds is not None:
                continue                                                      # a cube without data needs no room
            box = (0, grid.shape[1] - 1, 0, grid.shape[0] - 1)
        x_lo, x_hi, y_lo, y_hi = box
        # the outline of the footprint: the outer edges of its pixels, sampled along each side
        # (a tangent-plane rectangle stays one, but the corners alone would be enough only for it)
        xs = np.linspace(x_lo - 0.5, x_hi + 0.5, 9)
        ys = np.linspace(y_lo - 0.5, y_hi + 0.5, 9)
        edge_x = np.concatenate([xs, xs, np.full(9, x_lo - 0.5), np.full(9, x_hi + 0.5)])
        edge_y = np.concatenate([np.full(9, y_lo - 0.5), np.full(9, y_hi + 0.5), ys, ys])
        px, py = reference.pixels_of(grid, edge_x, edge_y)                    # in pixels of the reference lattice
        x_min, x_max = min(x_min, float(px.min())), max(x_max, float(px.max()))
        y_min, y_max = min(y_min, float(py.min())), max(y_max, float(py.max()))
    if not math.isfinite(x_min):
        raise ValueError("none of the grids carries data")
    # pixel i of the lattice spans [i - 0.5, i + 0.5]: the pixels that hold the outline
    tol = LATTICE_TOLERANCE_PIX
    ix_lo, ix_hi = math.floor(x_min + 0.5 + tol), math.ceil(x_max - 0.5 - tol)
    iy_lo, iy_hi = math.floor(y_min + 0.5 + tol), math.ceil(y_max - 0.5 - tol)
    tx, ty = reference.crpix[0] - 1.0, reference.crpix[1] - 1.0               # the tangent point, in those pixels
    if abs(tx - round(tx)) <= tol and abs(ty - round(ty)) <= tol:
        # on a pixel centre: the same number of spaxels on either side of it, as cube_build lays a cube out
        na = max(round(tx) - ix_lo, ix_hi - round(tx), 0)
        nb = max(round(ty) - iy_lo, iy_hi - round(ty), 0)
        ix_lo, ix_hi, iy_lo, iy_hi = round(tx) - na, round(tx) + na, round(ty) - nb, round(ty) + nb
    return SpatialGrid(reference.crval, (reference.crpix[0] - ix_lo, reference.crpix[1] - iy_lo), reference.cd,
                       (iy_hi - iy_lo + 1, ix_hi - ix_lo + 1))
