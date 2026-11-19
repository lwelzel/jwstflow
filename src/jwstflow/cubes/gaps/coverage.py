"""The coverage on the sky: how far each exposure reaches towards the gap, and the staircase beyond what all reach."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from ..wcs import sky_offsets
from .edges import BLUE_DETECTOR, RED_DETECTOR


def _disk(radius_cells: float) -> np.ndarray:
    k = math.floor(radius_cells + 1e-9)
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    return (yy**2 + xx**2) <= radius_cells**2 + 1e-9


def _kth(stack: np.ndarray, fraction: float, largest: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per cell, the value that at least ``fraction`` of the covering members reach: the k-th largest
    (``largest``) or smallest of the finite values, k = ceil(fraction x n). Returns (value, n)."""
    finite = np.isfinite(stack)
    n = finite.sum(axis=0)
    if stack.shape[0] == 0:
        return np.full(stack.shape[1:], np.nan), n
    fill = -np.inf if largest else np.inf
    srt = np.sort(np.where(finite, stack, fill), axis=0)
    if largest:
        srt = srt[::-1]
    k = np.clip(np.ceil(fraction * n - 1e-9).astype(int) - 1, 0, stack.shape[0] - 1)
    val = np.take_along_axis(srt, k[None], axis=0)[0]
    return np.where(n > 0, val, np.nan), n


class CoverageGrid:
    """A sky grid of ``cell``-arcsec cells around ``(ra0, dec0)``, North up and East to the left
    (``ix`` grows towards the West), covering a set of samples."""

    def __init__(self, ra0: float, dec0: float, cell: float, east_max: float, north_min: float, nx: int, ny: int):
        self.ra0, self.dec0, self.cell = float(ra0), float(dec0), float(cell)
        self.east_max, self.north_min, self.nx, self.ny = float(east_max), float(north_min), int(nx), int(ny)

    @classmethod
    def around(cls, ra: np.ndarray, dec: np.ndarray, cell: float, margin: float) -> CoverageGrid:
        ra0, dec0 = float(np.median(ra)), float(np.median(dec))
        east, north = cls._offsets(ra, dec, ra0, dec0)
        e_max, n_min = float(east.max()) + margin, float(north.min()) - margin
        nx = math.ceil((e_max - (float(east.min()) - margin)) / cell) + 1
        ny = math.ceil((float(north.max()) + margin - n_min) / cell) + 1
        return cls(ra0, dec0, cell, e_max, n_min, nx, ny)

    @staticmethod
    def _offsets(ra: Any, dec: Any, ra0: float, dec0: float) -> tuple[np.ndarray, np.ndarray]:
        return sky_offsets(ra, dec, ra0, dec0)

    def index(self, ra: Any, dec: Any) -> tuple[np.ndarray, np.ndarray]:
        """(iy, ix) of the cells holding sky positions (may fall outside the grid)."""
        east, north = self._offsets(ra, dec, self.ra0, self.dec0)
        return (np.rint((north - self.north_min) / self.cell).astype(int),
                np.rint((self.east_max - east) / self.cell).astype(int))

    def header(self) -> Any:
        """A celestial TAN header of the grid (the offsets are taken as tangent-plane offsets,
        exact to well below a cell over an IFU field)."""
        from astropy.io import fits

        h = fits.Header()
        h["WCSAXES"] = 2
        h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
        h["CUNIT1"], h["CUNIT2"] = "deg", "deg"
        h["CRVAL1"], h["CRVAL2"] = self.ra0, self.dec0
        h["CDELT1"], h["CDELT2"] = -self.cell / 3600.0, self.cell / 3600.0
        # FITS pixels are 1-based: cell ix sits at east = east_max - ix * cell, iy at north_min + iy * cell
        h["CRPIX1"] = 1.0 + self.east_max / self.cell
        h["CRPIX2"] = 1.0 - self.north_min / self.cell
        h["RADESYS"] = "ICRS"
        return h


def coverage_limits(samples: list[dict[str, Any]], *, cell: float = 0.025, footprint: float = 0.075,
                    fraction: float = 1.0) -> dict[str, Any]:
    """The completeness limits on both sides of the gap, on a sky grid.

    ``samples`` holds one entry per cal file: ``exposure``, ``detector`` and
    the arrays of :func:`gap_samples`. For every exposure and side the
    *reach* in every cell is the reddest NRS1 (bluest NRS2) wavelength among
    its samples within ``footprint`` arcsec of the cell; the *limit* is the
    reach that at least ``fraction`` of the exposures covering the cell
    attain (all of them by default). Returns the grid, the exposures and the
    maps of the product (``limlo``, ``limhi``, ``reachlo``, ``reachhi``,
    ``nexplo``, ``nexphi``).
    """
    from scipy.ndimage import maximum_filter, minimum_filter

    ra = np.concatenate([s["ra"] for s in samples]) if samples else np.empty(0)
    dec = np.concatenate([s["dec"] for s in samples]) if samples else np.empty(0)
    if ra.size == 0:
        raise ValueError("coverage_limits: no samples")
    grid = CoverageGrid.around(ra, dec, cell, margin=2 * footprint + cell)
    disk = _disk(footprint / cell)
    exposures = sorted({s["exposure"] for s in samples})
    shape = (len(exposures), grid.ny, grid.nx)
    reach = {BLUE_DETECTOR: np.full(shape, np.nan, dtype="f8"), RED_DETECTOR: np.full(shape, np.nan, dtype="f8")}
    for s in samples:
        det = s["detector"]
        if det not in reach or s["wave"].size == 0:
            continue
        j = exposures.index(s["exposure"])
        iy, ix = grid.index(s["ra"], s["dec"])
        blue = det == BLUE_DETECTOR
        fill = -np.inf if blue else np.inf
        img = np.full((grid.ny, grid.nx), fill)
        (np.maximum if blue else np.minimum).at(img, (iy, ix), s["wave"])
        img = (maximum_filter if blue else minimum_filter)(img, footprint=disk, mode="constant", cval=fill)
        prev = reach[det][j]
        new = np.where(np.isfinite(img), img, np.nan)
        reach[det][j] = np.where(np.isfinite(prev), (np.fmax if blue else np.fmin)(prev, new), new)
    limlo, nexplo = _kth(reach[BLUE_DETECTOR], fraction, largest=True)
    limhi, nexphi = _kth(reach[RED_DETECTOR], fraction, largest=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*All-NaN.*")
        reachlo = np.nanmax(reach[BLUE_DETECTOR], axis=0) if len(exposures) else np.full(shape[1:], np.nan)
        reachhi = np.nanmin(reach[RED_DETECTOR], axis=0) if len(exposures) else np.full(shape[1:], np.nan)
    return {"grid": grid, "exposures": exposures, "limlo": limlo, "limhi": limhi, "reachlo": reachlo,
            "reachhi": reachhi, "nexplo": nexplo.astype("i2"), "nexphi": nexphi.astype("i2")}


def staircase_pixels(sample: dict[str, Any], limits: dict[str, Any]) -> np.ndarray:
    """Boolean over the samples of one cal file: True for the pixels beyond the coverage limit at
    their sky position (NRS1: redder than it, NRS2: bluer) -- the pixels that build voxels only
    some of the exposures reach."""
    if sample["wave"].size == 0:
        return np.zeros(0, dtype=bool)
    grid = limits["grid"]
    iy, ix = grid.index(sample["ra"], sample["dec"])
    inside = (iy >= 0) & (iy < grid.ny) & (ix >= 0) & (ix < grid.nx)
    iy, ix = np.clip(iy, 0, grid.ny - 1), np.clip(ix, 0, grid.nx - 1)
    if sample["detector"] == BLUE_DETECTOR:
        out = sample["wave"] > np.where(np.isfinite(limits["limlo"]), limits["limlo"], np.inf)[iy, ix]
    elif sample["detector"] == RED_DETECTOR:
        out = sample["wave"] < np.where(np.isfinite(limits["limhi"]), limits["limhi"], -np.inf)[iy, ix]
    else:
        out = np.zeros(sample["wave"].shape, dtype=bool)
    return out & inside
