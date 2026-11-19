"""The NIRSpec detector gap: test the pixels next to it against the other exposures, keep the flux.

Next to the gap a cube is thinly sampled. With the high-resolution gratings
every NIRSpec IFU slice runs off NRS1 and resumes on NRS2, the 30 slices
cross the gap at staggered wavelengths (G235H/F170LP: NRS1 ends at
2.36-2.43 um, NRS2 starts at 2.42-2.49 um), and every dither moves a sky
position onto other slices. In the combined cube the gap of a spaxel is the
union of the gaps of the slices feeding it, and its edges are a staircase:
approaching the gap the exposures drop out one after the other, so the
voxels of the staircase (15-30 planes on each side, more at the field edges)
are built from a few exposures down to single pixels. Their noise grows
accordingly (2-5 times the noise of the same spaxel away from the gap in
ESO-Ha 569/574) and nothing marks them in the cube.

What those voxels hold is nevertheless physical. Compared with the other
exposures at the same sky position and wavelength, the detector pixels that
build them -- the "staircase pixels" -- agree as well as any pixel away from
the gap (ESO-Ha 574 G235H: 93 % of them testable, robust scatter 0.90 of
their errors, 0.01 % beyond 5 sigma; interior pixels 0.92 and 0.02 %), and
the detector-edge pixels are not biased. Masking by coverage -- dropping what
not all exposures reach, or voxels below some fraction of the usual weight --
removes real flux: the emission of the part of the disk the moving gap has
not yet covered (5-70 % of the H2 1-0 Q-line flux next to the gap in
ESO-Ha 574, even at 5 % of the usual weight). So this module never flags by
coverage.

:class:`FlagGapOutliers` (``flag_gap_outliers``) tests instead. For every
group of exposures combined into one cube (``group_by``) it measures on a
fine sky grid how far towards the gap each exposure reaches and from that
the staircase pixels -- those beyond the wavelength all (``fraction``)
exposures covering their position reach, where ``outlier_detection`` has the
fewest exposures to compare. Each staircase pixel is compared with the other
exposures' pixels at its sky position (within ``footprint``), interpolated to
its wavelength: real emission is in every exposure that sees the position,
an artefact (an unflagged hot pixel, a cosmic ray, a detector-edge effect)
in one. Pixels deviating by more than ``sigma`` (their errors plus a
``rel_tol`` fraction of the reference, for spatial structure below the
footprint) from the median of at least ``min_others`` other exposures get
``DO_NOT_USE | OUTLIER``; pixels with fewer exposures to compare are kept --
they are the only data there -- and counted as unverified.

The product (``<group>_gapcov.fits``, one per group) records the coverage on
the sky grid, with a celestial TAN WCS, so later steps can tell how thinly a
position is sampled at a wavelength (:func:`gap_limits_at`) without the cal
files, and ``qa_gap_coverage`` can draw it:

    LIMLO    f8 (ny, nx)  last wavelength [um] before the gap that the required exposures reach
    LIMHI    f8 (ny, nx)  first wavelength [um] after the gap that they reach
    REACHLO  f8 (ny, nx)  the reddest wavelength any exposure reaches before the gap
    REACHHI  f8 (ny, nx)  the bluest wavelength any exposure reaches after it
    NEXPLO   i2 (ny, nx)  exposures covering the cell on the blue side (NRS1)
    NEXPHI   i2 (ny, nx)  exposures covering the cell on the red side (NRS2)
    FILES    bintable     one row per cal file: pixels examined, in the staircase, tested, flagged

(NaN / 0 where no exposure covers a cell; between LIMLO and REACHLO, and
between REACHHI and LIMHI, a position is seen by only some exposures.)
Heavy imports stay inside the functions.
"""

from __future__ import annotations

import logging
import math
import re
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field, field_validator

from .naming import derived_name, slug
from .steps.base import RunContext, Step, StepParams

log = logging.getLogger(__name__)

GAPCOV_SUFFIX = "gapcov"   # *_gapcov.fits: the per-group coverage product

#: Detector holding the blue side of the gap (its spectra end there) and the red side (they start there).
BLUE_DETECTOR, RED_DETECTOR = "NRS1", "NRS2"

#: Default exposure grouping: the exposures one ``calwebb_spec3`` cube is built from (the presets'
#: spec3 ``group_by``), science and background apart.
DEFAULT_GROUP_BY: tuple[str, ...] = ("PROGRAM", "OBSERVTN", "GRATING", "FILTER", "BKGDTARG")

#: Columns of the FILES table.
FILE_COLUMNS: tuple[str, ...] = ("file", "exposure", "detector", "n_examined", "n_staircase", "n_tested",
                                 "n_flagged", "edge_min", "edge_max")

#: Image extensions of the product (name, dtype).
PRODUCT_MAPS: tuple[tuple[str, str], ...] = (("LIMLO", "f8"), ("LIMHI", "f8"), ("REACHLO", "f8"), ("REACHHI", "f8"),
                                             ("NEXPLO", "i2"), ("NEXPHI", "i2"))


# --------------------------------------------------------------------------- small helpers
def dq_value(names: list[str] | tuple[str, ...]) -> int:
    """The DQ bit mask of the named jwst pixel flags (``DO_NOT_USE`` is always included)."""
    from stdatamodels.jwst.datamodels.dqflags import pixel

    value = int(pixel["DO_NOT_USE"])
    for name in names:
        key = str(name).strip().upper()
        if key not in pixel:
            raise ValueError(f"unknown DQ flag {name!r}; known: {', '.join(sorted(pixel))}")
        value |= int(pixel[key])
    return value


def exposure_of(path: str | Path) -> str:
    """The exposure a cal file belongs to: its name without detector and suffix
    (``jw05299004001_03101_00001_nrs1_cal.fits`` -> ``jw05299004001_03101_00001``)."""
    name = Path(path).name
    m = re.match(r"(.+?)_(?:nrs[12]|mirifu\w*|mirimage)(?:_|\.)", name, flags=re.IGNORECASE)
    return m.group(1) if m else Path(path).stem


def group_key(header: Any, keys: list[str] | tuple[str, ...]) -> tuple[Any, ...]:
    """The values of ``keys`` in a primary header, normalised (strings lower-cased)."""
    out = []
    for key in keys:
        val = header.get(str(key).upper())
        out.append(str(val).strip().lower() if isinstance(val, str) else val)
    return tuple(out)


def group_name(header: Any, keys: list[str] | tuple[str, ...]) -> str:
    """A level-3-like base name for a group of exposures, e.g. ``jw05299-o004_nirspec_g235h-f170lp``
    (other grouping keywords are appended; ``_bkg`` for background exposures)."""
    keys = [str(k).upper() for k in keys]
    parts = []
    prog = header.get("PROGRAM")
    head = f"jw{str(prog).strip()}" if prog is not None else "jw"
    if "OBSERVTN" in keys and header.get("OBSERVTN") is not None:
        head += f"-o{str(header['OBSERVTN']).strip()}"
    parts.append(head)
    parts.append(str(header.get("INSTRUME", "nirspec")).strip().lower())
    optic = "-".join(str(header[k]).strip().lower() for k in ("GRATING", "FILTER") if k in keys and header.get(k))
    if optic:
        parts.append(optic)
    for k in keys:
        if k in ("PROGRAM", "OBSERVTN", "GRATING", "FILTER", "BKGDTARG", "INSTRUME"):
            continue
        if header.get(k) is not None:
            parts.append(slug(f"{k.lower()}{str(header[k]).strip().lower()}"))
    if "BKGDTARG" in keys and bool(header.get("BKGDTARG")):
        parts.append("bkg")
    return "_".join(p for p in parts if p)


def _usable(sci: np.ndarray, dq: np.ndarray) -> np.ndarray:
    return np.isfinite(sci) & ((np.asarray(dq) & 1) == 0)


# --------------------------------------------------------------------------- the gap window
def slice_edges(path: str | Path) -> dict[str, Any]:
    """Where the spectra of one cal file meet the gap, from its WAVELENGTH (and REGIONS) extension.

    Returns ``detector``, ``edges`` (per slice -- per row without REGIONS --
    the reddest usable wavelength on NRS1 or the bluest on NRS2 [um]) and
    ``dispersion`` (median wavelength step between neighbouring columns
    [um/pix]); ``edges`` is None when the file carries no wavelength image
    (the WCS is evaluated instead, :func:`slice_edges_from_wcs`).
    """
    from astropy.io import fits

    with fits.open(path) as hdul:
        detector = str(hdul[0].header.get("DETECTOR", "")).strip().upper()
        out: dict[str, Any] = {"detector": detector, "edges": None, "dispersion": np.nan}
        if "WAVELENGTH" not in hdul or detector not in (BLUE_DETECTOR, RED_DETECTOR):
            return out
        wl = np.asarray(hdul["WAVELENGTH"].data, dtype=float)
        good = _usable(np.asarray(hdul["SCI"].data), np.asarray(hdul["DQ"].data)) & np.isfinite(wl) & (wl > 0)
        if not good.any():
            return out
        regions = np.asarray(hdul["REGIONS"].data) if "REGIONS" in hdul else None
    wlg = np.where(good, wl, np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings("ignore", message=".*All-NaN.*")
        out["dispersion"] = float(np.nanmedian(np.abs(np.diff(wlg, axis=1))))
        if regions is not None and regions.shape == wl.shape and (regions[good] > 0).any():
            ids = regions[good]
            vals = wl[good]
            order = np.argsort(ids, kind="stable")
            ids, vals = ids[order], vals[order]
            starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
            reduce = np.maximum.reduceat if detector == BLUE_DETECTOR else np.minimum.reduceat
            edges = reduce(vals, starts)[ids[starts] > 0]
        else:
            per_row = np.nanmax(wlg, axis=1) if detector == BLUE_DETECTOR else np.nanmin(wlg, axis=1)
            edges = per_row[np.isfinite(per_row)]
    out["edges"] = np.asarray(edges, dtype=float)
    return out


def slice_edges_from_wcs(model: Any, *, step: int = 4) -> dict[str, Any]:
    """:func:`slice_edges` for a file without a wavelength image: the WCS evaluated on every
    ``step``-th column (NIRSpec IFU slice by slice), per row of every slice."""
    from gwcs.wcstools import grid_from_bounding_box

    detector = str(model.meta.instrument.detector or "").upper()
    usable = _usable(np.asarray(model.data), np.asarray(model.dq))
    edges: list[float] = []
    steps: list[float] = []
    for w in _slice_wcs(model):
        bbox = getattr(w, "bounding_box", None)
        if bbox is None:
            ny, nx = model.dq.shape
            x, y = np.meshgrid(np.arange(0, nx, step, dtype=float), np.arange(ny, dtype=float))
        else:
            x, y = (np.asarray(v, float) for v in grid_from_bounding_box(bbox, step=(step, 1)))
        with np.errstate(all="ignore"):
            lam = np.asarray(w(x, y, with_bounding_box=True)[2], dtype=float)
        if np.isfinite(lam).any() and np.nanmedian(lam) < 1e-3:
            lam = lam * 1e6
        xi, yi = np.rint(x).astype(int), np.rint(y).astype(int)
        inside = (xi >= 0) & (xi < usable.shape[1]) & (yi >= 0) & (yi < usable.shape[0])
        ok = np.isfinite(lam) & inside
        ok[inside] &= usable[yi[inside], xi[inside]]
        lam = np.where(ok, lam, np.nan)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.filterwarnings("ignore", message=".*All-NaN.*")
            per_row = np.nanmax(lam, axis=1) if detector == BLUE_DETECTOR else np.nanmin(lam, axis=1)
            steps.append(float(np.nanmedian(np.abs(np.diff(lam, axis=1)))) / step)
        edges.extend(per_row[np.isfinite(per_row)].tolist())
    disp = float(np.nanmedian(steps)) if steps else np.nan
    return {"detector": detector, "edges": np.asarray(edges, float) if edges else None, "dispersion": disp}


def gap_window(edges_blue: np.ndarray, edges_red: np.ndarray, dispersion: float, pad: int = 3) -> tuple[float, float]:
    """The wavelength window in which pixels can be flagged: from the bluest slice end on NRS1 to the
    reddest slice start on NRS2, padded by ``pad`` pixels of dispersion."""
    lo, hi = float(np.min(edges_blue)), float(np.max(edges_red))
    return lo - pad * dispersion, hi + pad * dispersion


def _slice_wcs(model: Any) -> list[Any]:
    wcs = getattr(model.meta, "wcs", None)
    if wcs is None:
        raise ValueError(f"{model.meta.filename}: no WCS; run assign_wcs (calwebb_spec2) first")
    exp_type = str(getattr(model.meta.exposure, "type", "") or "").upper()
    frames = set(getattr(wcs, "available_frames", []) or [])
    if exp_type == "NRS_IFU" and frames & {"gwa", "slicer"}:
        from jwst.assign_wcs.nirspec import nrs_ifu_wcs

        return list(nrs_ifu_wcs(model))
    return [wcs]


# --------------------------------------------------------------------------- the samples
def gap_samples(model: Any, window: tuple[float, float]) -> dict[str, np.ndarray]:
    """The usable detector pixels of ``model`` with a wavelength inside ``window``: their
    ``y``, ``x``, ``ra``, ``dec`` [deg], ``wave`` [um], ``value`` and ``err`` (SCI and ERR; WCS
    evaluated as in :func:`jwstflow.clusters.detector_sky_coordinates`: NIRSpec IFU slice by
    slice, only the columns that reach the window)."""
    from .clusters import detector_sky_coordinates

    sci, dq = np.asarray(model.data), np.asarray(model.dq)
    err = np.asarray(model.err) if getattr(model, "err", None) is not None else np.full(sci.shape, np.nan)
    parts = list(detector_sky_coordinates(model, [window]))
    keys = ("y", "x", "ra", "dec", "wave", "value", "err")
    if not parts:
        return {k: np.empty(0, dtype=int if k in ("y", "x") else float) for k in keys}
    cols = [np.concatenate([p[i] for p in parts]) for i in range(5)]
    y, x, ra, dec, lam = cols
    y, x = y.astype(int), x.astype(int)
    ok = (lam >= window[0]) & (lam <= window[1]) & _usable(sci[y, x], dq[y, x])
    # a pixel can sit in the bounding boxes of two slices: keep it once
    flat = (y * sci.shape[1] + x)[ok]
    _, first = np.unique(flat, return_index=True)
    sel = np.flatnonzero(ok)[first]
    y, x = y[sel], x[sel]
    return {"y": y, "x": x, "ra": ra[sel], "dec": dec[sel], "wave": lam[sel],
            "value": np.asarray(sci[y, x], dtype=float), "err": np.asarray(err[y, x], dtype=float)}


# --------------------------------------------------------------------------- the coverage
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
        from .clusters import sky_offsets

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


# --------------------------------------------------------------------------- the test
def cross_exposure_deviation(samples: list[dict[str, Any]], tested: list[np.ndarray], grid: CoverageGrid, *,
                             footprint: float = 0.075, dispersion: float, rel_tol: float = 0.2,
                             neighbours: int = 64, chunk: int = 50_000) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each tested pixel against the other exposures at its sky position and wavelength.

    Every other exposure contributes one value: its pixels within
    ``footprint`` arcsec on the sky and one ``dispersion`` step in wavelength,
    weighted by a tent in both (in wavelength that is the linear
    interpolation between the two pixels bracketing the tested one). The
    reference is the median over the exposures; the deviation

        z = (value - ref) / sqrt(err**2 + err_ref**2 + (rel_tol * ref)**2)

    with ``err_ref`` = 1.2533 x the median error of the exposures / sqrt(n).
    Returns per cal file ``(z, n)``: NaN and 0 for pixels not tested.
    """
    from scipy.spatial import cKDTree

    exposures = sorted({s["exposure"] for s in samples})
    cols = {k: [] for k in ("east", "north", "wave", "value", "err", "exp")}
    for s in samples:
        east, north = CoverageGrid._offsets(s["ra"], s["dec"], grid.ra0, grid.dec0)
        cols["east"].append(east)
        cols["north"].append(north)
        for k in ("wave", "value", "err"):
            cols[k].append(np.asarray(s[k], float))
        cols["exp"].append(np.full(s["wave"].size, exposures.index(s["exposure"])))
    a = {k: np.concatenate(v) if v else np.empty(0) for k, v in cols.items()}
    a["exp"] = a["exp"].astype(int)
    usable = np.isfinite(a["value"]) & np.isfinite(a["err"]) & (a["err"] > 0)
    pts = np.column_stack([a["east"] / footprint, a["north"] / footprint, a["wave"] / dispersion])
    out = [(np.full(s["wave"].size, np.nan), np.zeros(s["wave"].size, dtype=int)) for s in samples]
    if not usable.any():
        return out
    idx_usable = np.flatnonzero(usable)
    tree = cKDTree(pts[usable])
    n_exp = len(exposures)
    offsets = np.cumsum([0] + [s["wave"].size for s in samples])
    for f, (s, sel) in enumerate(zip(samples, tested, strict=True)):
        which = offsets[f] + np.flatnonzero(sel & usable[offsets[f]:offsets[f + 1]])
        for lo in range(0, which.size, chunk):
            q = which[lo:lo + chunk]
            dist, nb = tree.query(pts[q], k=min(neighbours, idx_usable.size), distance_upper_bound=1.5)
            dist, nb = np.atleast_2d(dist), np.atleast_2d(nb)
            valid = np.isfinite(dist)
            g = idx_usable[np.where(valid, nb, 0)]
            dsp = np.hypot(a["east"][g] - a["east"][q, None], a["north"][g] - a["north"][q, None]) / footprint
            dwl = np.abs(a["wave"][g] - a["wave"][q, None]) / dispersion
            w = np.clip(1.0 - dsp, 0, None) * np.clip(1.0 - dwl, 0, None) * valid * (a["exp"][g] != a["exp"][q, None])
            flat = (np.arange(q.size)[:, None] * n_exp + a["exp"][g]).ravel()
            size = q.size * n_exp
            sw = np.bincount(flat, w.ravel(), minlength=size).reshape(q.size, n_exp)
            swv = np.bincount(flat, (w * a["value"][g]).ravel(), minlength=size).reshape(q.size, n_exp)
            swe = np.bincount(flat, (w * a["err"][g]).ravel(), minlength=size).reshape(q.size, n_exp)
            have = sw > 0.25
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.filterwarnings("ignore", message=".*All-NaN.*")
                vals = np.where(have, swv / np.where(have, sw, 1.0), np.nan)
                errs = np.where(have, swe / np.where(have, sw, 1.0), np.nan)
                n = have.sum(axis=1)
                ref = np.nanmedian(vals, axis=1)
                err_ref = 1.2533 * np.nanmedian(errs, axis=1) / np.sqrt(np.clip(n, 1, None))
                z = (a["value"][q] - ref) / np.sqrt(a["err"][q] ** 2 + err_ref ** 2 + (rel_tol * ref) ** 2)
            local = q - offsets[f]
            out[f][0][local] = np.where(n > 0, z, np.nan)
            out[f][1][local] = n
    return out


# --------------------------------------------------------------------------- the product
def write_gap_product(path: Path, header: Any, limits: dict[str, Any], files: list[dict[str, Any]],
                      keys: dict[str, Any] | None = None) -> Path:
    """Write a ``*_gapcov.fits`` product (layout in the module docstring)."""
    from astropy.io import fits
    from astropy.table import Table

    prim = fits.PrimaryHDU(header=fits.Header())
    for card in ("TELESCOP", "INSTRUME", "EXP_TYPE", "PROGRAM", "OBSERVTN", "VISIT", "GRATING", "FILTER",
                 "BKGDTARG", "TARGPROP", "TARG_RA", "TARG_DEC", "PATTTYPE", "NUMDTHPT", "CAL_VER", "CRDS_CTX"):
        if card in header:
            prim.header[card] = header[card]
    for k, v in (keys or {}).items():
        prim.header[k] = v
    hdr2d = limits["grid"].header()
    hdus: list[Any] = [prim]
    for name, dtype in PRODUCT_MAPS:
        h = hdr2d.copy()
        if dtype.startswith("f"):
            h["BUNIT"] = "um"
        hdus.append(fits.ImageHDU(np.asarray(limits[name.lower()], dtype=dtype), header=h, name=name))
    table = {col: [f[col] for f in files] for col in FILE_COLUMNS} if files else {c: [] for c in FILE_COLUMNS}
    hdus.append(fits.BinTableHDU(Table(table), name="FILES"))
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return Path(path)


def read_gap_product(path: str | Path) -> dict[str, Any]:
    """A ``*_gapcov.fits`` product as plain arrays: the maps (lower-case names), ``header``
    (primary), ``map_header`` (the grid's WCS), ``files`` (table)."""
    from astropy.io import fits
    from astropy.table import Table

    with fits.open(path) as hdul:
        out: dict[str, Any] = {"path": Path(path), "header": hdul[0].header.copy(),
                               "map_header": hdul["LIMLO"].header.copy(),
                               "files": Table(hdul["FILES"].data) if "FILES" in hdul else Table()}
        for name, _ in PRODUCT_MAPS:
            out[name.lower()] = np.asarray(hdul[name].data)
    return out


def gap_limits_at(product: dict[str, Any] | str | Path, ra: Any, dec: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(limlo, limhi)`` [um] of a gap product at sky positions (degrees): at these positions all
    (the required fraction of) exposures contribute for wavelengths <= limlo and >= limhi; in
    between only some do (thinner sampling, more noise). NaN outside the grid or where no
    exposure covers the position."""
    from astropy.wcs import WCS

    prod = read_gap_product(product) if not isinstance(product, dict) else product
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wcs = WCS(prod["map_header"])
    px, py = wcs.all_world2pix(np.asarray(ra, float), np.asarray(dec, float), 0)
    ix, iy = np.rint(px).astype(int), np.rint(py).astype(int)
    ny, nx = prod["limlo"].shape
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    lo = np.full(np.shape(ix), np.nan)
    hi = np.full(np.shape(ix), np.nan)
    lo[ok] = prod["limlo"][iy[ok], ix[ok]]
    hi[ok] = prod["limhi"][iy[ok], ix[ok]]
    return lo, hi


# --------------------------------------------------------------------------- the step
class FlagGapOutliers(Step):
    """Test the pixels next to the NIRSpec detector gap against the other exposures; flag only those
    that disagree (DQ-edited cal copies) and record the coverage per group.

    One task receives every cal file (``batch = "all"``) and splits them into
    the groups combined into one cube (``group_by``; the default keeps
    observations, gratings and science/background apart). Per group: the gap
    window is read from the files' WAVELENGTH/REGIONS images (bluest NRS1
    slice end to reddest NRS2 slice start); the WCS of every file is
    evaluated over the columns reaching it; on a ``cell``-arcsec sky grid
    each exposure's reach towards the gap is taken from its pixels within
    ``footprint`` arcsec of every cell, and the coverage limit is the reach
    that ``fraction`` of the exposures covering the cell attain. The
    *staircase pixels* beyond that limit -- the ones that build thinly
    sampled voxels, where ``outlier_detection`` compares the fewest
    exposures -- are each tested against the other exposures at their sky
    position, interpolated to their wavelength: a pixel deviating by more
    than ``sigma`` from the median of at least ``min_others`` exposures (its
    error, the reference's and ``rel_tol`` times the reference in quadrature)
    gets ``dq_flags`` (default ``DO_NOT_USE | OUTLIER``). Everything else is
    kept, also the staircase pixels no other exposure can check: thinly
    sampled data are noisier, not wrong, and masking them by coverage removes
    the physical flux of whatever the moving gap has not yet covered.

    Every input is copied into the stage directory (files outside any
    NIRSpec IFU group with both detectors unchanged), keeping the official
    ``_cal`` name; JWFGPNST / JWFGPNTS / JWFGPNPX (staircase pixels, tested,
    flagged) and the parameters in the primary header record the step. Each
    group also gets a ``*_gapcov.fits`` product: the coverage limits and the
    single-exposure reach on the sky grid (TAN WCS), exposure counts and a
    per-file table; ``jwstflow.gaps.gap_limits_at`` looks a position up and
    ``qa_gap_coverage`` draws it.

    Put the step between stage 2 and the stage-3 cube builds (after other
    cal-level flagging such as ``flag_frames``). For a mosaic, group the
    exposures as the cube does (``group_by``).
    """

    name = "flag_gap_outliers"
    level = 2
    batch = "all"
    inputs = ("*_cal.fits",)
    outputs = (GAPCOV_SUFFIX,)
    writes_official_products = True   # DQ-edited copies keep the official _cal name; JWFGP* records the edit
    version = "1"

    class Params(StepParams):
        sigma: float = Field(5.0, gt=0, description="deviation from the other exposures that flags a pixel")
        rel_tol: float = Field(0.2, ge=0, description="fraction of the reference added to the noise "
                                                     "(spatial structure below the footprint)")
        min_others: int = Field(2, ge=2, description="other exposures needed to test a pixel")
        fraction: float = Field(1.0, gt=0, le=1, description="fraction of the exposures covering a position "
                                                             "that sets its coverage limit")
        footprint: float = Field(0.075, gt=0, description="radius [arcsec] within which an exposure's pixels "
                                                          "cover a sky position (its nearest pixels: half the "
                                                          "diagonal of the 0.1 arcsec pixel pitch)")
        cell: float = Field(0.025, gt=0, description="sky grid cell [arcsec] (finer than the footprint)")
        group_by: list[str] = Field(default_factory=lambda: list(DEFAULT_GROUP_BY),
                                    description="header keywords of the exposures combined into one cube")
        dq_flags: list[str] = Field(default_factory=lambda: ["DO_NOT_USE", "OUTLIER"],
                                    description="DQ flags set on the flagged pixels (DO_NOT_USE always)")

        @field_validator("group_by")
        @classmethod
        def _keys_upper(cls, v: list[str]) -> list[str]:
            if not v:
                raise ValueError("group_by needs at least one keyword")
            return [str(k).strip().upper() for k in v]

    def run(self, inputs: list[Path], ctx: RunContext, *, sigma: float = 5.0, rel_tol: float = 0.2,
            min_others: int = 2, fraction: float = 1.0, footprint: float = 0.075, cell: float = 0.025,
            group_by: list[str] | None = None, dq_flags: list[str] | None = None, **params: Any) -> list[Path]:
        from astropy.io import fits

        if not inputs:
            raise ValueError("flag_gap_outliers received no cal files")
        if cell > footprint:
            raise ValueError(f"cell ({cell} arcsec) must not exceed footprint ({footprint} arcsec)")
        keys = [str(k).upper() for k in (group_by or DEFAULT_GROUP_BY)]
        bits = dq_value(dq_flags if dq_flags is not None else ["DO_NOT_USE", "OUTLIER"])
        test = {"sigma": sigma, "rel_tol": rel_tol, "min_others": min_others, "fraction": fraction,
                "footprint": footprint, "cell": cell}
        groups: dict[tuple[Any, ...], list[tuple[Path, Any]]] = {}
        passthrough: list[Path] = []
        for path in sorted(Path(p) for p in inputs):
            hdr = fits.getheader(path)
            if str(hdr.get("EXP_TYPE", "")).upper() != "NRS_IFU":
                passthrough.append(path)
                continue
            groups.setdefault(group_key(hdr, keys), []).append((path, hdr))
        out: list[Path] = []
        for path in passthrough:
            log.info("%s: not a NIRSpec IFU exposure; copied unchanged", path.name)
            out.append(self._copy(path, ctx, counts=(0, 0, 0), product="none", bits=bits, test=test))
        for members in groups.values():
            out.extend(self._group(members, ctx, keys=keys, bits=bits, test=test))
        return out

    # -- one group of exposures -------------------------------------------------------------
    def _group(self, members: list[tuple[Path, Any]], ctx: RunContext, *, keys: list[str], bits: int,
               test: dict[str, Any]) -> list[Path]:
        from stdatamodels.jwst import datamodels

        first_hdr = members[0][1]
        name = group_name(first_hdr, keys)
        detectors = {str(h.get("DETECTOR", "")).upper() for _, h in members}
        if not {BLUE_DETECTOR, RED_DETECTOR} <= detectors:
            log.info("%s: %d file(s), detectors %s -- no gap between NRS1 and NRS2; copied unchanged",
                     name, len(members), ",".join(sorted(detectors)) or "?")
            return [self._copy(p, ctx, counts=(0, 0, 0), product="none", bits=bits, test=test) for p, _ in members]

        # 1. the gap window from the wavelength images (the WCS where a file has none)
        edges: dict[Path, dict[str, Any]] = {}
        for path, _ in members:
            e = slice_edges(path)
            if e["edges"] is None:
                with datamodels.open(path) as model:
                    e = slice_edges_from_wcs(model)
            edges[path] = e
        blue = [e["edges"] for e in edges.values() if e["detector"] == BLUE_DETECTOR and e["edges"] is not None
                and e["edges"].size]
        red = [e["edges"] for e in edges.values() if e["detector"] == RED_DETECTOR and e["edges"] is not None
               and e["edges"].size]
        if not blue or not red:
            log.warning("%s: no usable pixels on one side of the gap; copied unchanged", name)
            return [self._copy(p, ctx, counts=(0, 0, 0), product="none", bits=bits, test=test) for p, _ in members]
        disp = float(np.nanmedian([e["dispersion"] for e in edges.values()]))
        b, r = np.concatenate(blue), np.concatenate(red)
        window = gap_window(b, r, disp)
        log.info("%s: %d file(s); slice ends NRS1 %.4f-%.4f um, starts NRS2 %.4f-%.4f um; window %.4f-%.4f um",
                 name, len(members), b.min(), b.max(), r.min(), r.max(), *window)

        # 2. the pixels inside the window, on the sky
        samples: list[dict[str, Any]] = []
        for path, hdr in members:
            with datamodels.open(path) as model:
                s = gap_samples(model, window)
            s.update({"path": path, "exposure": exposure_of(path), "detector": str(hdr.get("DETECTOR", "")).upper()})
            samples.append(s)

        # 3. the coverage, the staircase pixels and their test against the other exposures
        limits = coverage_limits(samples, cell=test["cell"], footprint=test["footprint"], fraction=test["fraction"])
        stairs = [staircase_pixels(s, limits) for s in samples]
        results = cross_exposure_deviation(samples, stairs, limits["grid"], footprint=test["footprint"],
                                           dispersion=disp, rel_tol=test["rel_tol"])
        rows: list[dict[str, Any]] = []
        flags: dict[Path, np.ndarray] = {}
        for s, stair, (z, n) in zip(samples, stairs, results, strict=True):
            tested = stair & (n >= test["min_others"])
            flag = tested & (np.abs(z) > test["sigma"])
            flags[s["path"]] = flag
            e = edges[s["path"]]["edges"]
            rows.append({"file": s["path"].name, "exposure": s["exposure"], "detector": s["detector"],
                         "n_examined": int(s["wave"].size), "n_staircase": int(stair.sum()),
                         "n_tested": int(tested.sum()), "n_flagged": int(flag.sum()),
                         "edge_min": float(np.min(e)) if e is not None and e.size else np.nan,
                         "edge_max": float(np.max(e)) if e is not None and e.size else np.nan})
        n_stair = int(sum(r["n_staircase"] for r in rows))
        n_tested = int(sum(r["n_tested"] for r in rows))
        n_flag = int(sum(r["n_flagged"] for r in rows))
        fin = np.isfinite(limits["limlo"]) & np.isfinite(limits["reachlo"])
        fin_hi = np.isfinite(limits["limhi"]) & np.isfinite(limits["reachhi"])
        wid_lo = float(np.median((limits["reachlo"] - limits["limlo"])[fin])) if fin.any() else np.nan
        wid_hi = float(np.median((limits["limhi"] - limits["reachhi"])[fin_hi])) if fin_hi.any() else np.nan
        product = ctx.output_dir / derived_name(f"{name}.fits", GAPCOV_SUFFIX)
        write_gap_product(product, first_hdr, limits, rows, keys={
            "JWFGPGRP": (",".join(keys), "gap: grouping keywords"),
            "JWFGPNEX": (len(limits["exposures"]), "gap: exposures in the group"),
            "JWFGPNST": (n_stair, "gap: staircase pixels, all files"),
            "JWFGPNTS": (n_tested, "gap: staircase pixels tested"),
            "JWFGPNPX": (n_flag, "gap: pixels flagged, all files"),
            "JWFGPWLO": (float(window[0]), "gap: window start [um]"),
            "JWFGPWHI": (float(window[1]), "gap: window end [um]"),
            "JWFGPDLO": (wid_lo, "median partial stretch before the gap [um]"),
            "JWFGPDHI": (wid_hi, "median partial stretch after the gap [um]"),
            **self._param_cards(test, bits),
        })
        log.info("%s: %d exposure(s); partially covered over %.4f um (blue) / %.4f um (red) at the median sky "
                 "cell; %d staircase pixel(s), %d tested against the other exposures, %d flagged -> %s",
                 name, len(limits["exposures"]), wid_lo, wid_hi, n_stair, n_tested, n_flag, product.name)

        # 4. the cal copies
        out: list[Path] = [product]
        for s, row in zip(samples, rows, strict=True):
            flag = flags[s["path"]]
            out.append(self._copy(s["path"], ctx, counts=(row["n_staircase"], row["n_tested"], row["n_flagged"]),
                                  product=product.name, bits=bits, test=test, yx=(s["y"][flag], s["x"][flag])))
        return out

    @staticmethod
    def _param_cards(test: dict[str, Any], bits: int) -> dict[str, Any]:
        return {"JWFGPSIG": (float(test["sigma"]), "gap: flagging threshold [sigma]"),
                "JWFGPTOL": (float(test["rel_tol"]), "gap: relative tolerance"),
                "JWFGPMIN": (int(test["min_others"]), "gap: other exposures needed for a test"),
                "JWFGPFRC": (float(test["fraction"]), "gap: exposure fraction of the limit"),
                "JWFGPFTP": (float(test["footprint"]), "gap: footprint radius [arcsec]"),
                "JWFGPCEL": (float(test["cell"]), "gap: sky grid cell [arcsec]"),
                "JWFGPDQ": (int(bits), "gap: DQ bits set on flagged pixels")}

    @classmethod
    def _copy(cls, src: Path, ctx: RunContext, *, counts: tuple[int, int, int], product: str, bits: int,
              test: dict[str, Any], yx: tuple[np.ndarray, np.ndarray] | None = None) -> Path:
        """Copy a cal file into the stage directory, set ``bits`` on the pixels ``yx`` and record the step."""
        from astropy.io import fits

        dst = ctx.output_dir / src.name
        shutil.copy2(src, dst)
        with fits.open(dst, mode="update") as hdul:
            if yx is not None and len(yx[0]):
                dq = hdul["DQ"].data
                dq[yx] = dq[yx] | np.asarray(bits, dtype=dq.dtype)
            hdr = hdul[0].header
            hdr["JWFGPNST"] = (int(counts[0]), "gap: staircase pixels")
            hdr["JWFGPNTS"] = (int(counts[1]), "gap: staircase pixels tested")
            hdr["JWFGPNPX"] = (int(counts[2]), "gap: pixels flagged")
            for k, v in cls._param_cards(test, bits).items():
                hdr[k] = v
            hdr["JWFGPPRD"] = (product, "gap: coverage product of the group")
        return dst


__all__ = [
    "BLUE_DETECTOR",
    "DEFAULT_GROUP_BY",
    "GAPCOV_SUFFIX",
    "RED_DETECTOR",
    "CoverageGrid",
    "FlagGapOutliers",
    "coverage_limits",
    "cross_exposure_deviation",
    "dq_value",
    "exposure_of",
    "gap_limits_at",
    "gap_samples",
    "gap_window",
    "group_key",
    "group_name",
    "read_gap_product",
    "slice_edges",
    "slice_edges_from_wcs",
    "staircase_pixels",
    "write_gap_product",
]
