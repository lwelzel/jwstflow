"""Where the spectra meet the gap on the detector: the slice edges, the gap window, the pixels inside it."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ..detector import detector_sky_coordinates, slice_wcs

#: Detector holding the blue side of the gap (its spectra end there) and the red side (they start there).
BLUE_DETECTOR, RED_DETECTOR = "NRS1", "NRS2"


def _usable(sci: np.ndarray, dq: np.ndarray) -> np.ndarray:
    return np.isfinite(sci) & ((np.asarray(dq) & 1) == 0)


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
    for w in slice_wcs(model):
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


def gap_samples(model: Any, window: tuple[float, float]) -> dict[str, np.ndarray]:
    """The usable detector pixels of ``model`` with a wavelength inside ``window``: their
    ``y``, ``x``, ``ra``, ``dec`` [deg], ``wave`` [um], ``value`` and ``err`` (SCI and ERR; WCS
    evaluated by :func:`jwstflow.cubes.detector.detector_sky_coordinates`: NIRSpec IFU slice by
    slice, only the columns that reach the window)."""
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
