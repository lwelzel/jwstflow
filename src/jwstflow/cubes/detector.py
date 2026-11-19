"""The detector pixels behind a cube: their sky positions and wavelengths, and their DQ flags.

What steps share that go from calibrated exposures (``_cal`` files) to the sky and back --
flagging the detector pixels behind bad spaxels (:mod:`jwstflow.cubes.clusters`), testing the
pixels next to the detector gap against the other exposures (:mod:`jwstflow.cubes.gaps`).
Heavy imports (jwst, gwcs) happen inside the functions.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np


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


def slice_wcs(model: Any) -> list[Any]:
    """The WCS objects of an exposure's spectra: one per slice for NIRSpec IFU
    (``jwst.assign_wcs.nirspec.nrs_ifu_wcs``), else the exposure's own ``meta.wcs``."""
    wcs = getattr(model.meta, "wcs", None)
    if wcs is None:
        raise ValueError(f"{model.meta.filename}: no WCS; run assign_wcs (calwebb_spec2) first")
    exp_type = str(getattr(model.meta.exposure, "type", "") or "").upper()
    frames = set(getattr(wcs, "available_frames", []) or [])
    if exp_type == "NRS_IFU" and frames & {"gwa", "slicer"}:
        from jwst.assign_wcs.nirspec import nrs_ifu_wcs

        return list(nrs_ifu_wcs(model))
    return [wcs]


def detector_sky_coordinates(model: Any, windows: list[tuple[float, float]], *, coarse_step: int = 8,
                             ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """(y, x, ra, dec, wavelength) of the detector pixels of ``model`` that may fall
    inside one of the wavelength ``windows`` [um].

    NIRSpec IFU exposures are evaluated slice by slice
    (``jwst.assign_wcs.nirspec.nrs_ifu_wcs``); any other exposure through its
    ``meta.wcs`` and bounding box. A coarse pass (every ``coarse_step``-th
    column, every row) locates the columns whose wavelengths reach a window;
    only those (with a margin) are evaluated at full resolution. Wavelengths
    are returned in microns whatever the WCS emits.
    """
    import warnings

    from gwcs.wcstools import grid_from_bounding_box

    slices = slice_wcs(model)
    ny_det, nx_det = model.dq.shape

    def evaluate(w: Any, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.errstate(all="ignore"):
            ra, dec, lam = w(x, y, with_bounding_box=True)
        ra, dec, lam = (np.asarray(v, dtype=float) for v in (ra, dec, lam))
        if np.isfinite(lam).any() and np.nanmedian(lam) < 1e-3:   # metres -> microns
            lam = lam * 1e6
        return ra, dec, lam

    for w in slices:
        bbox = getattr(w, "bounding_box", None)
        if bbox is None:
            xc, yc = np.meshgrid(np.arange(0, nx_det, coarse_step), np.arange(ny_det))
        else:
            xc, yc = grid_from_bounding_box(bbox, step=(coarse_step, 1))
        xc, yc = np.asarray(xc, float), np.asarray(yc, float)
        if xc.ndim == 1:
            xc, yc = xc[None], yc[None]
        _, _, lam_c = evaluate(w, xc, yc)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.filterwarnings("ignore", message=".*All-NaN.*")
            col_lo, col_hi = np.nanmin(lam_c, axis=0), np.nanmax(lam_c, axis=0)
        # a window may lie entirely between two coarse columns: bracket each column with its neighbour
        span_lo, span_hi = col_lo.copy(), col_hi.copy()
        span_lo[:-1] = np.fmin(col_lo[:-1], col_lo[1:])
        span_hi[:-1] = np.fmax(col_hi[:-1], col_hi[1:])
        keep = np.zeros(xc.shape[1], dtype=bool)
        for wlo, whi in windows:
            keep |= np.isfinite(span_lo) & (span_lo <= whi) & (span_hi >= wlo)
        if not keep.any():
            continue
        cols = xc[0]
        idx = np.flatnonzero(keep)
        x_lo = int(np.floor(cols[max(0, idx.min() - 2)]))
        x_hi = int(np.ceil(cols[min(len(cols) - 1, idx.max() + 2)]))
        if idx.max() + 2 > len(cols) - 1:
            # the coarse grid stops up to coarse_step - 1 columns short of the box's end (on NRS1 the
            # detector gap): evaluate to the detector edge, the bounding box masks what lies beyond
            x_hi = nx_det - 1
        y_lo, y_hi = int(np.floor(np.nanmin(yc))), int(np.ceil(np.nanmax(yc)))
        x_lo, x_hi = max(0, x_lo), min(nx_det - 1, x_hi)
        y_lo, y_hi = max(0, y_lo), min(ny_det - 1, y_hi)
        yy, xx = np.mgrid[y_lo:y_hi + 1, x_lo:x_hi + 1]
        ra, dec, lam = evaluate(w, xx.astype(float), yy.astype(float))
        ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(lam)
        if ok.any():
            yield yy[ok], xx[ok], ra[ok], dec[ok], lam[ok]
