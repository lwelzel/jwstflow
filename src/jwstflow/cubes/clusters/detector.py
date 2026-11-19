"""Back to the detector: which pixels of a cal file land on flagged spaxels."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from ..morphology import dilate


def _dq_bits() -> int:
    try:
        from stdatamodels.jwst.datamodels.dqflags import pixel

        return int(pixel["DO_NOT_USE"]) | int(pixel["OUTLIER"])
    except ImportError:  # pragma: no cover - jwst is a dependency
        return 1 | 16


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

    wcs = getattr(model.meta, "wcs", None)
    if wcs is None:
        raise ValueError(f"{model.meta.filename}: no WCS; run assign_wcs (calwebb_spec2) first")
    exp_type = str(getattr(model.meta.exposure, "type", "") or "").upper()
    frames = set(getattr(wcs, "available_frames", []) or [])
    if exp_type == "NRS_IFU" and frames & {"gwa", "slicer"}:
        from jwst.assign_wcs.nirspec import nrs_ifu_wcs

        slice_wcs = nrs_ifu_wcs(model)
    else:
        slice_wcs = [wcs]
    ny_det, nx_det = model.dq.shape

    def evaluate(w: Any, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.errstate(all="ignore"):
            ra, dec, lam = w(x, y, with_bounding_box=True)
        ra, dec, lam = (np.asarray(v, dtype=float) for v in (ra, dec, lam))
        if np.isfinite(lam).any() and np.nanmedian(lam) < 1e-3:   # metres -> microns
            lam = lam * 1e6
        return ra, dec, lam

    for w in slice_wcs:
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
        y_lo, y_hi = int(np.floor(np.nanmin(yc))), int(np.ceil(np.nanmax(yc)))
        x_lo, x_hi = max(0, x_lo), min(nx_det - 1, x_hi)
        y_lo, y_hi = max(0, y_lo), min(ny_det - 1, y_hi)
        yy, xx = np.mgrid[y_lo:y_hi + 1, x_lo:x_hi + 1]
        ra, dec, lam = evaluate(w, xx.astype(float), yy.astype(float))
        ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(lam)
        if ok.any():
            yield yy[ok], xx[ok], ra[ok], dec[ok], lam[ok]


def detector_pixels_in_masks(model: Any, products: list[dict[str, Any]], *, grow: int = 0, plane_pad: int = 1,
                             ) -> tuple[np.ndarray, list[str]]:
    """Boolean detector image of the pixels that land on flagged spaxels of ``products``,
    and the product names that contributed.

    The cube masks are widened first: ``plane_pad`` planes along wavelength and
    ``grow`` spaxels spatially, so every detector pixel drizzled into a
    flagged spaxel is caught. A pixel is flagged when its wavelength's nearest
    plane and its sky position's nearest spaxel are inside the widened mask.
    """
    from astropy.wcs import WCS
    from scipy.ndimage import maximum_filter1d

    flag = np.zeros(model.dq.shape, dtype=bool)
    targets = []
    for p in products:
        mask = p["mask"]
        if not mask.any():
            continue
        wide = mask.astype("u1")
        if plane_pad > 0:
            wide = maximum_filter1d(wide, size=2 * plane_pad + 1, axis=0, mode="constant")
        wide = wide > 0
        if grow > 0:
            wide = np.stack([dilate(pl, grow) if pl.any() else pl for pl in wide])
        waves = p["waves"]
        step = float(np.median(np.abs(np.diff(waves)))) if waves.size > 1 else 0.0
        planes = np.flatnonzero(wide.any(axis=(1, 2)))
        window = (float(waves[planes].min() - step), float(waves[planes].max() + step))
        hdr = p["mask_header"].copy()
        hdr["NAXIS"] = 3
        wcs2d = WCS(hdr).celestial
        targets.append((p["path"].name, wcs2d, waves, step, wide, window))
    if not targets:
        return flag, []
    used: set[str] = set()
    for yy, xx, ra, dec, lam in detector_sky_coordinates(model, [t[-1] for t in targets]):
        for name, wcs2d, waves, step, wide, window in targets:
            inside = (lam >= window[0]) & (lam <= window[1])
            if not inside.any():
                continue
            k = np.clip(np.rint(np.interp(lam[inside], waves, np.arange(waves.size))).astype(int), 0, waves.size - 1)
            ok = np.abs(waves[k] - lam[inside]) <= max(step, 1e-9)
            with np.errstate(all="ignore"):
                px, py = wcs2d.all_world2pix(ra[inside], dec[inside], 0)
            ix, iy = np.rint(px).astype(int), np.rint(py).astype(int)
            _, ny, nx = wide.shape
            ok &= (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            hit = np.zeros(inside.sum(), dtype=bool)
            hit[ok] = wide[k[ok], iy[ok], ix[ok]]
            if hit.any():
                sel = np.flatnonzero(inside)[hit]
                flag[yy[sel], xx[sel]] = True
                used.add(name)
    return flag, sorted(used)
