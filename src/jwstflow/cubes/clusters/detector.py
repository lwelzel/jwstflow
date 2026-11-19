"""Back to the detector: which pixels of a cal file land on flagged spaxels."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..detector import detector_sky_coordinates
from ..morphology import dilate


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
