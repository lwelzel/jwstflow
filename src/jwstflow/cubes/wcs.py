"""The grids of IFU cubes: the celestial WCS, the wavelength of every plane, regridding between cubes.

Heavy imports (astropy, scipy) happen inside the functions.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def celestial_wcs(cube: Any) -> Any:
    """Celestial (2-D) WCS of an IFU cube datamodel, built from ``meta.wcsinfo``."""
    from astropy.io import fits
    from astropy.wcs import WCS

    w = cube.meta.wcsinfo
    hdr = fits.Header()
    for i, axis in ((1, "1"), (2, "2")):
        hdr[f"CTYPE{i}"] = getattr(w, f"ctype{axis}")
        hdr[f"CRVAL{i}"] = getattr(w, f"crval{axis}")
        hdr[f"CRPIX{i}"] = getattr(w, f"crpix{axis}")
        hdr[f"CDELT{i}"] = getattr(w, f"cdelt{axis}")
        hdr[f"CUNIT{i}"] = getattr(w, f"cunit{axis}") or "deg"
    for key in ("pc1_1", "pc1_2", "pc2_1", "pc2_2"):
        val = getattr(w, key, None)
        if val is not None:
            hdr[key.upper()] = val
    hdr["NAXIS"] = 2
    return WCS(hdr).celestial


def cube_wavelengths(cube: Any) -> np.ndarray:
    """Wavelength [um] of every plane of an ``IFUCubeModel`` (linear WCS or wavetable)."""
    wavetable = getattr(cube, "wavetable", None)
    if wavetable is not None and len(wavetable) > 0:
        return np.asarray(wavetable["wavelength"], dtype=float).ravel()
    n = cube.data.shape[0]
    w = cube.meta.wcsinfo
    return w.crval3 + (np.arange(n) + 1 - w.crpix3) * w.cdelt3


def pixel_map(src_wcs: Any, dst_wcs: Any, dst_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(y, x) coordinates in `src_wcs` pixels of every pixel centre of the destination grid."""
    yy, xx = np.indices(dst_shape)
    ra, dec = dst_wcs.all_pix2world(xx, yy, 0)
    sx, sy = src_wcs.all_world2pix(ra, dec, 0)
    return np.asarray(sy, float), np.asarray(sx, float)


def resample_image(img: np.ndarray, src_wcs: Any, dst_wcs: Any, dst_shape: tuple[int, int]) -> np.ndarray:
    """Bilinear, NaN-aware resampling of `img` onto the destination grid (outside -> NaN)."""
    from scipy.ndimage import map_coordinates

    sy, sx = pixel_map(src_wcs, dst_wcs, dst_shape)
    finite = np.isfinite(img)
    v = map_coordinates(np.where(finite, img, 0.0), [sy, sx], order=1, mode="constant", cval=0.0)
    w = map_coordinates(finite.astype(float), [sy, sx], order=1, mode="constant", cval=0.0)
    return np.where(w > 0.5, v / np.maximum(w, 1e-12), np.nan)


def sample_mask(mask: np.ndarray, src_wcs: Any, dst_wcs: Any, dst_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour sampling of a boolean mask onto the destination grid (outside -> False)."""
    from scipy.ndimage import map_coordinates

    sy, sx = pixel_map(src_wcs, dst_wcs, dst_shape)
    return map_coordinates(mask.astype("u1"), [sy, sx], order=0, mode="constant", cval=0) > 0


def strip_wave_axis(header: Any) -> Any:
    """Copy of a cube SCI header with the 3rd (wavelength) axis removed, for 2-D extensions."""
    hdr = header.copy()
    for key in ("CTYPE3", "CRVAL3", "CRPIX3", "CDELT3", "CUNIT3", "CRDER3", "NAXIS3",
                "PC1_3", "PC2_3", "PC3_1", "PC3_2", "PC3_3", "CD1_3", "CD2_3", "CD3_1", "CD3_2", "CD3_3",
                "PS3_0", "PS3_1", "PV3_0", "PV3_1"):
        hdr.remove(key, ignore_missing=True, remove_all=True)
    if "WCSAXES" in hdr:
        hdr["WCSAXES"] = 2
    return hdr
