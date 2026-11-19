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


def celestial_wcs_from_header(header: Any) -> Any:
    """Celestial (2-D) WCS from the header of a cube extension, whatever its spectral axis is.

    Only the keywords of the two celestial axes are read, so a cube whose
    wavelength axis is tabular (``CTYPE3 = 'WAVE-TAB'``: astropy cannot build
    its WCS from the header alone) works like a linear one.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    hdr = fits.Header()
    for i in (1, 2):
        hdr[f"CTYPE{i}"] = header[f"CTYPE{i}"]
        hdr[f"CRVAL{i}"] = header[f"CRVAL{i}"]
        hdr[f"CRPIX{i}"] = header[f"CRPIX{i}"]
        hdr[f"CUNIT{i}"] = header.get(f"CUNIT{i}") or "deg"
        if f"CDELT{i}" in header:
            hdr[f"CDELT{i}"] = header[f"CDELT{i}"]
        for j in (1, 2):
            for name in (f"PC{i}_{j}", f"CD{i}_{j}"):
                if name in header:
                    hdr[name] = header[name]
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


def hdu_wavelengths(hdul: Any, ext: str = "SCI") -> np.ndarray:
    """Wavelength [um] of every plane of an open cube FITS file: the ``WCS-TABLE`` extension when
    it holds one value per plane (jwst writes it for every cube), else the linear WCS of ``ext``."""
    n = int(hdul[ext].data.shape[0])
    if "WCS-TABLE" in hdul and hdul["WCS-TABLE"].data is not None:
        names = {name.lower(): name for name in hdul["WCS-TABLE"].data.names}
        if "wavelength" in names:
            wave = np.asarray(hdul["WCS-TABLE"].data[names["wavelength"]], dtype=float).ravel()
            if wave.size == n:
                return wave
    header = hdul[ext].header
    if "CDELT3" not in header or "CRVAL3" not in header:
        raise ValueError(f"{ext} has neither a wavelength table nor a linear wavelength axis")
    return float(header["CRVAL3"]) + (np.arange(n) + 1 - float(header.get("CRPIX3", 1.0))) * float(header["CDELT3"])


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


def sky_offsets(ra: np.ndarray, dec: np.ndarray, ra0: float, dec0: float) -> tuple[np.ndarray, np.ndarray]:
    """Tangent-plane offsets [arcsec] (east, north) of positions from ``(ra0, dec0)``."""
    dra = (np.asarray(ra, float) - ra0 + 180.0) % 360.0 - 180.0
    east = dra * np.cos(np.radians(dec0)) * 3600.0
    north = (np.asarray(dec, float) - dec0) * 3600.0
    return east, north



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
