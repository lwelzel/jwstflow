"""Small JWST-like products for tests: a rate/cal image, a resampled image, an IFU cube, an extracted spectrum."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _primary(instrument: str, **keys: Any) -> Any:
    from astropy.io import fits

    hdr = fits.Header()
    base = {"TELESCOP": "JWST", "INSTRUME": instrument, "PROGRAM": "01234", "OBSERVTN": "001", "VISIT": "001",
            "TARGPROP": "TEST-TARGET", "TARG_RA": 10.0, "TARG_DEC": -20.0, "BKGDTARG": False, "IS_IMPRT": False,
            "PATT_NUM": 1, "DATE-OBS": "2024-01-01", "CAL_VER": "3.0.0", "CRDS_CTX": "jwst_1584.pmap"}
    base.update(keys)
    for k, v in base.items():
        hdr[k] = v
    return fits.PrimaryHDU(header=hdr)


def synthetic_image(path: Path, *, instrument: str = "NIRSPEC", shape: tuple[int, int] = (64, 64),
                    value: float = 1.0, **keys: Any) -> Path:
    """A rate/cal-like file with SCI, ERR and DQ extensions."""
    from astropy.io import fits

    keys.setdefault("EXP_TYPE", "NRS_IFU" if instrument == "NIRSPEC" else "MIR_MRS")
    keys.setdefault("DETECTOR", "NRS1" if instrument == "NIRSPEC" else "MIRIFUSHORT")
    sci = np.full(shape, value, dtype="f4")
    fits.HDUList([_primary(instrument, **keys), fits.ImageHDU(sci, name="SCI"), fits.ImageHDU(np.full(shape, 0.1, "f4"), name="ERR"),
                  fits.ImageHDU(np.zeros(shape, "u4"), name="DQ")]).writeto(path, overwrite=True)
    return Path(path)


def synthetic_i2d(path: Path, *, data: np.ndarray | None = None, instrument: str = "NIRCAM", filter: str = "F200W",
                  pupil: str | None = "CLEAR", shape: tuple[int, int] = (120, 100), pix_arcsec: float = 0.031,
                  ra: float = 10.0, dec: float = -20.0, crpix: tuple[float, float] | None = None, error: float = 1.0,
                  n_exp: int = 3, **keys: Any) -> Path:
    """A resampled image as image3 writes it: SCI / ERR / WHT / CON on a north-up TAN grid in MJy/sr.

    ``data`` is the scene (default: a ramp, every pixel different); the grid is centred on
    (``ra``, ``dec``) unless ``crpix`` (1-based x, y) says otherwise. The context image has ``n_exp``
    exposures, exposure k covering all but the last 10 k columns.
    """
    from astropy.io import fits

    if data is None:
        data = np.arange(shape[0] * shape[1], dtype="f4").reshape(shape)
    data = np.asarray(data, dtype="f4")
    ny, nx = data.shape
    keys = {"FILTER": filter, "TARG_RA": ra, "TARG_DEC": dec, "EXP_TYPE": "NRC_IMAGE" if instrument == "NIRCAM"
            else "MIR_IMAGE" if instrument == "MIRI" else "NIS_IMAGE", **keys}
    if pupil is not None:
        keys["PUPIL"] = pupil
    sci = fits.ImageHDU(data, name="SCI")
    cx, cy = crpix if crpix is not None else ((nx + 1) / 2.0, (ny + 1) / 2.0)
    sci.header.update({"BUNIT": "MJy/sr", "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CUNIT1": "deg", "CUNIT2": "deg",
                       "CRVAL1": ra, "CRVAL2": dec, "CRPIX1": cx, "CRPIX2": cy, "CDELT1": -pix_arcsec / 3600.0,
                       "CDELT2": pix_arcsec / 3600.0, "PIXAR_SR": (pix_arcsec / 206264.80624709636) ** 2,
                       "PIXAR_A2": pix_arcsec**2})
    con = np.zeros((1, ny, nx), dtype="i4")
    for k in range(n_exp):
        con[0, :, : max(nx - 10 * k, 1)] |= 1 << k
    fits.HDUList([_primary(instrument, **keys), sci, fits.ImageHDU(np.full(data.shape, error, "f4"), name="ERR"),
                  fits.ImageHDU(np.full(data.shape, 1.0 / error**2, "f4"), name="WHT"),
                  fits.ImageHDU(con, name="CON")]).writeto(path, overwrite=True)
    return Path(path)


def synthetic_cube(path: Path, *, instrument: str = "MIRI", nwave: int = 50, size: int = 40, wave_min: float = 4.9,
                   wave_step: float = 0.01, source_flux_jy: float = 1e-3, fwhm_pix: float = 2.5, background: float = 0.0,
                   pix_arcsec: float = 0.13, **keys: Any) -> Path:
    """An IFU cube (``IFUCubeModel`` when stdatamodels is available, plain FITS otherwise) with a
    Gaussian point source of ``source_flux_jy`` in every plane plus a flat background (MJy/sr)."""
    yy, xx = np.mgrid[:size, :size]
    sig = fwhm_pix / 2.3548
    area_sr = (pix_arcsec / 206265.0) ** 2
    peak = source_flux_jy / (2 * np.pi * sig**2) / area_sr / 1e6
    psf = peak * np.exp(-((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (2 * sig**2)) + background
    data = np.repeat(psf[None], nwave, axis=0).astype("f4")
    try:
        from stdatamodels.jwst import datamodels
    except ImportError:
        datamodels = None
    if datamodels is not None:
        cube = datamodels.IFUCubeModel(data=data, err=np.full_like(data, 1e-3), dq=np.zeros(data.shape, "u4"))
        cube.meta.instrument.name = instrument
        if instrument == "MIRI":
            cube.meta.instrument.channel, cube.meta.instrument.band, cube.meta.exposure.type = "1", "SHORT", "MIR_MRS"
        else:
            cube.meta.instrument.grating, cube.meta.instrument.filter, cube.meta.exposure.type = "G235H", "F170LP", "NRS_IFU"
        cube.meta.target.proposer_name, cube.meta.target.ra, cube.meta.target.dec = "TEST-TARGET", 10.0, -20.0
        w = cube.meta.wcsinfo
        w.ctype1, w.ctype2, w.ctype3 = "RA---TAN", "DEC--TAN", "WAVE"
        w.crval1, w.crval2, w.crval3 = 10.0, -20.0, wave_min
        w.crpix1, w.crpix2, w.crpix3 = size / 2 + 1, size / 2 + 1, 1.0
        w.cdelt1, w.cdelt2, w.cdelt3 = -pix_arcsec / 3600, pix_arcsec / 3600, wave_step
        w.cunit1 = w.cunit2 = "deg"
        cube.meta.photometry.pixelarea_steradians, cube.meta.photometry.pixelarea_arcsecsq = area_sr, pix_arcsec**2
        if keys:  # extra_fits must be assigned whole: appending to a fresh model raises
            cube.extra_fits = {"PRIMARY": {"header": [[k, v, ""] for k, v in keys.items()]}}
        cube.save(str(path))
        return Path(path)
    from astropy.io import fits

    sci = fits.ImageHDU(data, name="SCI")
    for k, v in {"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CTYPE3": "WAVE", "CRVAL1": 10.0, "CRVAL2": -20.0, "CRVAL3": wave_min,
                 "CRPIX1": size / 2 + 1, "CRPIX2": size / 2 + 1, "CRPIX3": 1.0, "CDELT1": -pix_arcsec / 3600,
                 "CDELT2": pix_arcsec / 3600, "CDELT3": wave_step, "PIXAR_SR": area_sr, "PIXAR_A2": pix_arcsec**2}.items():
        sci.header[k] = v
    fits.HDUList([_primary(instrument, **keys), sci, fits.ImageHDU(np.full_like(data, 1e-3), name="ERR")]).writeto(path, overwrite=True)
    return Path(path)


def synthetic_x1d(path: Path, *, wave: np.ndarray | None = None, flux: np.ndarray | None = None,
                  instrument: str = "MIRI", **keys: Any) -> Path:
    """An extracted spectrum in the jwst x1d table layout."""
    from ..spectra import Spectrum1D, write_x1d

    w = np.linspace(4.9, 5.7, 200) if wave is None else np.asarray(wave, float)
    f = np.ones_like(w) if flux is None else np.asarray(flux, float)
    header = {"INSTRUME": instrument, "TARGPROP": "TEST-TARGET", "TARG_RA": 10.0, "TARG_DEC": -20.0, "SRCNAME": "test"}
    header.update(keys)
    return write_x1d(Path(path), Spectrum1D(w, f, np.full_like(f, 0.01)), header=header)
