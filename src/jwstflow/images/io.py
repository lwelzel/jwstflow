"""Imaging products read into plain arrays: the image, its errors and weights, its sky grid and band.

:func:`read_image` reads a resampled image (``*_i2d.fits``: SCI, ERR, WHT, CON) or a calibrated
exposure (``*_cal.fits``: SCI, ERR, DQ) into an :class:`Image`. :meth:`Image.cutout` cuts a box
around a sky position out of it with a WCS of its own (CRPIX shifted by the box's origin), so a
product written from a cutout lands where it belongs on the sky; the origin in the parent array
travels with it (``origin``, and ``JWFCUTX0`` / ``JWFCUTY0`` in the header it writes).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .filters import Band, band_of

log = logging.getLogger(__name__)

#: Square degrees -> steradian.
_ARCSEC2_PER_SR = (math.degrees(1.0) * 3600.0) ** 2


@dataclass
class Image:
    """One image on its own pixel grid, with what a measurement on it needs."""

    data: np.ndarray                 #: (ny, nx) SCI
    error: np.ndarray | None         #: (ny, nx) ERR
    weight: np.ndarray | None        #: (ny, nx) WHT of a resampled image
    context: np.ndarray | None       #: (ncon, ny, nx) CON: one bit per input exposure
    dq: np.ndarray | None            #: (ny, nx) DQ of a calibrated exposure
    wcs: Any                         #: celestial astropy WCS (None: a detector image)
    header: Any                      #: the SCI header (its WCS keywords shifted for a cutout)
    primary: Any                     #: the primary header
    unit: str | None                 #: BUNIT
    pixar_sr: float | None           #: solid angle of a pixel [sr]
    band: Band | None                #: the filter
    origin: tuple[int, int] = (0, 0)     #: (y, x) of pixel [0, 0] in the array of the product read
    path: Path | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.data.shape)  # type: ignore[return-value]

    @property
    def pixscale_arcsec(self) -> float | None:
        """Side of a pixel [arcsec], from its solid angle, else from the WCS."""
        if self.pixar_sr:
            return math.sqrt(self.pixar_sr * _ARCSEC2_PER_SR)
        if self.wcs is not None:
            from astropy.wcs.utils import proj_plane_pixel_scales

            return float(np.sqrt(np.prod(proj_plane_pixel_scales(self.wcs))) * 3600.0)
        return None

    @property
    def fwhm_pix(self) -> float | None:
        """The diffraction-limited FWHM of the band in pixels of this image (None when unknown)."""
        scale = self.pixscale_arcsec
        return None if self.band is None or not scale else self.band.fwhm_arcsec / scale

    @property
    def n_exposures(self) -> np.ndarray | None:
        """(ny, nx) number of input exposures that contribute to each pixel (from CON)."""
        if self.context is None:
            return None
        con = np.ascontiguousarray(self.context.astype(np.uint32))
        bits = np.unpackbits(con.view(np.uint8).reshape(*con.shape, 4), axis=-1)
        return bits.sum(axis=(0, -1)).astype(np.int16)

    @property
    def label(self) -> str:
        """The band's name, else the FILTER keyword."""
        return self.band.name if self.band is not None else str(self.primary.get("FILTER", "?")).strip()

    def pixel(self, ra: float, dec: float) -> tuple[float, float]:
        """(y, x) of a sky position [deg] on this image (0-based)."""
        if self.wcs is None:
            raise ValueError("image without a celestial WCS")
        x, y = self.wcs.all_world2pix(ra, dec, 0)
        return float(y), float(x)

    def world(self, y: float, x: float) -> tuple[float, float]:
        """(ra, dec) [deg] of a pixel (0-based)."""
        if self.wcs is None:
            raise ValueError("image without a celestial WCS")
        ra, dec = self.wcs.all_pix2world(x, y, 0)
        return float(ra), float(dec)

    def cutout(self, ra: float, dec: float, size_arcsec: float) -> Image:
        """The box of ``size_arcsec`` around a sky position (clipped at the edges of the image), with
        its WCS and header shifted to the box; arrays are copies."""
        scale = self.pixscale_arcsec
        if self.wcs is None or not scale:
            raise ValueError("a cutout needs a celestial WCS and a pixel scale")
        yc, xc = self.pixel(ra, dec)
        half = int(math.ceil(0.5 * size_arcsec / scale))
        ny, nx = self.shape
        y0, y1 = max(0, int(round(yc)) - half), min(ny, int(round(yc)) + half + 1)
        x0, x1 = max(0, int(round(xc)) - half), min(nx, int(round(xc)) + half + 1)
        if y0 >= y1 or x0 >= x1:
            raise ValueError(f"({ra:.6f}, {dec:.6f}) lies outside the image")
        box = (slice(y0, y1), slice(x0, x1))
        header = self.header.copy()
        for key, shift in (("CRPIX1", x0), ("CRPIX2", y0)):
            if key in header:
                header[key] = float(header[key]) - shift
        oy, ox = self.origin[0] + y0, self.origin[1] + x0
        header["JWFCUTX0"] = (ox, "x of pixel 0 in the image cut from")
        header["JWFCUTY0"] = (oy, "y of pixel 0 in the image cut from")
        header["NAXIS1"], header["NAXIS2"] = x1 - x0, y1 - y0

        def cut(a: np.ndarray | None) -> np.ndarray | None:
            return None if a is None else np.array(a[(..., *box)])

        return replace(self, data=cut(self.data), error=cut(self.error), weight=cut(self.weight),
                       context=cut(self.context), dq=cut(self.dq), wcs=self.wcs.slice(box), header=header,
                       origin=(oy, ox))


def read_image(path: Path | str) -> Image | None:
    """The image of an imaging product (None, with a warning, without a 2-D SCI extension)."""
    from astropy.io import fits
    from astropy.wcs import WCS

    path = Path(path)
    with fits.open(path) as hdul:
        if "SCI" not in hdul or hdul["SCI"].data is None or hdul["SCI"].data.ndim != 2:
            log.warning("%s has no 2-D SCI extension; skipped", path.name)
            return None
        header = hdul["SCI"].header.copy()
        primary = hdul[0].header.copy()
        data = np.asarray(hdul["SCI"].data, dtype=float)

        def ext(name: str, ndim: int = 2) -> np.ndarray | None:
            if name not in hdul or hdul[name].data is None or hdul[name].data.ndim != ndim:
                return None
            arr = np.asarray(hdul[name].data)
            return arr if arr.shape[-2:] == data.shape else None

        error = ext("ERR")
        weight = ext("WHT")
        context = ext("CON", 3)
        dq = ext("DQ")
    wcs = None
    try:
        wcs = WCS(header).celestial
        if not wcs.has_celestial:
            wcs = None
    except Exception as exc:  # a header without a usable WCS: a detector image
        log.info("%s: no celestial WCS in the SCI header (%s)", path.name, exc)
    pixar_sr = header.get("PIXAR_SR")
    if pixar_sr is None and header.get("PIXAR_A2") is not None:
        pixar_sr = float(header["PIXAR_A2"]) / _ARCSEC2_PER_SR
    band = band_of(primary.get("INSTRUME"), primary.get("FILTER"), primary.get("PUPIL"))
    return Image(data=data, error=None if error is None else error.astype(float),
                 weight=None if weight is None else weight.astype(float), context=context, dq=dq, wcs=wcs,
                 header=header, primary=primary, unit=header.get("BUNIT"),
                 pixar_sr=None if pixar_sr is None else float(pixar_sr), band=band, path=path)
