"""The record of a neighbour's PSF subtraction: ``*_nbrfit.fits``.

``subtract_psf_neighbour`` writes, next to the cleaned cube, one product per
cube that says what was taken out and how well it went -- everything
``qa_psf_neighbour`` draws, so the figure needs neither cube::

    <base>_nbrfit.fits             derived from the s3d cube
      PRIMARY                      header copied from the cube + JWFNB* keywords: the positions
                                   (given, fitted; target), the PSF library, the parameters,
                                   and the medians of the diagnostics
      PLANES     bintable          one row per plane (columns in PLANE_COLUMNS)
      IMAGES     f4 (3, nb, ny, nx) data, model and residual (data - neighbour - background), each the
                                   median over a block of planes; 2-D WCS of the cube in the header
      BLOCKS     bintable          the blocks: FIRST, LAST plane, WAVELENGTH [um] (middle)
      REGIONS    u1 (ny, nx)       bit 1 fitted, 2 the target's protected region, 4 the control ring,
                                   8 beyond the PSF library's field (not fitted, not subtracted),
                                   16 the neighbour's core, blanked in the cleaned cube
      PROFILE    bintable          azimuthal means about the neighbour, outside the protected region:
                                   R_ARCSEC and, per block b, BEFORE_b (data - background), MODEL_b,
                                   AFTER_b (residual) [cube units]
      POSITIONS  bintable          the position fit per block: WAVELENGTH, X, Y (fitted), XC, YC
                                   (centroid), CHI2

Pixel positions are 0-based array coordinates of the cube. Surface
brightness columns are in the cube's unit (``BUNIT`` in PRIMARY, MJy/sr for
jwst cubes); fluxes in Jy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Product suffix (``*_nbrfit.fits``).
NBRFIT_SUFFIX = "nbrfit"

#: REGIONS bits.
FITTED, PROTECTED, RING, BEYOND_LIBRARY, BLANKED = 1, 2, 4, 8, 16

#: PLANES columns: name -> (unit, meaning).
PLANE_COLUMNS = {
    "WAVELENGTH": ("um", "wavelength of the plane"),
    "FLUX": ("Jy", "the neighbour's total flux: all model components, all of the light"),
    "FLUX_ERR": ("Jy", "its 1-sigma error (scaled by sqrt(reduced chi2) where > 1)"),
    "FLUX_PSF": ("Jy", "the PSF component alone"),
    "FLUX_WING": ("Jy", "the light of the wing profile (within the field), when fitted; included in FLUX"),
    "FLUX_HALO1": ("Jy", "the first broadened PSF copy (halo_sigma), when fitted; FLUX_HALO2 ... likewise"),
    "SUBTRACTED": ("Jy", "the flux removed from the cube's grid (the model as subtracted)"),
    "CONST": ("cube", "background constant (fitted, not subtracted)"),
    "GRADX": ("cube/pix", "background gradient along x (fitted, not subtracted)"),
    "GRADY": ("cube/pix", "background gradient along y"),
    "CHI2R": ("", "reduced chi2 of the fit"),
    "NFIT": ("", "spaxels fitted"),
    "NCLIP": ("", "spaxels clipped"),
    "FITRMS": ("cube", "rms of the residual over the fitted spaxels"),
    "RING": ("cube", "mean residual in the control ring"),
    "RING_ERR": ("cube", "its formal error (sqrt(sum ERR^2) / n)"),
    "RING_SIDE": ("cube", "mean residual in the half of the ring facing the target"),
    "RING_BACK": ("cube", "mean residual in the opposite half"),
    "RING_BEFORE": ("cube", "mean of (data - background) in the ring: the neighbour's light there before"),
    "SKY_FAR": ("cube", "mean residual beyond the ring (farther from the neighbour), off the protected region"),
    "TARGET": ("cube", "mean of (data - neighbour - background) within one FWHM of the target"),
    "FILLED": ("", "1 where the fit failed and the amplitude was interpolated from other planes"),
}


@dataclass
class PsfSubtraction:
    """A ``*_nbrfit.fits`` product in memory."""

    planes: dict[str, np.ndarray]              #: the PLANES columns
    images: np.ndarray                         #: (3, nb, ny, nx): data, model, residual per block
    blocks: dict[str, np.ndarray]              #: FIRST, LAST, WAVELENGTH
    regions: np.ndarray                        #: (ny, nx) u1 bits
    profile: dict[str, np.ndarray]             #: R_ARCSEC, BEFORE_b, MODEL_b, AFTER_b
    positions: dict[str, np.ndarray]           #: WAVELENGTH, X, Y, XC, YC, CHI2
    keys: dict[str, Any] = field(default_factory=dict)   #: JWFNB* keywords
    header: Any = None                         #: PRIMARY header after read()
    image_header: Any = None                   #: 2-D WCS header of IMAGES after read()

    # -- convenience -----------------------------------------------------------
    def key(self, name: str, default: Any = None) -> Any:
        """A PRIMARY keyword (after :meth:`read`) or an entry of :attr:`keys`."""
        if self.header is not None and name in self.header:
            return self.header[name]
        value = self.keys.get(name, default)
        return value[0] if isinstance(value, tuple) else value

    def mask(self, bit: int) -> np.ndarray:
        return (np.asarray(self.regions) & bit) > 0

    # -- I/O -------------------------------------------------------------------
    def write(self, path: Path | str, cube_path: Path | str) -> Path:
        from astropy.io import fits
        from astropy.table import Table

        from ..wcs import strip_wave_axis

        with fits.open(cube_path) as hdul:
            primary = fits.PrimaryHDU(header=hdul[0].header.copy())
            sci = hdul["SCI"].header.copy() if "SCI" in hdul else fits.Header()
        for k, v in self.keys.items():
            primary.header[k] = v
        primary.header["BUNIT"] = (str(sci.get("BUNIT", "MJy/sr")), "unit of the surface brightness columns")
        hdr2d = strip_wave_axis(sci)
        images = fits.ImageHDU(np.asarray(self.images, dtype="f4"), header=hdr2d.copy(), name="IMAGES")
        images.header["PLANE1"] = ("data", "IMAGES[0]: block medians of the cube")
        images.header["PLANE2"] = ("model", "IMAGES[1]: the subtracted neighbour")
        images.header["PLANE3"] = ("residual", "IMAGES[2]: data - neighbour - background")
        planes = Table({k: np.asarray(v) for k, v in self.planes.items()})
        for name, (unit, _) in PLANE_COLUMNS.items():
            if name in planes.colnames and unit and "cube" not in unit:
                planes[name].unit = unit
        hdus = [primary,
                fits.BinTableHDU(planes, name="PLANES"),
                images,
                fits.BinTableHDU(Table({k: np.asarray(v) for k, v in self.blocks.items()}), name="BLOCKS"),
                fits.ImageHDU(np.asarray(self.regions, dtype="u1"), header=hdr2d.copy(), name="REGIONS"),
                fits.BinTableHDU(Table({k: np.asarray(v) for k, v in self.profile.items()}), name="PROFILE"),
                fits.BinTableHDU(Table({k: np.asarray(v) for k, v in self.positions.items()}), name="POSITIONS")]
        fits.HDUList(hdus).writeto(path, overwrite=True)
        return Path(path)

    @classmethod
    def read(cls, path: Path | str) -> PsfSubtraction:
        from astropy.io import fits

        def table(hdu: Any) -> dict[str, np.ndarray]:
            return {name: np.asarray(hdu.data[name]) for name in hdu.columns.names}

        with fits.open(path) as hdul:
            return cls(planes=table(hdul["PLANES"]), images=np.asarray(hdul["IMAGES"].data, dtype=float),
                       blocks=table(hdul["BLOCKS"]), regions=np.asarray(hdul["REGIONS"].data),
                       profile=table(hdul["PROFILE"]), positions=table(hdul["POSITIONS"]),
                       header=hdul[0].header.copy(), image_header=hdul["IMAGES"].header.copy())


__all__ = ["BEYOND_LIBRARY", "BLANKED", "FITTED", "NBRFIT_SUFFIX", "PLANE_COLUMNS", "PROTECTED", "RING", "PsfSubtraction"]
