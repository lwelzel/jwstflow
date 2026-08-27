"""Small helpers for 1-D spectra in the jwst ``x1d`` format and IFU cubes.

They are the common currency of custom extraction / stitching / cleaning steps:
every step reads and writes the pipeline's own ``EXTRACT1D`` table layout so
official tools (``combine_1d``, jdaviz, ``load_asn`` products) keep working.
``stdatamodels`` is imported lazily so the module is importable without jwst.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

X1D_COLUMNS: tuple[str, ...] = (
    "WAVELENGTH", "FLUX", "FLUX_ERROR", "FLUX_VAR_POISSON", "FLUX_VAR_RNOISE", "FLUX_VAR_FLAT",
    "SURF_BRIGHT", "SB_ERROR", "SB_VAR_POISSON", "SB_VAR_RNOISE", "SB_VAR_FLAT", "DQ",
    "BACKGROUND", "BKGD_ERROR", "BKGD_VAR_POISSON", "BKGD_VAR_RNOISE", "BKGD_VAR_FLAT", "NPIXELS",
)
X1D_DTYPE = [(n, "u4" if n == "DQ" else "f8") for n in X1D_COLUMNS]


@dataclass
class Spectrum1D:
    """A 1-D spectrum: wavelength [um], flux [Jy], error [Jy] plus free-form metadata."""

    wavelength: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    dq: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.wavelength = np.asarray(self.wavelength, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        self.error = np.asarray(self.error, dtype=float)
        if self.dq is None:
            self.dq = np.zeros(len(self.wavelength), dtype="u4")

    @property
    def good(self) -> np.ndarray:
        return np.isfinite(self.flux) & (self.dq == 0)

    def sorted(self) -> Spectrum1D:
        order = np.argsort(self.wavelength)
        return Spectrum1D(self.wavelength[order], self.flux[order], self.error[order], self.dq[order], dict(self.meta))


def x1d_table(wave: np.ndarray, **columns: Any) -> np.ndarray:
    """Structured array in the jwst EXTRACT1D layout; unspecified columns are zero."""
    tab = np.zeros(len(wave), dtype=X1D_DTYPE)
    tab["WAVELENGTH"] = wave
    for name, values in columns.items():
        tab[name.upper()] = values
    return tab


def write_x1d(path: Path, spectrum: Spectrum1D, *, like: Any = None, surf_bright: np.ndarray | None = None,
              header: dict[str, Any] | None = None) -> Path:
    """Write ``spectrum`` as a ``MultiSpecModel`` (one EXTRACT1D extension).

    ``like`` is a datamodel (or path) whose primary metadata is copied so
    instrument keywords travel along; ``header`` adds/overrides primary keywords.
    """
    from stdatamodels.jwst import datamodels

    cols = {"flux": spectrum.flux, "flux_error": spectrum.error, "dq": spectrum.dq}
    if surf_bright is not None:
        cols["surf_bright"] = surf_bright
    spec = datamodels.SpecModel()
    spec.spec_table = x1d_table(spectrum.wavelength, **cols)
    multi = datamodels.MultiSpecModel()
    multi.spec.append(spec)
    if like is not None:
        with datamodels.open(like) if not hasattr(like, "meta") else _nullcontext(like) as src:
            multi.update(src, only="PRIMARY")
    multi.meta.filename = Path(path).name
    multi.save(str(path))
    if header:
        from astropy.io import fits

        with fits.open(path, mode="update") as hdul:
            for key, value in header.items():
                hdul[0].header[key] = value
    return Path(path)


def read_x1d(path: Path, index: int = 0) -> Spectrum1D:
    """Read one spectrum of an x1d file (plain astropy, no jwst needed)."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        hdr = dict(hdul[0].header)
        exts = [h for h in hdul if h.name == "EXTRACT1D"]
        if not exts:
            raise ValueError(f"{path}: no EXTRACT1D extension")
        tab = exts[index].data
        cols = {c.upper() for c in tab.columns.names}
        wave = np.asarray(tab["WAVELENGTH"], dtype=float)
        flux = np.asarray(tab["FLUX"], dtype=float)
        err = np.asarray(tab["FLUX_ERROR"], dtype=float) if "FLUX_ERROR" in cols else np.full_like(flux, np.nan)
        dq = np.asarray(tab["DQ"], dtype="u4") if "DQ" in cols else np.zeros(len(wave), "u4")
    meta = {k: v for k, v in hdr.items() if isinstance(v, (str, int, float, bool))}
    meta["path"] = str(path)
    return Spectrum1D(wave, flux, err, dq, meta)


def cube_wavelengths(cube: Any) -> np.ndarray:
    """Wavelength [um] of every plane of an ``IFUCubeModel`` (linear WCS or wavetable)."""
    wavetable = getattr(cube, "wavetable", None)
    if wavetable is not None and len(wavetable) > 0:
        return np.asarray(wavetable["wavelength"], dtype=float).ravel()
    n = cube.data.shape[0]
    w = cube.meta.wcsinfo
    return w.crval3 + (np.arange(n) + 1 - w.crpix3) * w.cdelt3


def mrs_psf_fwhm_arcsec(wavelength_um: np.ndarray | float) -> np.ndarray | float:
    """MIRI MRS PSF FWHM (arcsec) versus wavelength, Law et al. 2023: 0.033 lambda + 0.106."""
    return 0.033 * np.asarray(wavelength_um, dtype=float) + 0.106


class _nullcontext:
    def __init__(self, obj: Any):
        self.obj = obj

    def __enter__(self) -> Any:
        return self.obj

    def __exit__(self, *exc: Any) -> None:
        return None
