"""PSF engines: stpsf's IFU mode (science grade) and an analytic Gaussian (offline).

An engine is anything with a ``name`` and
``compute(config, wavelengths_um, grid, *, date_obs=None) -> (nw, npix, npix) array``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from .grid import PsfGrid
from .instrument import _FWHM_PER_SIGMA, IfuConfig, approx_fwhm_arcsec

log = logging.getLogger(__name__)


class GaussianPsfEngine:
    """Analytic Gaussian PSF slices at the :func:`approx_fwhm_arcsec` width.

    An offline approximation: no stpsf, no data files, instant. Useful for
    pipeline dry-runs and tests, and as a sanity reference next to the stpsf
    product -- but it carries no diffraction structure (spikes, Airy rings)
    and no MRS anisotropy, so science-grade convolution should use
    :class:`StpsfEngine`. Slices integrate to ~1 (minus grid truncation).
    """

    name = "gaussian"

    def compute(self, config: IfuConfig, wavelengths_um: np.ndarray, grid: PsfGrid, *,
                date_obs: str | None = None) -> np.ndarray:
        n = grid.npix
        centre = (n - 1) / 2.0
        yy, xx = np.indices((n, n))
        r2 = (yy - centre) ** 2 + (xx - centre) ** 2
        out = np.empty((len(wavelengths_um), n, n), dtype=float)
        for i, lam in enumerate(np.asarray(wavelengths_um, dtype=float)):
            sigma_pix = float(approx_fwhm_arcsec(config, lam)) / _FWHM_PER_SIGMA / grid.pixelscale_arcsec
            out[i] = np.exp(-r2 / (2 * sigma_pix**2)) / (2 * np.pi * sigma_pix**2)
        return out


@dataclass
class StpsfEngine:
    """PSF slices from stpsf's IFU mode (NIRSpec IFU / MIRI MRS).

    Needs the ``jwstflow[psf]`` extra and the stpsf data files (set
    ``$STPSF_PATH``, or let stpsf download them on first use). Slices are the
    broadened ``DET_DIST`` output of ``calc_psf`` -- for MIRI MRS the
    empirical model of Argyriou/Law et al. 2023 (Gaussian along alpha, a
    slice-width box along beta), for NIRSpec a 0.05"-sigma Gaussian -- so they
    describe the PSF *as realised in reconstructed cubes*, not the bare
    optical PSF. The instrument's native pixel scale is overridden with the
    requested grid after band selection (stpsf resets it on any IFU aperture
    change), and NIRSpec's extra IFU-align output rotation is disabled so
    both instruments come out with array +y along the aperture ideal +y axis
    (the ``PA_APER`` convention that :func:`rotate_cube_to_sky` assumes).

    ``method='exact'`` runs one full ``calc_psf`` per wavelength;
    ``method='fast'`` uses ``calc_datacube_fast`` (one pupil propagation for
    all wavelengths, ~100x faster) and applies stpsf's own broadening per
    slice afterwards -- the trade-offs are stpsf's (assumes a
    wavelength-independent exit-pupil wavefront). ``opd`` selects the
    wavefront map: ``'default'`` (stpsf's stock OPD, offline), ``'by_date'``
    (the measured in-flight OPD nearest the observation -- queries MAST) or a
    local OPD file path.
    """

    opd: str = "default"
    broadening: str = "default"
    method: str = "exact"
    name: str = field(init=False, default="stpsf")

    #: MRS across-slice widths [arcsec] per channel (Argyriou et al. 2023), the
    #: fallback when stpsf's private attribute moves.
    MRS_SLICE_WIDTH: ClassVar[dict[str, float]] = {"1": 0.177, "2": 0.280, "3": 0.390, "4": 0.656}

    def compute(self, config: IfuConfig, wavelengths_um: np.ndarray, grid: PsfGrid, *,
                date_obs: str | None = None) -> np.ndarray:
        stpsf = _import_stpsf()
        inst = self._instrument(stpsf, config, grid, date_obs)
        waves = np.asarray(wavelengths_um, dtype=float)
        if self.method == "fast":
            stack = self._fast(stpsf, inst, config, waves, grid)
        else:
            stack = self._exact(inst, waves, grid)
        return stack

    # -- configuration ------------------------------------------------------
    def _instrument(self, stpsf: Any, config: IfuConfig, grid: PsfGrid, date_obs: str | None) -> Any:
        if config.instrument == "MIRI":
            inst = stpsf.MIRI()
            inst.mode = "IFU"
            inst.band = config.mrs_band
        else:
            inst = stpsf.NIRSpec()
            inst.mode = "IFU"
            # keep array +y = ideal +y (stpsf's default adds 90 deg for the IFU-align cube convention)
            inst.options["ifualign_rotation"] = False
            if config.grating:
                try:
                    inst.disperser = config.grating
                except Exception as exc:
                    log.warning("stpsf rejected disperser %r (%s); PSF is monochromatic per slice anyway", config.grating, exc)
            if config.filter:
                try:
                    inst.filter = config.filter
                except Exception as exc:
                    log.warning("stpsf rejected filter %r (%s); continuing with its default", config.filter, exc)
        if self.broadening != "default":
            inst.options["ifu_broadening"] = None if self.broadening.lower() == "none" else self.broadening
        if self.opd == "by_date":
            if date_obs is None:
                log.warning("opd='by_date' but the cube has no DATE-OBS; using the default OPD")
            else:
                inst.load_wss_opd_by_date(date_obs)
        elif self.opd not in ("default", ""):
            inst.pupilopd = self.opd
        # last: IFU band/aperture selection always resets the pixel scale to the spaxel size
        inst.pixelscale = grid.pixelscale_arcsec
        return inst

    # -- computation --------------------------------------------------------
    def _exact(self, inst: Any, waves_um: np.ndarray, grid: PsfGrid) -> np.ndarray:
        out = np.empty((len(waves_um), grid.npix, grid.npix), dtype=float)
        for i, lam in enumerate(waves_um):
            hdul = inst.calc_psf(monochromatic=lam * 1e-6, fov_pixels=grid.npix, oversample=1,
                                 add_distortion=True)
            out[i] = _distorted_slice(hdul)
            log.debug("stpsf %s: %.4f um (%d/%d)", getattr(inst, "name", "?"), lam, i + 1, len(waves_um))
        return out

    def _fast(self, stpsf: Any, inst: Any, config: IfuConfig, waves_um: np.ndarray, grid: PsfGrid) -> np.ndarray:
        import astropy.units as u
        from astropy.io import fits

        cube = inst.calc_datacube_fast(waves_um * 1e-6 * u.m, fov_pixels=grid.npix, oversample=1,
                                       add_distortion=False)
        stack = np.asarray(cube[0].data, dtype=float)
        if (self.broadening or "default").lower() == "none":
            return stack
        # calc_datacube_fast returns the unbroadened oversampled cube only; apply
        # stpsf's own per-slice broadening (it reads PIXELSCL/WAVELEN, so any grid works)
        from stpsf import detectors

        slice_width = getattr(inst, "_ifu_slice_width", None)
        if config.instrument == "MIRI" and slice_width is None:
            slice_width = self.MRS_SLICE_WIDTH[config.mrs_band[0]]
        for i, lam in enumerate(waves_um):
            hdr = fits.Header({"PIXELSCL": grid.pixelscale_arcsec, "WAVELEN": lam * 1e-6, "EXTNAME": "OVERDIST"})
            hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(stack[i], header=hdr)])
            if config.instrument == "MIRI":
                detectors.apply_miri_ifu_broadening(hdul, inst.options, slice_width=slice_width)
            else:
                detectors.apply_nirspec_ifu_broadening(hdul, inst.options)
            stack[i] = hdul[1].data
        return stack


def _distorted_slice(hdul: Any) -> np.ndarray:
    """The broadened, detector-sampled plane of a calc_psf result (DET_DIST, with fallbacks)."""
    for name in ("DET_DIST", "OVERDIST", "DET_SAMP"):
        try:
            return np.asarray(hdul[name].data, dtype=float)
        except KeyError:
            continue
    return np.asarray(hdul[-1].data, dtype=float)


def _import_stpsf() -> Any:
    try:
        import stpsf
    except ImportError as exc:
        raise RuntimeError(
            "PSF generation needs stpsf: install the extra with `pip install 'jwstflow[psf]'` "
            "(plus the stpsf data files -- set $STPSF_PATH or let stpsf download them; "
            "see docs/psf_cubes.md). Offline alternative: engine: gaussian."
        ) from exc
    return stpsf


def stpsf_version() -> str | None:
    try:
        import stpsf

        return str(stpsf.__version__)
    except ImportError:
        return None
