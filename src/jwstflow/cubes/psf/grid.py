"""The grids PSF slices live on: the spatial grid and the wavelength sub-sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The ecosystem's model-rendering convention: slices of MODEL_NPIX x MODEL_NPIX
#: pixels spanning +-MODEL_FOV_AU/2 at the target (rendered at 1 pc, where
#: 1 AU subtends exactly 1 arcsec, then scaled by 1/distance).
MODEL_FOV_AU = 600.0
MODEL_NPIX = 256


def model_pixel_scale(distance_pc: float, *, fov_au: float = MODEL_FOV_AU, npix: int = MODEL_NPIX) -> float:
    """Pixel scale [arcsec] of the model grid for a target at ``distance_pc``.

    A modeling-side convenience (the PSF library itself is
    distance-independent): the convention renders ``npix`` pixels over
    ``fov_au``; at 1 pc, 1 AU subtends 1 arcsec (the definition of the
    parsec), so the angular pixel scale is ``(fov_au / npix) / distance_pc``
    -- the natural target for :meth:`PsfCubeProduct.resampled`.
    """
    if distance_pc <= 0:
        raise ValueError(f"distance_pc must be positive, not {distance_pc}")
    return (float(fov_au) / int(npix)) / float(distance_pc)


@dataclass(frozen=True)
class PsfGrid:
    """The spatial grid PSF slices are computed on: ``npix`` x ``npix`` pixels of
    ``pixelscale_arcsec``, centred on the geometric array centre ((npix-1)/2)."""

    pixelscale_arcsec: float
    npix: int

    @property
    def fov_arcsec(self) -> float:
        return self.pixelscale_arcsec * self.npix

    @classmethod
    def for_model(cls, distance_pc: float, *, fov_au: float = MODEL_FOV_AU, npix: int = MODEL_NPIX) -> PsfGrid:
        return cls(model_pixel_scale(distance_pc, fov_au=fov_au, npix=npix), int(npix))


def subsample_wavelengths(wave_min: float, wave_max: float, *, max_fractional_step: float = 0.02,
                          n: int | None = None, min_n: int = 5) -> np.ndarray:
    """Log-spaced PSF wavelengths [um] spanning ``[wave_min, wave_max]``, endpoints included.

    Without ``n``, the count is chosen so adjacent wavelengths differ by at
    most ``max_fractional_step`` (the PSF FWHM grows ~linearly with lambda, so
    this bounds the FWHM mismatch of the *nearest* slice to half the step and
    of linear interpolation between slices to O(step^2/8)); never fewer than
    ``min_n``. Interpolation by :meth:`PsfCubeProduct.at` never extrapolates
    because the endpoints are exact.
    """
    lo, hi = float(wave_min), float(wave_max)
    if not (0 < lo <= hi):
        raise ValueError(f"invalid wavelength range [{wave_min}, {wave_max}] um")
    if hi == lo:
        return np.array([lo])
    if n is None:
        if not 0 < max_fractional_step:
            raise ValueError("max_fractional_step must be positive")
        n = int(np.ceil(np.log(hi / lo) / np.log1p(max_fractional_step))) + 1
        n = max(n, min_n)
    if n < 2:
        raise ValueError("n must be at least 2 for a wavelength range")
    return np.geomspace(lo, hi, int(n))
