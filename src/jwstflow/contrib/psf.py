"""PSF-cube steps: per-observation instrument PSFs for IFU model comparison.

``psf_cube`` generates one ``*_psfcube.fits`` per input ``*_s3d.fits`` (the
psfcube-product contract, :mod:`jwstflow.psf`); ``qa_psf_cube`` plots it.
The stpsf engine needs the ``jwstflow[psf]`` extra and the stpsf data files;
everything else in the product path (reading, interpolation, convolution, QA)
stays dependency-light.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from .. import qafig
from ..psf import (
    MODEL_FOV_AU,
    MODEL_NPIX,
    PSFCUBE_SUFFIX,
    GaussianPsfEngine,
    PsfCubeProduct,
    PsfGrid,
    StpsfEngine,
    approx_fwhm_arcsec,
    diffraction_fwhm_arcsec,
    read_cube_geometry,
    rotate_cube_to_sky,
    stpsf_version,
    subsample_wavelengths,
)
from ..steps.base import RunContext, Step, StepParams

log = logging.getLogger(__name__)


class PsfCube(Step):
    """Spectrally sub-sampled PSF cube matching an IFU cube, for model convolution.

    For each ``*_s3d.fits`` cube the instrument configuration (NIRSpec
    grating/filter or MRS channel/band) is read from the header, the PSF is
    computed with stpsf's IFU mode -- including its empirical broadening, so
    the slices describe the PSF as realised in reconstructed cubes -- at
    log-spaced wavelengths spanning the cube (adjacent wavelengths within
    ``max_fractional_step``, default 2%), rotated to the sky frame of
    skyalign cubes (north up, east left) using the header position angle, and
    written as a ``*_psfcube.fits`` product (contract: ``jwstflow.psf``).

    The spatial grid defaults to the ecosystem's model-rendering convention:
    ``npix`` (256) pixels spanning ``fov_au`` (600 AU, i.e. +-300 AU) at the
    target, so ``distance_pc`` is required and the pixel scale becomes
    ``(fov_au/npix)/distance`` arcsec -- model slices then convolve without
    resampling (``PsfCubeProduct.read(...).convolve(model, waves)``).
    ``grid: native`` uses the cube's own spaxel scale instead (e.g. for PSF
    photometry on the cube itself), and an explicit ``pixelscale_arcsec``
    overrides both.

    Engines: ``stpsf`` (science grade; needs ``pip install 'jwstflow[psf]'``
    and the stpsf data files via ``$STPSF_PATH``) or ``gaussian`` (offline
    analytic approximation at the empirical FWHM -- no diffraction structure;
    for dry-runs and tests). ``method: fast`` reuses one pupil propagation
    for all wavelengths (stpsf's ``calc_datacube_fast``, ~100x faster,
    approximate); ``opd: by_date`` fetches the measured in-flight wavefront
    nearest the observation from MAST instead of the stock map.
    """

    inputs = ("*_s3d.fits",)
    outputs = (PSFCUBE_SUFFIX,)
    version = "1"

    class Params(StepParams):
        distance_pc: float | None = Field(None, gt=0, description="target distance [pc]; sets the model-grid pixel "
                                          "scale (fov_au/npix)/distance (required for grid: model)")
        grid: Literal["model", "native"] = Field("model", description="spatial grid: the +-300 AU model convention, "
                                                 "or the cube's own spaxel scale")
        fov_au: float = Field(MODEL_FOV_AU, gt=0, description="model-grid field of view [AU] (grid: model)")
        npix: int = Field(MODEL_NPIX, ge=16, le=2048, description="spatial size of the PSF slices [pixels]")
        pixelscale_arcsec: float | None = Field(None, gt=0, description="explicit pixel scale [arcsec]; overrides grid")
        max_fractional_step: float = Field(0.02, gt=0, le=0.5, description="wavelength sub-sampling: adjacent PSF "
                                           "wavelengths differ by at most this fraction")
        n_wavelengths: int | None = Field(None, ge=2, description="fixed number of PSF wavelengths instead of "
                                          "max_fractional_step")
        engine: Literal["stpsf", "gaussian"] = Field("stpsf", description="PSF engine: stpsf IFU mode, or the "
                                                     "offline Gaussian approximation")
        method: Literal["exact", "fast"] = Field("exact", description="stpsf only: full calc_psf per wavelength, or "
                                                 "one shared pupil propagation (calc_datacube_fast)")
        opd: str = Field("default", description="stpsf wavefront map: 'default' (offline), 'by_date' (measured, "
                         "queries MAST) or an OPD file path")
        broadening: str = Field("default", description="stpsf ifu_broadening: default (empirical MRS / Gaussian "
                                "NIRSpec), 'gaussian', or 'none' (bare optical PSF)")
        rotate_to_sky: bool = Field(True, description="rotate slices to north-up/east-left (skyalign cubes)")
        position_angle_deg: float | None = Field(None, description="aperture +y position angle [deg E of N]; "
                                                 "overrides the header (PA_APER)")

        @model_validator(mode="after")
        def _needs_a_scale(self) -> PsfCube.Params:
            if self.grid == "model" and self.pixelscale_arcsec is None and self.distance_pc is None:
                raise ValueError("grid 'model' needs distance_pc (pixel scale = (fov_au/npix)/distance arcsec); "
                                 "alternatively give pixelscale_arcsec or grid: native")
            return self

    def run(self, inputs: list[Path], ctx: RunContext, *, distance_pc: float | None = None,
            grid: str = "model", fov_au: float = MODEL_FOV_AU, npix: int = MODEL_NPIX,
            pixelscale_arcsec: float | None = None, max_fractional_step: float = 0.02,
            n_wavelengths: int | None = None, engine: str = "stpsf", method: str = "exact",
            opd: str = "default", broadening: str = "default", rotate_to_sky: bool = True,
            position_angle_deg: float | None = None, **params: Any) -> list[Path]:
        (cube_path,) = inputs
        geom = read_cube_geometry(cube_path)
        if pixelscale_arcsec is not None:
            psf_grid = PsfGrid(float(pixelscale_arcsec), int(npix))
        elif grid == "native":
            if geom.pixelscale_arcsec is None:
                raise ValueError(f"{cube_path.name} carries no spatial WCS scale; give pixelscale_arcsec")
            psf_grid = PsfGrid(geom.pixelscale_arcsec, int(npix))
        else:
            psf_grid = PsfGrid.for_model(float(distance_pc), fov_au=fov_au, npix=int(npix))
        waves = subsample_wavelengths(float(geom.wavelengths.min()), float(geom.wavelengths.max()),
                                      max_fractional_step=max_fractional_step, n=n_wavelengths)
        eng = GaussianPsfEngine() if engine == "gaussian" else StpsfEngine(opd=opd, broadening=broadening,
                                                                          method=method)
        ctx.log.info("%s: %s %s, %d wavelengths %.3f-%.3f um, %d px of %.4f\" (%.2f\" field), engine %s",
                     cube_path.name, geom.config.instrument, geom.config.label, len(waves),
                     waves[0], waves[-1], psf_grid.npix, psf_grid.pixelscale_arcsec, psf_grid.fov_arcsec, eng.name)
        stack = eng.compute(geom.config, waves, psf_grid, date_obs=geom.date_obs)

        pa = position_angle_deg if position_angle_deg is not None else geom.position_angle_deg
        rotated = rotate_to_sky and pa is not None
        if rotated:
            stack = rotate_cube_to_sky(stack, float(pa))
        elif rotate_to_sky:
            ctx.log.warning("%s: no PA_APER/ROLL_REF+V3I_YANG in the header and no position_angle_deg given; "
                            "slices stay in the instrument frame (matters for the anisotropic MRS broadening)",
                            cube_path.name)

        meta: dict[str, Any] = {
            "JWFPSFVE": (self.version, "psf_cube step version"),
            "JWFPSFEN": (eng.name, "PSF engine"),
            "JWFPSFST": (stpsf_version() or "none", "stpsf version"),
            "JWFPSFMD": (method if eng.name == "stpsf" else "analytic", "computation method"),
            "JWFPSFOP": (opd if eng.name == "stpsf" else "none", "OPD map selection"),
            "JWFPSFBR": (broadening, "IFU broadening model"),
            "JWFPSFNW": (len(waves), "number of PSF wavelengths"),
            "JWFPSFFS": (max_fractional_step, "max fractional wavelength step"),
            "JWFPSFGR": (grid if pixelscale_arcsec is None else "explicit", "grid choice"),
            "JWFPSFNP": (psf_grid.npix, "spatial size [pix]"),
            "JWFPSFPS": (psf_grid.pixelscale_arcsec, "[arcsec/pix] PSF sampling"),
            "JWFPSFPA": (float(pa) if rotated else "none", "[deg E of N] aperture +y PA applied"),
        }
        if grid == "model" and pixelscale_arcsec is None:
            meta["JWFPSFDS"] = (float(distance_pc), "[pc] target distance of the model grid")
            meta["JWFPSFFA"] = (float(fov_au), "[AU] model-grid field of view")
        product = PsfCubeProduct(stack, waves, psf_grid.pixelscale_arcsec, meta=meta)
        out = ctx.derived_path(cube_path, PSFCUBE_SUFFIX)
        product.write(out, like=cube_path)
        sums = product.sums()
        ctx.log.info("%s -> %s (in-field energy %.3f-%.3f)", cube_path.name, out.name,
                     float(sums.min()), float(sums.max()))
        return [out]


class QaPsfCube(Step):
    """Figure of a PSF-cube product: slices across wavelength, FWHM and in-field energy.

    For each ``*_psfcube.fits``, ``n_slices`` PSF slices spread over the
    stored wavelengths are imaged (log stretch, axes in arcsec offsets), next
    to the measured FWHM against wavelength -- with the diffraction limit
    1.025 lambda/D and, for MIRI MRS, the Law et al. 2023 empirical relation
    as references -- and the per-slice in-field energy fraction. Everything is
    read from the product itself (the psfcube contract is self-describing);
    one PNG per input, named after it, following the jwstflow QA figure
    standard (``docs/qa_figures.md``).
    """

    level = "qa"
    inputs = (f"*_{PSFCUBE_SUFFIX}.fits",)
    version = "1"

    def run(self, inputs: list[Path], ctx: RunContext, *, n_slices: int = 4, dpi: int = 150,
            fmt: str = "png", xscale: str = "log", **params: Any) -> list[Path]:
        from ..psf import IfuConfig

        out: list[Path] = []
        for inp in inputs:
            product = PsfCubeProduct.read(inp)
            try:
                config = IfuConfig.from_header(product.header)
            except ValueError:
                config = None
            n = max(2, min(int(n_slices), len(product.wavelengths)))
            picks = np.unique(np.linspace(0, len(product.wavelengths) - 1, n).round().astype(int))
            plt = qafig.use_agg()
            fig = plt.figure(figsize=(3.0 * len(picks), 5.6), layout="constrained")
            gs = fig.add_gridspec(2, len(picks))
            half = product.pixelscale_arcsec * product.data.shape[1] / 2.0
            for col, k in enumerate(picks):
                ax = fig.add_subplot(gs[0, col])
                qafig.imshow(ax, product.data[k], stretch="log", cbar=(col == len(picks) - 1),
                             extent=(-half, half, -half, half))
                ax.text(0.04, 0.94, f"{product.wavelengths[k]:.2f} um", transform=ax.transAxes,
                        color="white", fontsize=8, va="top",
                        bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.5, "edgecolor": "none"})
                ax.set_xlabel("x offset [arcsec]")
                if col == 0:
                    ax.set_ylabel("y offset [arcsec]")
                else:
                    ax.set_yticklabels([])
            ax_fwhm = fig.add_subplot(gs[1, : max(1, len(picks) // 2)])
            ax_sum = fig.add_subplot(gs[1, max(1, len(picks) // 2):])
            wave, fwhm = product.wavelengths, product.fwhm_arcsec()
            qafig.step(ax_fwhm, wave, fwhm, color=qafig.MAIN_COLOR, label="measured")
            fine = np.geomspace(wave[0], wave[-1], 200)
            ax_fwhm.plot(fine, diffraction_fwhm_arcsec(fine), color="gray", lw=0.8, ls="--",
                         label="diffraction 1.025$\\lambda$/D")
            if config is not None and config.instrument == "MIRI":
                ax_fwhm.plot(fine, approx_fwhm_arcsec(config, fine), color="gray", lw=0.8, ls=":",
                             label="Law+2023 empirical")
            ax_fwhm.set_xlabel(qafig.WAVE_LABEL)
            ax_fwhm.set_ylabel("PSF FWHM [arcsec]")
            qafig.set_wave_scale(ax_fwhm, xscale)
            qafig.step(ax_sum, wave, product.sums(), color=qafig.MAIN_COLOR)
            ax_sum.set_xlabel(qafig.WAVE_LABEL)
            ax_sum.set_ylabel("in-field energy fraction")
            ax_sum.set_ylim(0, 1.05)
            qafig.set_wave_scale(ax_sum, xscale)
            pa = product.meta.get("JWFPSFPA", "?")
            qafig.annotate(ax_fwhm, inp.name,
                           f"{product.data.shape[2]} px of {product.pixelscale_arcsec:.4f}\", PA {pa}",
                           f"engine {product.meta.get('JWFPSFEN', '?')} ({product.meta.get('JWFPSFMD', '?')})")
            qafig.figlegend(fig)
            out.append(qafig.save(fig, ctx.output_dir / f"{inp.stem}.{fmt}", dpi=dpi))
        return out
