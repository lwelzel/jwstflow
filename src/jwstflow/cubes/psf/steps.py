"""The ``psf_cube`` step: one PSF library per IFU cube.

``psf_cube`` generates one ``*_psfcube.fits`` per input ``*_s3d.fits`` (the
psfcube-product contract, :mod:`jwstflow.cubes.psf`); ``qa_psf_cube``
(:mod:`jwstflow.qa.steps.psf`) plots it. The stpsf engine needs the
``jwstflow[psf]`` extra and the stpsf data files; everything else in the
product path (reading, resampling, interpolation, convolution, QA) stays
dependency-light.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

from ...steps import RunContext, Step, StepParams
from .engines import GaussianPsfEngine, StpsfEngine, stpsf_version
from .grid import PsfGrid, subsample_wavelengths
from .instrument import read_cube_geometry
from .product import PSFCUBE_SUFFIX, PsfCubeProduct, rotate_cube_to_sky

log = logging.getLogger(__name__)

#: DQ bits (jwst) set on the spaxels of a neighbour's core that subtract_psf_neighbour blanks.
DO_NOT_USE, OTHER_BAD_PIXEL = 1, 2**30


class PsfCube(Step):
    """Spectrally sub-sampled instrument PSF library matching an IFU cube.

    For each ``*_s3d.fits`` cube the instrument configuration (NIRSpec
    grating/filter or MRS channel/band) is read from the header and the PSF
    is computed with stpsf's IFU mode -- including its empirical broadening,
    so the slices describe the PSF as realised in reconstructed cubes -- at
    log-spaced wavelengths spanning the cube (adjacent wavelengths within
    ``max_fractional_step``, default 2%), and written as a ``*_psfcube.fits``
    product (contract: ``jwstflow.cubes.psf``).

    The product is a **distance-independent library**: slices live on a
    fixed, fine angular grid -- the cube's own spaxel scale divided by
    ``oversample`` (default 4: 26 mas for NIRSpec, 87-33 mas for MRS ch4-ch1,
    comfortably Nyquist for the broadened PSF) over ``fov_arcsec`` (default
    6") -- and, with the default ``frame: ideal``, stay in the aperture-ideal
    frame stpsf computes in, so nothing is ever interpolated at generation.
    The sky position angle of the slices' +y axis is recorded (``JWFPSFPA``:
    the PA of V3 at the aperture, ``PA_APER`` minus its V3IdlYAngle, plus the
    angle of stpsf's frame from V3 -- for MRS cubes 7.5-8.4 deg more than
    ``PA_APER``, for NIRSpec ``PA_APER`` itself); modeling code then scales
    *and* orients the library onto each model grid in one interpolation:
    ``PsfCubeProduct.read(f).resampled(model_pixel_scale(d), 256,
    to_sky=True).convolve(model, waves)`` -- one library serves every trial
    distance. ``frame: sky`` instead rotates the slices at generation
    (north up, east left), for direct overlay on skyalign cubes.

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
    version = "3"   # 2 -> 3: the PA of the frame stpsf computes in (MRS: +7.5-8.4 deg; was PA_APER)

    class Params(StepParams):
        oversample: int = Field(4, ge=1, le=16, description="library sampling: the cube's spaxel scale divided "
                                "by this (1 = the spaxel grid itself)")
        fov_arcsec: float = Field(6.0, gt=0, le=60, description="angular field of the PSF slices [arcsec] "
                                  "(the pixel count follows from the sampling, rounded up to odd)")
        pixelscale_arcsec: float | None = Field(None, gt=0, description="explicit library pixel scale [arcsec]; "
                                                "overrides oversample")
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
        frame: Literal["ideal", "sky"] = Field("ideal", description="slice orientation: the aperture-ideal frame "
                                               "(no interpolation; PA recorded for consumers) or rotated to "
                                               "north-up/east-left at generation")
        position_angle_deg: float | None = Field(None, description="sky PA [deg E of N] of the slices' +y axis; "
                                                 "overrides what the header and the engine give")

    def run(self, inputs: list[Path], ctx: RunContext, *, oversample: int = 4, fov_arcsec: float = 6.0,
            pixelscale_arcsec: float | None = None, max_fractional_step: float = 0.02,
            n_wavelengths: int | None = None, engine: str = "stpsf", method: str = "exact",
            opd: str = "default", broadening: str = "default", frame: str = "ideal",
            position_angle_deg: float | None = None, **params: Any) -> list[Path]:
        (cube_path,) = inputs
        geom = read_cube_geometry(cube_path)
        if pixelscale_arcsec is not None:
            scale = float(pixelscale_arcsec)
        else:
            if geom.pixelscale_arcsec is None:
                raise ValueError(f"{cube_path.name} carries no spatial WCS scale; give pixelscale_arcsec")
            scale = geom.pixelscale_arcsec / int(oversample)
        npix = int(np.ceil(fov_arcsec / scale)) | 1        # odd: pixel-centred kernels resample cleanly
        psf_grid = PsfGrid(scale, npix)
        waves = subsample_wavelengths(float(geom.wavelengths.min()), float(geom.wavelengths.max()),
                                      max_fractional_step=max_fractional_step, n=n_wavelengths)
        eng = GaussianPsfEngine() if engine == "gaussian" else StpsfEngine(opd=opd, broadening=broadening,
                                                                          method=method)
        ctx.log.info("%s: %s %s, %d wavelengths %.3f-%.3f um, %d px of %.4f\" (%.2f\" field), engine %s",
                     cube_path.name, geom.config.instrument, geom.config.label, len(waves),
                     waves[0], waves[-1], psf_grid.npix, psf_grid.pixelscale_arcsec, psf_grid.fov_arcsec, eng.name)
        stack = eng.compute(geom.config, waves, psf_grid, date_obs=geom.date_obs)

        pa = position_angle_deg if position_angle_deg is not None else _frame_position_angle(geom, eng, cube_path)
        frame_used = frame
        if frame == "sky":
            if pa is None:
                ctx.log.warning("%s: frame 'sky' but no PA_APER/ROLL_REF+V3I_YANG in the header and no "
                                "position_angle_deg given; slices stay in the ideal frame", cube_path.name)
                frame_used = "ideal"
            else:
                stack = rotate_cube_to_sky(stack, float(pa))
        elif pa is None:
            ctx.log.warning("%s: no PA_APER/ROLL_REF+V3I_YANG in the header; the library carries no sky "
                            "orientation -- resampled(to_sky=True) will need an explicit rotation_deg",
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
            "JWFPSFOV": (int(oversample) if pixelscale_arcsec is None else "none",
                         "library sampling: spaxel/this"),
            "JWFPSFNS": (geom.pixelscale_arcsec if geom.pixelscale_arcsec is not None else "none",
                         "[arcsec/pix] native spaxel scale"),
            "JWFPSFFV": (float(fov_arcsec), "[arcsec] requested angular field"),
            "JWFPSFNP": (psf_grid.npix, "spatial size [pix]"),
            "JWFPSFPS": (psf_grid.pixelscale_arcsec, "[arcsec/pix] PSF sampling"),
            "JWFPSFFR": (frame_used, "slice orientation: ideal | sky"),
            "JWFPSFPA": (float(pa) if pa is not None else "none",
                         "[deg E of N] sky PA of the slices' +y axis"),
            "JWFPSFV3": (float(eng.frame_v3_angle) if eng.frame_v3_angle is not None else "none",
                         "[deg] slice +y from V3 (engine frame)"),
        }
        product = PsfCubeProduct(stack, waves, psf_grid.pixelscale_arcsec, meta=meta)
        out = ctx.derived_path(cube_path, PSFCUBE_SUFFIX)
        product.write(out, like=cube_path)
        sums = product.sums()
        ctx.log.info("%s -> %s (in-field energy %.3f-%.3f)", cube_path.name, out.name,
                     float(sums.min()), float(sums.max()))
        return [out]


def _frame_position_angle(geom: Any, engine: Any, cube_path: Path) -> float | None:
    """Sky PA [deg E of N] of the slices' +y axis: the cube's ``PA_APER`` when the engine's frame is the
    aperture's own (or circular), else the PA of V3 at the aperture plus the engine's frame angle.

    For an MRS cube the two differ by 7.5-8.4 deg (the along-slice axis of
    the band against V3; the SIAF gives the MRS apertures no V3IdlYAngle,
    so ``PA_APER`` is the PA of V3 there); for the NIRSpec IFU by nothing.
    Measured on HH 48 SW (MRS ch1-ch4): at this angle the chi2 of the
    library fitted to the star lies within 5 % of its minimum over the
    angle (which falls within 2 deg of it); at PA_APER it is 1.5-2.8 times
    that minimum.
    """
    frame_v3 = getattr(engine, "frame_v3_angle", None)
    if frame_v3 is None or geom.position_angle_deg is None:
        return geom.position_angle_deg
    v3_pa = geom.v3_position_angle()
    if v3_pa is None:
        log.warning("%s: no V3IdlYAngle for aperture %r (no V3I_YANG, no pysiaf); the slices are recorded at "
                    "PA_APER, which is off by the angle of stpsf's frame from the aperture's", cube_path.name,
                    geom.aperture_name)
        return geom.position_angle_deg
    return (v3_pa + float(frame_v3)) % 360.0


class SubtractPsfNeighbour(Step):
    """Model a bright neighbour's PSF in each IFU cube and subtract it, so a faint target next to it can be extracted.

    Made for a faint target with a much brighter point source beside it --
    HH 48 NE, an edge-on disk of a few mJy, 2.3" from HH 48 SW, 10-100x
    brighter in the mid-infrared: in the MRS cubes SW's PSF wings lie over
    the whole field and the disk's aperture ends where SW's light takes
    over. Per ``*_s3d.fits``:

    1. **Where the neighbour is.** ``ra``/``dec`` place it (its centre may
       be outside the field). When its core is in the field, the position
       is fitted (``refine``): the PSF fitted on ``position_blocks``
       block-median images, each started from a centroid, within
       ``refine_radius`` of the given position; the median is adopted.
    2. **Its PSF.** The library of ``psf_stage`` (``psf_cube``) for the
       cube's band, placed at that position on the cube's grid, oriented by
       the library's recorded position angle, interpolated to every plane
       (`jwstflow.cubes.psf.neighbour`). The library must reach every spaxel
       from the neighbour; the step logs the field that takes
       (``JWFNBNED``) and fits only what it reaches. For HH 48 NE that is
       12" (MRS ch1-ch2), 14" (ch3), 17" (ch4), 10" (NIRSpec, SW outside
       the field) -- the default 6" of ``psf_cube`` is not enough.
    3. **Its brightness, plane by plane.** Amplitude + a constant (``background:
       plane`` adds a gradient) fitted to every plane over the field minus
       the target's protected region (``exclude_radius`` + ``exclude_fwhm``
       PSF widths around the target, and the target's contour from
       ``mask_stage`` if given), minus the control ring
       (``holdout_ring``), weighted by ERR with a fractional model error
       (``model_error``), ``clip``-sigma outliers left out. Broadened
       copies of the PSF (``halo_sigma``) can be fitted alongside for light
       the PSF model does not hold. A neighbour outside the field is fitted
       from the part of its wing inside.
       **Line planes** (``line_planes: core``; default ``field``: fitted like
       every plane). Gas-line emission spread over the field is more
       structured than the background terms, and a fit over the field takes
       part of it for the neighbour (HH 48 NE: SW's line flux 1.3-3.2x its
       own aperture's, the excess subtracted under NE). With ``core``, the
       planes within ``line_window`` km/s of the lines of ``line_features``
       take the model of the ``line_context`` continuum planes on either
       side -- every component, the wing, the background, interpolated in
       wavelength -- scaled by the neighbour's brightness in the plane over
       that in the continuum, both fitted on its core (``line_core`` FWHM,
       with a local ``line_background``) with one fixed weighting, so that the
       core's mismatch with the model cancels in the ratio. Continuum planes
       stay exactly as with ``field``. When the core is not in the field
       (or too little of it), the fit over the field stands (logged).
    4. **The subtraction.** The neighbour's model -- not the background
       terms -- is subtracted from every plane; the cube keeps its name and
       every extension (an edited copy of the official product; ``JWFNB*``
       keywords and HISTORY record the edit) and its ERR gains the model's
       error (``inflate_err``). Planes whose fit failed take amplitudes
       interpolated from their neighbours; ``smooth_planes`` runs a median
       over the amplitudes for a neighbour too faint to fit plane by plane.
       Where the neighbour's model exceeds ``blank_core`` times the target's
       core brightness, the cube is blanked (NaN, DQ DO_NOT_USE): no PSF
       model is good to the percent in the core of a star 10-100x the
       target, and the residual there would be taken for a source by the
       mask that follows (HH 48 SW's ended NE's contour).
    5. **The record**, ``*_nbrfit.fits`` (`jwstflow.cubes.psf.subtraction`):
       per plane the neighbour's total flux [Jy] and error (with ``line_planes:
       core`` also the fit over the field's, ``FLUX_FIELD``, and the scale of
       the line planes), the fit's chi2 and residual, and the **control ring**: the mean residual in the
       annulus about the neighbour at the target's distance, outside the
       target's protected region -- what the subtraction leaves at the
       target's distance; held out of the fit by default, so it is a test
       and not a fitted quantity, and consistent with zero when the PSF
       model is right. Block images (data, model, residual), the regions,
       radial profiles about the neighbour and the position fits are stored
       for ``qa_psf_neighbour``.
    """

    inputs = ("*_s3d.fits",)
    outputs = ("nbrfit",)
    writes_official_products = True   # the cleaned cube keeps the official _s3d name; JWFNB* + HISTORY record the edit
    version = "2"

    class Params(StepParams):
        ra: float | str = Field(..., description="the neighbour's right ascension [deg or sexagesimal]")
        dec: float | str = Field(..., description="the neighbour's declination [deg or sexagesimal]")
        target_ra: float | str | None = Field(None, description="the target's RA (default: the workflow's target "
                                              "coordinates, else TARG_RA of the cube)")
        target_dec: float | str | None = Field(None, description="the target's Dec (as target_ra)")
        psf_stage: str = Field("psf_cube", description="stage holding the *_psfcube.fits library of each cube")
        mask_stage: str = Field("", description="optional stage with a mask product of the target (its contour, "
                                "CONT, joins the protected region)")
        mask_suffix: str = Field("diskmask", description="suffix of the mask products in mask_stage")
        exclude_radius: float = Field(1.0, ge=0, description="protected radius about the target [arcsec], before "
                                      "the PSF widths of exclude_fwhm")
        exclude_fwhm: float = Field(1.0, ge=0, description="PSF FWHMs (at the cube's longest wavelength) added to "
                                    "the protected radius and around a mask contour")
        refine: bool = Field(True, description="fit the neighbour's position when its core is in the field")
        refine_radius: float = Field(0.5, gt=0, description="largest shift of the fitted position from ra/dec [arcsec]")
        position_blocks: int = Field(8, ge=1, description="block-median images the position is fitted on")
        background: Literal["constant", "plane", "none"] = Field("constant", description="background terms fitted "
                                                                 "alongside (never subtracted)")
        halo_sigma: list[float] = Field(default_factory=list, description="Gaussian sigmas [arcsec] of broadened "
                                        "PSF copies fitted (and subtracted) with the PSF")
        wing_profile: bool = Field(True, description="fit (and subtract) a radial profile about the neighbour "
                                   "beyond wing_start FWHM: the wing light the PSF model lacks")
        wing_start: float = Field(1.0, gt=0, description="innermost node of the wing profile [FWHM]")
        wing_blocks: int = Field(8, ge=1, description="blocks of planes the wing profile is measured in "
                                 "(interpolated between them)")
        wing_step: float = Field(1.4, gt=1, description="ratio of successive node radii of the wing profile")
        model_error: float = Field(0.05, ge=0, description="fractional uncertainty of the PSF model: in the weights "
                                   "and, with inflate_err, in the cube's ERR")
        clip: float = Field(5.0, gt=0, description="spaxels further off the fit [sigma] are left out (iterated)")
        sampling: Literal["point", "spaxel"] = Field("point", description="spaxel value of the library: sample at "
                                                     "the centre, or mean over the spaxel")
        ring_width: float = Field(0.25, gt=0, description="half-width of the control ring [arcsec] (at least a spaxel)")
        holdout_ring: bool = Field(True, description="leave the control ring out of the fit (an independent test)")
        edge: int = Field(1, ge=0, description="spaxels at the edge of the footprint left out of the fit")
        smooth_planes: int = Field(1, ge=1, description="running median over this many planes of the amplitudes "
                                   "subtracted (1: plane by plane)")
        inflate_err: bool = Field(True, description="add the model's errors to ERR")
        blank_core: float = Field(0.5, ge=0, description="blank (NaN, DQ DO_NOT_USE) the spaxels where the "
                                  "neighbour's model exceeds this multiple of the target's core brightness -- "
                                  "never within exclude_radius of the target or in its mask contour (0: keep all)")
        show_blocks: int = Field(4, ge=1, le=12, description="blocks of planes stored as images for the QA figure")
        line_planes: Literal["field", "core"] = Field("field", description="the planes of gas lines: fitted over the "
                                                      "field like every plane (field), or the neighbouring continuum "
                                                      "planes' model scaled by the neighbour's brightness on its core, "
                                                      "line plane over continuum (core): line emission spread over the "
                                                      "field is then not taken for the neighbour's")
        line_features: Any = Field(default_factory=lambda: {"gas_lines": "all"}, description="the lines whose planes "
                                   "line_planes: core treats (jwstflow feature selection; bands are ignored)")
        line_window: float = Field(300.0, gt=0, description="half-width [km/s] of the planes of a line")
        line_core: float = Field(1.0, gt=0, description="radius [PSF FWHM at the line] of the neighbour's core that "
                                 "the line planes' scale is fitted on")
        line_context: int = Field(5, ge=1, description="continuum planes on either side of a line that its model is "
                                  "interpolated from")
        line_background: Literal["constant", "plane"] = Field("plane", description="local background fitted with the "
                                                              "neighbour's model on its core (the field's emission "
                                                              "across it)")

    def run(self, inputs: list[Path], ctx: RunContext, *, ra: float | str, dec: float | str,
            target_ra: float | str | None = None, target_dec: float | str | None = None, psf_stage: str = "psf_cube",
            mask_stage: str = "", mask_suffix: str = "diskmask", exclude_radius: float = 1.0,
            exclude_fwhm: float = 1.0, refine: bool = True, refine_radius: float = 0.5, position_blocks: int = 8,
            background: str = "constant", halo_sigma: list[float] | None = None, wing_profile: bool = True,
            wing_start: float = 1.0, wing_step: float = 1.4, wing_blocks: int = 8, model_error: float = 0.05,
            clip: float = 5.0, sampling: str = "point", ring_width: float = 0.25, holdout_ring: bool = True,
            edge: int = 1, smooth_planes: int = 1, inflate_err: bool = True, blank_core: float = 0.5,
            show_blocks: int = 4, line_planes: str = "field", line_features: Any = None, line_window: float = 300.0,
            line_core: float = 1.0, line_context: int = 5, line_background: str = "plane",
            **params: Any) -> list[Path]:
        import warnings

        from astropy.io import fits
        from scipy.ndimage import binary_erosion

        from ...data.targets import parse_coords
        from ...spectra.features import select_features
        from ..masks import mask_from_stage
        from ..morphology import dilate
        from ..wcs import celestial_wcs_from_header, hdu_wavelengths
        from . import neighbour as nb
        from .instrument import IfuConfig, approx_fwhm_arcsec
        from .product import psf_from_stage
        from .subtraction import (
            BEYOND_LIBRARY,
            BLANKED,
            FITTED,
            NBRFIT_SUFFIX,
            PROTECTED,
            RING,
            PsfSubtraction,
        )

        (cube_path,) = inputs
        halo_sigma = list(halo_sigma or [])
        if line_planes not in ("field", "core"):
            raise ValueError(f"line_planes must be 'field' or 'core', not {line_planes!r}")
        line_features = {"gas_lines": "all"} if line_features is None else line_features
        with fits.open(cube_path) as hdul:
            primary = hdul[0].header.copy()
            sci_hdr = hdul["SCI"].header.copy()
            data = np.asarray(hdul["SCI"].data, dtype="f8")
            err = np.asarray(hdul["ERR"].data, dtype="f8") if "ERR" in hdul else np.ones_like(data)
            dq = np.asarray(hdul["DQ"].data) if "DQ" in hdul else None
            wave = hdu_wavelengths(hdul)
        usable = data.copy()
        if dq is not None:
            usable[(dq & 1) > 0] = np.nan                       # DO_NOT_USE: not fitted (still subtracted)
        wcs = celestial_wcs_from_header(sci_hdr)
        shape = data.shape[1:]
        ny, nx = shape
        pix = (float(np.sqrt(sci_hdr["PIXAR_A2"])) if sci_hdr.get("PIXAR_A2")
               else abs(float(sci_hdr["CDELT2"])) * 3600.0)
        area_sr = float(sci_hdr.get("PIXAR_SR") or (pix / 206264.806) ** 2)
        unit = str(sci_hdr.get("BUNIT", "MJy/sr")).strip()
        to_jy = area_sr * 1e6 if unit.replace(" ", "") in ("MJy/sr", "MJy.sr-1") else 1.0
        if to_jy == 1.0:
            ctx.log.warning("%s: BUNIT %r is not MJy/sr; FLUX columns are in cube units x spaxel", cube_path.name, unit)

        # -- the library and its orientation on this cube -------------------
        psf = psf_from_stage(cube_path, ctx, psf_stage)
        if psf is None:
            raise FileNotFoundError(f"{cube_path.name}: subtract_psf_neighbour needs the PSF library of stage "
                                    f"{psf_stage!r} (psf_cube)")
        here, up = wcs.all_pix2world([[nx / 2, ny / 2], [nx / 2, ny / 2 + 1]], 0)
        north = _position_angle(here, up)                       # sky PA of the cube's +y
        if psf.frame == "ideal":
            if psf.position_angle_deg is None:
                raise ValueError(f"{cube_path.name}: the PSF library records no position angle (JWFPSFPA)")
            rotation = float(psf.position_angle_deg) - north
        else:
            rotation = -north

        # -- positions -------------------------------------------------------
        nra, ndec = parse_coords(ra, dec)
        if target_ra is not None and target_dec is not None:
            tra, tdec = parse_coords(target_ra, target_dec)
        elif ctx.target_coords and ctx.target_coords.get("ra") is not None:
            tra, tdec = parse_coords(ctx.target_coords["ra"], ctx.target_coords["dec"])
        elif primary.get("TARG_RA") is not None:
            tra, tdec = float(primary["TARG_RA"]), float(primary["TARG_DEC"])
        else:
            raise ValueError(f"{cube_path.name}: no target position (target_ra/target_dec, the workflow's target "
                             "coordinates or TARG_RA/TARG_DEC)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (gx, gy), (tx, ty) = (tuple(float(v) for v in wcs.all_world2pix(r, d, 0)) for r, d in
                                  ((nra, ndec), (tra, tdec)))
        config = IfuConfig.from_header(primary)
        fwhm_max = float(approx_fwhm_arcsec(config, float(np.nanmax(wave))))
        fwhm_mid = float(approx_fwhm_arcsec(config, float(np.nanmedian(wave))))

        # -- regions ---------------------------------------------------------
        yy, xx = np.indices(shape, dtype=float)
        protect_radius = exclude_radius + exclude_fwhm * fwhm_max
        protected = np.hypot(xx - tx, yy - ty) * pix <= protect_radius
        inner = np.hypot(xx - tx, yy - ty) * pix <= exclude_radius
        contour = np.zeros(shape, bool)
        mask_note = "none"
        if mask_stage:
            product = mask_from_stage(cube_path, ctx, mask_stage, mask_suffix)
            if product is not None:
                if product["cont"].shape != shape:
                    raise ValueError(f"{cube_path.name}: the mask of {mask_stage!r} is {product['cont'].shape}, "
                                     f"the cube {shape}")
                contour = product["cont"]
                protected |= dilate(contour, int(np.ceil(exclude_fwhm * fwhm_max / pix)))
                mask_note = product["path"].name
        footprint = np.isfinite(data).any(axis=0)
        fit_footprint = binary_erosion(footprint, iterations=edge) if edge > 0 else footprint

        located = nb.locate(usable, err, wave, psf, (gx, gy), pixel_arcsec=pix, rotation_deg=rotation,
                            exclude=protected, fwhm_pix=fwhm_mid / pix, max_shift=refine_radius / pix,
                            n_blocks=position_blocks, model_error=model_error, sampling=sampling, refine=refine)
        x0, y0 = located.x, located.y
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fra, fdec = (float(v) for v in wcs.all_pix2world(x0, y0, 0))
        reach = nb.library_reach(psf, shape, x0, y0, pix)
        needed = nb.required_field(shape, footprint, x0, y0, pix)
        lib_field = psf.data.shape[-1] * psf.pixelscale_arcsec
        if not reach[footprint].all():
            ctx.log.warning("%s: the PSF library's %.1f\" field does not reach %d of %d spaxels from the neighbour; "
                            "they are neither fitted nor cleaned -- regenerate it with psf_cube fov_arcsec >= %.1f",
                            cube_path.name, lib_field, int((footprint & ~reach).sum()), int(footprint.sum()), needed)
        separation = float(np.hypot(tx - x0, ty - y0))
        half = max(1.0, ring_width / pix)
        ring = nb.ring_mask(shape, (x0, y0), separation, half) & ~protected & fit_footprint & reach
        toward = ((xx - x0) * (tx - x0) + (yy - y0) * (ty - y0)) > 0
        fit_mask = fit_footprint & ~protected & reach & ~(ring if holdout_ring else False)
        far = fit_footprint & ~protected & reach & (np.hypot(xx - x0, yy - y0) > separation + half)

        # -- the fit -----------------------------------------------------------
        libraries = [psf] + [nb.broadened(psf, s) for s in halo_sigma]
        comps = nb.components(libraries, shape, x0, y0, wave, pixel_arcsec=pix, rotation_deg=rotation,
                              sampling=sampling)
        r_far = float(np.hypot(xx - x0, yy - y0)[footprint & reach].max())
        nodes = np.zeros(0)
        if wing_profile and located.refined:
            nodes = nb.radial_nodes(fwhm_mid / pix, r_far, start=wing_start, factor=wing_step)
        elif wing_profile:
            ctx.log.info("%s: the neighbour's core is outside the field; no wing profile (the wing is all "
                         "there is to fit the PSF to)", cube_path.name)
        fit = nb.fit_planes(usable, err, comps, fit_mask, background=background, origin=(tx, ty),
                            model_error=model_error, clip=clip)
        wing = np.zeros_like(comps[0])
        wing_coeffs = np.zeros((0, len(nodes)))
        if len(nodes):
            tents = nb.radial_tents(shape, x0, y0, nodes, r_far)
            psf_only = comps[0]
            for _ in range(2):   # residual profile of the current model, then the planes refitted with it
                amp = nb.fill_amplitudes(fit.amplitudes)[0][:, 0]
                with np.errstate(invalid="ignore", divide="ignore"):
                    frac = (usable - fit.model(comps) - fit.background(shape)) / amp[:, None, None]
                    frac_err = err / np.abs(amp)[:, None, None]
                images, errors, ranges = nb.block_stack(frac, frac_err, wing_blocks)
                block_coeffs = np.stack([nb.fit_wing(im, er, tents, fit_mask) for im, er in zip(images, errors)])
                block_waves = np.array([np.nanmedian(wave[a:b]) for a, b in ranges])
                wing = wing + nb.wing_shape(block_coeffs, block_waves, wave, tents)
                wing_coeffs = block_coeffs if not len(wing_coeffs) else wing_coeffs + block_coeffs
                # a wing profile may lower the model's wings, never below zero: a negative model would
                # add light where it is subtracted (it took up background structure in ch1 of HH 48 NE)
                comps[0] = np.clip(psf_only + wing, 0.0, None)
                fit = nb.fit_planes(usable, err, comps, fit_mask, background=background, origin=(tx, ty),
                                    model_error=model_error, clip=clip)
        amps, filled = nb.fill_amplitudes(fit.amplitudes, smooth=smooth_planes)
        model = sum(amps[:, i, None, None] * c for i, c in enumerate(comps))
        model[:, ~reach] = 0.0
        bkg = fit.background(shape)
        # the neighbour's core (blanked below) is judged on the fit over the field, in every plane
        core = np.hypot(xx - tx, yy - ty) * pix <= fwhm_mid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            blank_image = np.nanmedian(model, axis=0)
        blank_level = float(np.nanmedian(nb.ring_residual(data - model - bkg, err, core)[0]))

        # -- line planes: the continuum's model, scaled on the neighbour's core -----
        field_total, field_psf = fit.total.copy(), fit.amplitudes[:, 0].copy()
        line = np.zeros(len(wave), bool)
        unlisted = np.zeros(len(wave), bool)
        rescaled = None
        if line_planes == "core":
            centres = [f.wavelength for f in select_features(line_features)
                       if f.kind == "line" and f.wavelength is not None]
            line = nb.feature_planes(wave, centres, line_window)
            unlisted = nb.unlisted_excess(field_total, line)
            if not located.refined:
                ctx.log.warning("%s: the neighbour's core is not in the field; its %d line planes keep the fit over "
                                "the field", cube_path.name, int(line.sum()))
            elif line.any():
                rescaled = nb.scale_line_planes(
                    usable, err, comps, amps, line, wave, centre=(x0, y0),
                    core_radius=line_core * np.asarray(approx_fwhm_arcsec(config, wave)) / pix,
                    allowed=fit_footprint & ~protected & reach, const=fit.const, gradient=fit.gradient,
                    total_err=fit.total_err, valid=~filled, context=line_context, background=line_background)
                rows = rescaled.line
                amps[rows] = rescaled.amplitudes[rows]
                fit.amplitudes[rows] = amps[rows]
                fit.total[rows] = amps[rows].sum(axis=1)
                fit.total_err[rows] = rescaled.total_err[rows]
                fit.const[rows] = rescaled.const[rows]
                fit.gradient[rows] = rescaled.gradient[rows]
                line_model = sum(amps[rows][:, i, None, None] * (c if len(c) == 1 else c[rows])
                                 for i, c in enumerate(comps))
                line_model[:, ~reach] = 0.0
                model[rows] = line_model
                bkg[rows] = fit.background(shape)[rows]
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = fit.total[rows] / field_total[rows]
                ctx.log.info("%s: %d planes of %d lines (+-%g km/s) take the continuum model scaled on the "
                             "neighbour's core (scale %.2f-%.2f; against the fit over the field %.2f-%.2f)",
                             cube_path.name, int(rows.sum()), len(nb.plane_runs(line)), line_window,
                             float(np.nanmin(rescaled.scale)) if rows.any() else np.nan,
                             float(np.nanmax(rescaled.scale)) if rows.any() else np.nan,
                             float(np.nanmin(ratio)) if rows.any() else np.nan,
                             float(np.nanmax(ratio)) if rows.any() else np.nan)
                if rescaled.skipped:
                    ctx.log.warning("%s: %d line planes keep the fit over the field (%s)", cube_path.name,
                                    len(rescaled.skipped), "; ".join(sorted({why for _, why in rescaled.skipped})))
            if unlisted.any():
                at = ", ".join(f"{wave[(a + b - 1) // 2]:.3f}" for a, b in nb.plane_runs(unlisted)[:12])
                ctx.log.warning("%s: the fitted flux of the neighbour jumps in %d planes outside the listed lines "
                                "(at %s um): lines line_features does not name?", cube_path.name, int(unlisted.sum()),
                                at)
        cleaned = data - model
        resid = cleaned - bkg
        new_err = err
        if inflate_err:
            with np.errstate(invalid="ignore", divide="ignore"):
                relative = np.where(np.abs(fit.total) > 0, fit.total_err / np.abs(fit.total), 0.0)
            new_err = np.sqrt(err**2 + (np.nan_to_num(relative)[:, None, None] * model) ** 2
                              + (model_error * model) ** 2)

        # -- diagnostics --------------------------------------------------------
        ring_mean, ring_err, _ = nb.ring_residual(resid, err, ring)
        side, _, _ = nb.ring_residual(resid, err, ring & toward)
        back, _, _ = nb.ring_residual(resid, err, ring & ~toward)
        before, _, _ = nb.ring_residual(data - bkg, err, ring)
        sky_far, _, _ = nb.ring_residual(resid, err, far)
        target_sb, _, _ = nb.ring_residual(resid, err, core)
        with np.errstate(invalid="ignore"):
            fit_rms = np.sqrt(np.nanmean(np.where(fit_mask[None], resid, np.nan) ** 2, axis=(1, 2)))
        planes = {
            "WAVELENGTH": wave, "FLUX": fit.total * to_jy, "FLUX_ERR": fit.total_err * to_jy,
            "FLUX_PSF": fit.amplitudes[:, 0] * to_jy, "SUBTRACTED": model.sum(axis=(1, 2)) * to_jy,
            "CONST": fit.const, "GRADX": fit.gradient[:, 0], "GRADY": fit.gradient[:, 1],
            "CHI2R": fit.reduced_chi2, "NFIT": fit.nfit, "NCLIP": fit.nclip, "FITRMS": fit_rms,
            "RING": ring_mean, "RING_ERR": ring_err, "RING_SIDE": side, "RING_BACK": back, "RING_BEFORE": before,
            "SKY_FAR": sky_far, "TARGET": target_sb, "FILLED": filled.astype("u1"),
        }
        for i, s in enumerate(halo_sigma, start=1):
            planes[f"FLUX_HALO{i}"] = fit.amplitudes[:, i] * to_jy
        if len(nodes):
            wing_share = np.nansum(wing, axis=(1, 2))                        # wing light per unit amplitude
            planes["FLUX_PSF"] = fit.amplitudes[:, 0] * to_jy
            planes["FLUX_WING"] = fit.amplitudes[:, 0] * wing_share * to_jy
            planes["FLUX"] = planes["FLUX"] + planes["FLUX_WING"]
            planes["FLUX_ERR"] = planes["FLUX_ERR"] * (1.0 + wing_share)
        if line_planes == "core":
            planes["FLUX_FIELD"] = field_total * to_jy            # as FLUX would be without line_planes, to the bit
            if len(nodes):
                planes["FLUX_FIELD"] = planes["FLUX_FIELD"] + field_psf * wing_share * to_jy
            done = rescaled.line if rescaled is not None else np.zeros(len(wave), bool)
            planes["LINE"] = done.astype("u1")
            planes["LINE_SCALE"] = rescaled.scale if rescaled is not None else np.full(len(wave), np.nan)
            planes["LINE_SCALE_ERR"] = rescaled.scale_err if rescaled is not None else np.full(len(wave), np.nan)
        edges = np.linspace(0, len(wave), int(show_blocks) + 1).round().astype(int)
        firsts, lasts = edges[:-1], np.maximum(edges[1:] - 1, edges[:-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            images = np.stack([np.stack([np.nanmedian(cube_[a:b + 1], axis=0) for a, b in zip(firsts, lasts)])
                               for cube_ in (data, np.where(np.isfinite(data), model, np.nan), resid)])
            before_images = np.stack([np.nanmedian((data - bkg)[a:b + 1], axis=0) for a, b in zip(firsts, lasts)])
        r_step = max(1.0, 0.1 / pix)
        r_edges = np.arange(0.0, float(np.hypot(xx - x0, yy - y0)[footprint].max()) + r_step, r_step)
        profile: dict[str, np.ndarray] = {"R_ARCSEC": 0.5 * (r_edges[:-1] + r_edges[1:]) * pix}
        outside = ~protected & footprint
        for b in range(len(firsts)):
            profile[f"BEFORE_{b}"] = nb.radial_profile(before_images[b], (x0, y0), r_edges, mask=outside)
            profile[f"MODEL_{b}"] = nb.radial_profile(images[1, b], (x0, y0), r_edges, mask=outside)
            profile[f"AFTER_{b}"] = nb.radial_profile(images[2, b], (x0, y0), r_edges, mask=outside)
        # the neighbour's core: its residual outshines the target, so it is not handed on as data
        blanked = np.zeros(shape, bool)
        if blank_core > 0 and np.isfinite(blank_level) and blank_level > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                # never within the bare exclude radius of the target, nor in its mask contour
                blanked = (blank_image > blank_core * blank_level) & ~inner & ~contour
        regions = ((fit_mask * FITTED) | (protected * PROTECTED) | (ring * RING)
                   | ((footprint & ~reach) * BEYOND_LIBRARY) | (blanked * BLANKED)).astype("u1")

        ok = np.isfinite(ring_mean)
        ring_median = float(np.nanmedian(ring_mean)) if ok.any() else float("nan")
        ring_scatter = float(1.4826 * np.nanmedian(np.abs(ring_mean - ring_median))) if ok.any() else float("nan")
        keys: dict[str, Any] = {
            "JWFNBVER": (self.version, "subtract_psf_neighbour step version"),
            "JWFNBRA": (nra, "[deg] neighbour RA given"), "JWFNBDEC": (ndec, "[deg] neighbour Dec given"),
            "JWFNBFRA": (fra, "[deg] neighbour RA adopted"), "JWFNBFDE": (fdec, "[deg] neighbour Dec adopted"),
            "JWFNBX": (x0, "neighbour x adopted [0-based pix]"), "JWFNBY": (y0, "neighbour y adopted [0-based pix]"),
            "JWFNBGX": (gx, "neighbour x given [0-based pix]"), "JWFNBGY": (gy, "neighbour y given [0-based pix]"),
            "JWFNBREF": (located.refined, "position fitted on the cube (core in field)"),
            "JWFNBCOV": (located.coverage, "share of the core's light on finite spaxels"),
            "JWFNBTRA": (tra, "[deg] target RA"), "JWFNBTDE": (tdec, "[deg] target Dec"),
            "JWFNBTX": (tx, "target x [0-based pix]"), "JWFNBTY": (ty, "target y [0-based pix]"),
            "JWFNBSEP": (separation * pix, "[arcsec] target-neighbour distance"),
            "JWFNBPIX": (pix, "[arcsec] spaxel"),
            "JWFNBLIB": (psf.source.name if psf.source is not None else "", "PSF library"),
            "JWFNBFOV": (lib_field, "[arcsec] PSF library field"),
            "JWFNBNED": (needed, "[arcsec] library field reaching every spaxel"),
            "JWFNBROT": (rotation, "[deg] library +y from the cube's +y"),
            "JWFNBPRO": (protect_radius, "[arcsec] protected radius about the target"),
            "JWFNBMSK": (mask_note, "mask product joined to the protected region"),
            "JWFNBRNG": (separation * pix, "[arcsec] control ring radius"),
            "JWFNBRWD": (half * pix, "[arcsec] control ring half-width"),
            "JWFNBHLD": (holdout_ring, "control ring held out of the fit"),
            "JWFNBBKG": (background, "background terms fitted (not subtracted)"),
            "JWFNBHAL": (",".join(f"{s:g}" for s in halo_sigma) or "none", "[arcsec] halo sigmas"),
            "JWFNBWNG": (",".join(f"{r * pix:.2f}" for r in nodes) or "none", "[arcsec] wing profile nodes"),
            "JWFNBMER": (model_error, "fractional PSF model error"),
            "JWFNBCLP": (clip, "[sigma] clip"), "JWFNBSMP": (sampling, "library sampling of a spaxel"),
            "JWFNBSMO": (smooth_planes, "running median of amplitudes [planes]"),
            "JWFNBERR": (inflate_err, "model errors added to ERR"),
            "JWFNBFLX": (float(np.nanmedian(planes["FLUX"])), "[Jy] median neighbour flux"),
            "JWFNBCHI": (float(np.nanmedian(fit.reduced_chi2)), "median reduced chi2"),
            "JWFNBRMD": (ring_median, "median control-ring residual [BUNIT]"),
            "JWFNBRSC": (ring_scatter, "its robust scatter over planes [BUNIT]"),
            "JWFNBRBF": (float(np.nanmedian(before)), "median ring level before [BUNIT]"),
            "JWFNBTSB": (float(np.nanmedian(target_sb)), "median target core SB after [BUNIT]"),
            "JWFNBNFL": (int(filled.sum()), "planes with interpolated amplitudes"),
            "JWFNBBLK": (blank_core, "blanked where model > this x target core"),
            "JWFNBNBL": (int(blanked.sum()), "spaxels blanked (neighbour's core)"),
            "JWFNBLPL": (line_planes, "line planes: field fit | core-scaled continuum"),
        }
        if line_planes == "core":
            keys.update({
                "JWFNBLFT": (str(line_features)[:40], "lines treated (line_features)"),
                "JWFNBLWN": (line_window, "[km/s] half-width of a line's planes"),
                "JWFNBLCO": (line_core, "[FWHM] core radius of the line-plane scale"),
                "JWFNBLCX": (line_context, "continuum planes per side of a line"),
                "JWFNBLBG": (line_background, "local background of the core fit"),
                "JWFNBNLN": (int(line.sum()), "planes within the window of a listed line"),
                "JWFNBNLR": (int(rescaled.line.sum()) if rescaled is not None else 0,
                             "line planes scaled on the neighbour's core"),
                "JWFNBNUN": (int(unlisted.sum()), "planes with a flux jump off the listed lines"),
            })
        out_product = ctx.derived_path(cube_path, NBRFIT_SUFFIX)
        PsfSubtraction(planes=planes, images=images, blocks={"FIRST": firsts, "LAST": lasts,
                                                             "WAVELENGTH": wave[(firsts + lasts) // 2]},
                       regions=regions, profile=profile, positions=located.blocks,
                       keys=keys).write(out_product, cube_path)

        out_cube = ctx.output_dir / cube_path.name
        cleaned[:, blanked] = np.nan
        new_err = np.array(new_err, copy=True)
        new_err[:, blanked] = np.nan
        with fits.open(cube_path) as hdul:
            hdul["SCI"].data = cleaned.astype(hdul["SCI"].data.dtype)
            if "ERR" in hdul:
                hdul["ERR"].data = np.asarray(new_err, dtype=hdul["ERR"].data.dtype)
            if "DQ" in hdul and blanked.any():
                dq_out = np.array(hdul["DQ"].data, copy=True)
                dq_out[:, blanked] |= np.asarray(DO_NOT_USE | OTHER_BAD_PIXEL, dtype=dq_out.dtype)
                hdul["DQ"].data = dq_out
            hdr = hdul[0].header
            hdr["JWFNBSUB"] = (True, "subtract_psf_neighbour: a neighbour's PSF subtracted")
            for k in ("JWFNBVER", "JWFNBFRA", "JWFNBFDE", "JWFNBLIB", "JWFNBHAL", "JWFNBFLX", "JWFNBRMD", "JWFNBNBL",
                      "JWFNBLPL"):
                hdr[k] = keys[k]
            hdr["JWFNBREC"] = (out_product.name, "subtract_psf_neighbour: record of the fit")
            hdr.add_history(f"subtract_psf_neighbour v{self.version}: PSF of the neighbour at RA {fra:.6f} "
                            f"Dec {fdec:.6f} (library {keys['JWFNBLIB'][0] or 'psf_cube'}"
                            f"{', halo ' + keys['JWFNBHAL'][0] if halo_sigma else ''}) fitted plane by plane "
                            f"and subtracted from SCI; background terms fitted, not subtracted; ERR "
                            f"{'includes' if inflate_err else 'excludes'} the model error; {int(blanked.sum())} "
                            f"spaxels of the neighbour's core blanked (DQ DO_NOT_USE)"
                            + (f"; in {int(rescaled.line.sum())} planes of gas lines the continuum's model scaled "
                               f"on the neighbour's core" if rescaled is not None else "")
                            + f"; see {out_product.name}")
            hdul.writeto(out_cube, overwrite=True)
        ctx.log.info("%s: neighbour %s at (%.2f, %.2f) [given (%.2f, %.2f), core coverage %.2f]; library %.1f\" "
                     "(needs %.1f\"); median flux %.2f mJy, reduced chi2 %.2f; control ring at %.2f\": %+.3f "
                     "(scatter %.3f) vs %+.3f before, target core %.3f [%s]", cube_path.name,
                     "fitted" if located.refined else "as given", x0, y0, gx, gy, located.coverage, lib_field,
                     needed, keys["JWFNBFLX"][0] * 1e3, keys["JWFNBCHI"][0], separation * pix, ring_median,
                     ring_scatter, keys["JWFNBRBF"][0], keys["JWFNBTSB"][0], unit)
        return [out_cube, out_product]


def _position_angle(here: Any, there: Any) -> float:
    """Sky position angle [deg E of N] of ``there`` seen from ``here`` (both (ra, dec) in deg)."""
    ra1, dec1, ra2, dec2 = (np.deg2rad(float(v)) for v in (*here, *there))
    y = np.sin(ra2 - ra1) * np.cos(dec2)
    x = np.cos(dec1) * np.sin(dec2) - np.sin(dec1) * np.cos(dec2) * np.cos(ra2 - ra1)
    return float(np.rad2deg(np.arctan2(y, x)))
