"""The PSF-cube product contract: spectrally sub-sampled instrument PSFs for IFU cubes.

Forward-modeling an IFU observation -- rendering a model of the target (a
disk, an envelope) and comparing it with an observed ``*_s3d.fits`` cube --
needs the instrument PSF at the observed wavelengths, sampled on the model's
own pixel grid. This module fixes one file layout for such products so that
the generating step (``psf_cube``), QA and downstream modeling code all speak
the same product:

    <base>_psfcube.fits           derived from the s3d cube it matches
      PRIMARY                     header copied from the source cube (instrument keys
                                  travel along) + JWFPSF* provenance keywords
      PSF       f4 (nw, ny, nx)   PSF slices, each centred on the geometric array
                                  centre ((ny-1)/2, (nx-1)/2); slice sums record the
                                  in-field energy fraction (stpsf ``normalize='first'``)
      WAVETAB   bintable          one row per slice: wavelength_um, psf_sum,
                                  fwhm_arcsec (measured on the slice)

Producers build a :class:`PsfCubeProduct` and call :meth:`PsfCubeProduct.write`;
consumers call :meth:`PsfCubeProduct.read` directly, or :func:`psf_from_stage`
to locate the product of a companion stage that matches a cube's instrument
configuration. :meth:`PsfCubeProduct.at` interpolates a kernel at any
wavelength and :meth:`PsfCubeProduct.convolve` convolves a whole model cube,
so modeling code never re-implements the interpolation or the centring rules.

The design choices (docs/psf_cubes.md carries the full rationale):

* **Wavelengths are sub-sampled.** The PSF varies smoothly with wavelength
  (FWHM ~ lambda/D), so slices at log-spaced wavelengths whose adjacent ratio
  is at most ``1 + max_fractional_step`` (default 2%) bound the FWHM error of
  the nearest slice to ~1% and of linear interpolation to O(step^2/8) ~ 1e-4.
* **The spatial grid defaults to the modeling convention of this ecosystem**:
  a 256x256 grid spanning +-300 AU at the target. At 1 pc, 1 AU subtends
  exactly 1 arcsec, so the pixel scale is (600 AU / 256 px) / d[pc] arcsec
  and model slices convolve without any resampling.
* **Orientation**: slices are stored rotated to the sky frame of ``skyalign``
  cubes (north up, east left) using the aperture position angle from the cube
  header (``PA_APER``, else ``ROLL_REF + V3I_YANG``); the angle used is
  recorded in ``JWFPSFPA``. This matters most for MIRI MRS, whose empirical
  broadening is anisotropic (a slice-width box along the beta axis).
* **Engines are pluggable**: :class:`StpsfEngine` drives stpsf's IFU mode
  (``pip install 'jwstflow[psf]'`` plus the stpsf data files) and
  :class:`GaussianPsfEngine` is an analytic, offline approximation that also
  powers the tests.

Everything imports its heavy dependencies lazily so the module stays
importable without astropy/scipy/stpsf.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .steps.base import RunContext

log = logging.getLogger(__name__)

#: Product suffix of PSF cubes (``*_psfcube.fits``).
PSFCUBE_SUFFIX = "psfcube"

#: The ecosystem's model-rendering convention: slices of MODEL_NPIX x MODEL_NPIX
#: pixels spanning +-MODEL_FOV_AU/2 at the target (rendered at 1 pc, where
#: 1 AU subtends exactly 1 arcsec, then scaled by 1/distance).
MODEL_FOV_AU = 600.0
MODEL_NPIX = 256

#: JWST circumscribed pupil diameter [m] (outer B-segment corners; stpsf constant).
JWST_DIAMETER_M = 6.603464

#: Gaussian sigma [arcsec] of the NIRSpec IFU broadening stpsf applies (half a spaxel).
NIRSPEC_IFU_BROADENING_SIGMA = 0.05

_FWHM_PER_SIGMA = 2.354820045  # 2 sqrt(2 ln 2)
_ARCSEC_PER_RAD = 180.0 / np.pi * 3600.0


# --------------------------------------------------------------------------- grids & sampling
def model_pixel_scale(distance_pc: float, *, fov_au: float = MODEL_FOV_AU, npix: int = MODEL_NPIX) -> float:
    """Pixel scale [arcsec] of the model grid for a target at ``distance_pc``.

    The convention renders ``npix`` pixels over ``fov_au``; at 1 pc, 1 AU
    subtends 1 arcsec (the definition of the parsec), so the angular pixel
    scale is ``(fov_au / npix) / distance_pc``.
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


# --------------------------------------------------------------------------- instrument configuration
#: MRS dichroic position -> sub-band letter (jwst headers say SHORT/MEDIUM/LONG;
#: stpsf and cube filenames say A/B/C).
MRS_SUBBAND = {"SHORT": "A", "MEDIUM": "B", "LONG": "C"}


@dataclass(frozen=True)
class IfuConfig:
    """Instrument configuration of one IFU cube (what selects the PSF model)."""

    instrument: str                 # "MIRI" | "NIRSPEC"
    grating: str | None = None      # NIRSpec disperser (G235H, ..., PRISM)
    filter: str | None = None       # NIRSpec blocking filter (F170LP, ...)
    channel: str | None = None      # MRS channel ("1".."4")
    band: str | None = None         # MRS dichroic (SHORT/MEDIUM/LONG or A/B/C)

    @classmethod
    def from_header(cls, header: Any) -> IfuConfig:
        instrument = str(header.get("INSTRUME", "")).strip().upper()
        if instrument == "MIRI":
            return cls("MIRI", channel=_strip(header.get("CHANNEL")), band=_strip(header.get("BAND")))
        if instrument == "NIRSPEC":
            return cls("NIRSPEC", grating=_strip(header.get("GRATING")), filter=_strip(header.get("FILTER")))
        raise ValueError(
            f"PSF cubes support NIRSpec IFU and MIRI MRS, not INSTRUME={instrument or None!r}"
        )

    @property
    def mrs_band(self) -> str:
        """MRS band in stpsf notation ('1A' ... '4C'); requires a single-channel cube."""
        channel, band = self.channel or "", (self.band or "").upper()
        if len(channel) != 1 or channel not in "1234":
            raise ValueError(
                f"need a single-channel MRS cube to select the PSF model, got CHANNEL={self.channel!r}; "
                "build per-band cubes (cube_build: {output_type: band})"
            )
        letter = MRS_SUBBAND.get(band, band if band in ("A", "B", "C") else None)
        if letter is None:
            raise ValueError(f"unrecognised MRS BAND={self.band!r} (expected SHORT/MEDIUM/LONG)")
        return channel + letter

    @property
    def label(self) -> str:
        """Human-readable band label: '1A' (MRS) or 'G395H/F290LP' (NIRSpec)."""
        if self.instrument == "MIRI":
            try:
                return self.mrs_band
            except ValueError:
                return f"ch{self.channel}-{self.band}"
        return f"{self.grating}/{self.filter}"


def _strip(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


@dataclass
class CubeGeometry:
    """What :func:`read_cube_geometry` extracts from an s3d cube for PSF generation."""

    config: IfuConfig
    wavelengths: np.ndarray            # [um], one per plane
    shape: tuple[int, int]             # spatial (ny, nx)
    pixelscale_arcsec: float | None
    position_angle_deg: float | None   # PA of the aperture ideal +y axis (deg E of N)
    date_obs: str | None               # ISO time for OPD selection
    header: Any                        # the cube's PRIMARY header


def read_cube_geometry(path: Path | str) -> CubeGeometry:
    """Instrument configuration, wavelength grid and orientation of an s3d cube.

    Reads plain FITS (works for jwst products and the synthetic test cubes
    alike): wavelengths from the SCI WCS -- the ``WAVE-TAB`` lookup table when
    the cube carries one (``PS3_0``), the linear axis otherwise -- and the
    aperture position angle from ``PA_APER``, else ``ROLL_REF + V3I_YANG``
    (None when neither is present).
    """
    from astropy.io import fits

    with fits.open(path) as hdul:
        primary = hdul[0].header.copy()
        sci = hdul["SCI"].header.copy() if "SCI" in hdul else hdul[1].header.copy()
        nwave = int(sci.get("NAXIS3", 0) or (hdul["SCI"].data.shape[0] if "SCI" in hdul else 0))
        if sci.get("PS3_0"):
            tab = hdul[str(sci["PS3_0"])].data
            wave = np.asarray(tab[str(sci.get("PS3_1", "wavelength"))], dtype=float).ravel()
        else:
            crval3, cdelt3 = float(sci.get("CRVAL3", 0.0)), float(sci.get("CDELT3", 0.0))
            crpix3 = float(sci.get("CRPIX3", 1.0))
            wave = crval3 + (np.arange(nwave) + 1 - crpix3) * cdelt3
        if str(sci.get("CUNIT3", "um")).strip().lower() in ("m", "meter", "metre"):
            wave = wave * 1e6
        ny, nx = int(sci.get("NAXIS2", 0)), int(sci.get("NAXIS1", 0))
        cdelt2 = sci.get("CDELT2")
        pixelscale = abs(float(cdelt2)) * 3600.0 if cdelt2 is not None else None

    def _angle(*keys: str) -> float | None:
        for hdr in (sci, primary):
            values = [hdr.get(k) for k in keys]
            if all(v is not None for v in values):
                return float(sum(float(v) for v in values))
        return None

    pa = _angle("PA_APER")
    if pa is None:
        pa = _angle("ROLL_REF", "V3I_YANG")
    date_obs = primary.get("DATE-BEG") or primary.get("DATE-OBS")
    if date_obs and primary.get("TIME-OBS") and "T" not in str(date_obs):
        date_obs = f"{date_obs}T{primary['TIME-OBS']}"
    return CubeGeometry(IfuConfig.from_header(primary), np.asarray(wave, float), (ny, nx),
                        pixelscale, pa, str(date_obs) if date_obs else None, primary)


# --------------------------------------------------------------------------- expected widths
def approx_fwhm_arcsec(config: IfuConfig, wavelength_um: np.ndarray | float) -> np.ndarray | float:
    """Approximate as-observed PSF FWHM [arcsec]: MIRI MRS uses the Law et al. 2023
    empirical relation; NIRSpec IFU the diffraction limit (1.025 lambda/D) plus the
    stpsf IFU broadening (sigma = 0.05 arcsec) in quadrature."""
    lam = np.asarray(wavelength_um, dtype=float)
    if config.instrument == "MIRI":
        from .spectra import mrs_psf_fwhm_arcsec

        return mrs_psf_fwhm_arcsec(lam)
    diffraction = diffraction_fwhm_arcsec(lam)
    return np.sqrt(diffraction**2 + (_FWHM_PER_SIGMA * NIRSPEC_IFU_BROADENING_SIGMA) ** 2)


def diffraction_fwhm_arcsec(wavelength_um: np.ndarray | float) -> np.ndarray | float:
    """Diffraction-limited FWHM [arcsec], 1.025 lambda/D for the JWST circumscribed pupil."""
    return 1.025 * np.asarray(wavelength_um, dtype=float) * 1e-6 / JWST_DIAMETER_M * _ARCSEC_PER_RAD


# --------------------------------------------------------------------------- engines
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
    MRS_SLICE_WIDTH = {"1": 0.177, "2": 0.280, "3": 0.390, "4": 0.656}

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


# --------------------------------------------------------------------------- orientation
def rotate_cube_to_sky(stack: np.ndarray, position_angle_deg: float) -> np.ndarray:
    """Rotate PSF slices from the instrument frame into the sky frame of skyalign cubes.

    The slices enter with array +y along the aperture ideal +y axis (the
    :class:`StpsfEngine` convention) and leave with north up and east left.
    ``position_angle_deg`` is the sky position angle (deg E of N) of that
    ideal +y axis -- ``PA_APER`` -- and rotation is about the kernel centre
    ((n-1)/2, the same centre scipy rotates about), so centring is preserved.
    In a north-up/east-left frame a *negative* scipy angle moves +y to
    position angle +PA (pinned by a test).
    """
    from scipy.ndimage import rotate

    angle = float(position_angle_deg) % 360.0
    if angle == 0.0:
        return np.asarray(stack, dtype=float)
    out = np.empty_like(stack, dtype=float)
    for i, plane in enumerate(stack):
        out[i] = rotate(plane, -angle, reshape=False, order=1, mode="constant", cval=0.0)
    return out


# --------------------------------------------------------------------------- FWHM measurement
def measure_fwhm_arcsec(image: np.ndarray, pixelscale_arcsec: float) -> float:
    """FWHM [arcsec] of a PSF slice from its azimuthally averaged radial profile.

    The profile is binned in 1-pixel annuli about the geometric centre and the
    half-maximum crossing linearly interpolated; anisotropic PSFs give an
    azimuthal mean. NaN when the profile never falls below half maximum
    (kernel wider than the grid).
    """
    img = np.asarray(image, dtype=float)
    ny, nx = img.shape
    yy, xx = np.indices((ny, nx))
    r = np.hypot(yy - (ny - 1) / 2.0, xx - (nx - 1) / 2.0).ravel()
    bins = r.astype(int)
    finite = np.isfinite(img).ravel()
    counts = np.bincount(bins, weights=finite.astype(float))
    profile = np.bincount(bins, weights=np.where(finite, img.ravel(), 0.0)) / np.clip(counts, 1, None)
    radius = np.bincount(bins, weights=r) / np.clip(np.bincount(bins), 1, None)   # mean radius per annulus
    peak = float(profile.max(initial=0.0))
    if peak <= 0:
        return float("nan")
    below = np.nonzero(profile < peak / 2)[0]
    if below.size == 0 or below[0] == 0:
        return float("nan")
    j = int(below[0])
    frac = (profile[j - 1] - peak / 2) / max(profile[j - 1] - profile[j], 1e-30)
    r_half = radius[j - 1] + (radius[j] - radius[j - 1]) * float(frac)
    return 2.0 * r_half * pixelscale_arcsec


# --------------------------------------------------------------------------- the product
@dataclass
class PsfCubeProduct:
    """A PSF cube in memory: slices, their wavelengths, and the grid they live on."""

    data: np.ndarray                   # (nw, ny, nx)
    wavelengths: np.ndarray            # [um]
    pixelscale_arcsec: float
    meta: dict[str, Any] = field(default_factory=dict)   # JWFPSF* provenance (written to PRIMARY)
    header: Any = None                 # PRIMARY header after read()

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float)
        self.wavelengths = np.asarray(self.wavelengths, dtype=float)
        if self.data.ndim != 3 or len(self.wavelengths) != len(self.data):
            raise ValueError(f"PSF cube needs (nw, ny, nx) data matching {len(self.wavelengths)} wavelengths, "
                             f"got {self.data.shape}")
        if np.any(np.diff(self.wavelengths) <= 0):
            raise ValueError("PSF wavelengths must be strictly increasing")

    # -- derived ------------------------------------------------------------
    @property
    def centre(self) -> tuple[float, float]:
        """Kernel centre in (y, x) pixels: the geometric array centre."""
        return ((self.data.shape[1] - 1) / 2.0, (self.data.shape[2] - 1) / 2.0)

    def sums(self) -> np.ndarray:
        """Per-slice sums: the fraction of the source flux landing inside the grid."""
        return np.nansum(self.data, axis=(1, 2))

    def fwhm_arcsec(self) -> np.ndarray:
        return np.array([measure_fwhm_arcsec(plane, self.pixelscale_arcsec) for plane in self.data])

    # -- interpolation & convolution ---------------------------------------
    def _weights(self, wavelength_um: float) -> tuple[int, int, float, float]:
        """Bracketing slice indices and linear weights (clamped at the range ends)."""
        w = float(wavelength_um)
        waves = self.wavelengths
        if w <= waves[0]:
            return 0, 0, 1.0, 0.0
        if w >= waves[-1]:
            return len(waves) - 1, len(waves) - 1, 1.0, 0.0
        j = int(np.searchsorted(waves, w))
        i = j - 1
        b = (w - waves[i]) / (waves[j] - waves[i])
        return i, j, 1.0 - float(b), float(b)

    def at(self, wavelength_um: float, *, normalized: bool = True) -> np.ndarray:
        """The PSF at ``wavelength_um``: linear interpolation between the bracketing
        slices (each first normalized to unit sum unless ``normalized=False``),
        clamped to the stored range."""
        i, j, a, b = self._weights(wavelength_um)
        planes = self.data[i], self.data[j]
        if normalized:
            planes = tuple(p / s if (s := float(np.nansum(p))) > 0 else p for p in planes)
        return a * planes[0] + b * planes[1]

    def convolve(self, cube: np.ndarray, wavelengths_um: np.ndarray, *, normalized: bool = True) -> np.ndarray:
        """Convolve a model cube (nw, my, mx) with the wavelength-matched PSF.

        Kernel slices are transformed once; each model plane multiplies the
        linearly interpolated kernel transform (interpolation and the Fourier
        transform commute). ``normalized=True`` scales every kernel slice to
        unit sum, so surface brightness is conserved; the kernel centre
        ((n-1)/2, a half-pixel for even grids) is shifted to the origin
        exactly via the Fourier shift theorem, so the model is not displaced
        (pinned by a delta-function test). NaNs in the model are treated as
        zero flux.
        """
        model = np.asarray(cube, dtype=float)
        squeeze = model.ndim == 2
        if squeeze:
            model = model[None]
        waves = np.atleast_1d(np.asarray(wavelengths_um, dtype=float))
        if len(waves) != len(model):
            raise ValueError(f"model cube has {len(model)} planes but {len(waves)} wavelengths were given")
        my, mx = model.shape[1:]
        _, ky, kx = self.data.shape
        shape = (my + ky, mx + kx)
        fy = np.fft.fftfreq(shape[0])[:, None]
        fx = np.fft.rfftfreq(shape[1])[None, :]
        cy, cx = self.centre
        phase = np.exp(2j * np.pi * (fy * cy + fx * cx))   # translate the kernel centre to the origin
        kernels = np.empty((len(self.data), shape[0], shape[1] // 2 + 1), dtype=complex)
        for i, plane in enumerate(self.data):
            kernels[i] = np.fft.rfft2(np.where(np.isfinite(plane), plane, 0.0), s=shape) * phase
            if normalized:
                dc = kernels[i, 0, 0].real
                if dc > 0:
                    kernels[i] /= dc
        out = np.empty_like(model)
        for p, (plane, w) in enumerate(zip(model, waves)):
            i, j, a, b = self._weights(w)
            transform = np.fft.rfft2(np.where(np.isfinite(plane), plane, 0.0), s=shape)
            kernel = kernels[i] if i == j else a * kernels[i] + b * kernels[j]
            out[p] = np.fft.irfft2(transform * kernel, s=shape)[:my, :mx]
        return out[0] if squeeze else out

    # -- I/O ----------------------------------------------------------------
    def write(self, path: Path | str, *, like: Path | str | None = None, keys: dict[str, Any] | None = None) -> Path:
        """Write the product (layout in the module docstring).

        ``like`` is the source cube whose PRIMARY header is copied so
        instrument keywords travel along; ``keys`` and :attr:`meta`
        add/override PRIMARY keywords (the producing step records its
        parameters there).
        """
        from astropy.io import fits
        from astropy.table import Table

        primary = fits.PrimaryHDU()
        if like is not None:
            primary.header = fits.getheader(like).copy()
        for k, v in {**self.meta, **(keys or {})}.items():
            primary.header[k] = v
        ny, nx = self.data.shape[1:]
        psf = fits.ImageHDU(self.data.astype("f4"), name="PSF")
        for axis, n in ((1, nx), (2, ny)):
            psf.header[f"CTYPE{axis}"] = "OFFSET"
            psf.header[f"CUNIT{axis}"] = "arcsec"
            psf.header[f"CRPIX{axis}"] = (n + 1) / 2.0        # FITS 1-indexed geometric centre
            psf.header[f"CRVAL{axis}"] = 0.0
            psf.header[f"CDELT{axis}"] = self.pixelscale_arcsec
        psf.header["CTYPE3"] = ("WAVE", "irregular; see WAVETAB")
        psf.header["CUNIT3"] = "um"
        psf.header["PIXELSCL"] = (self.pixelscale_arcsec, "[arcsec/pix] PSF sampling")
        table = Table({"wavelength": self.wavelengths, "psf_sum": self.sums(),
                       "fwhm_arcsec": self.fwhm_arcsec()})
        table["wavelength"].unit, table["fwhm_arcsec"].unit = "um", "arcsec"
        fits.HDUList([primary, psf, fits.BinTableHDU(table, name="WAVETAB")]).writeto(path, overwrite=True)
        return Path(path)

    @classmethod
    def read(cls, path: Path | str) -> PsfCubeProduct:
        from astropy.io import fits

        with fits.open(path) as hdul:
            data = np.asarray(hdul["PSF"].data, dtype=float)
            pixelscale = float(hdul["PSF"].header["PIXELSCL"])
            wave = np.asarray(hdul["WAVETAB"].data["wavelength"], dtype=float)
            header = hdul[0].header.copy()
        meta = {k: header[k] for k in header if str(k).startswith("JWFPSF")}
        return cls(data, wave, pixelscale, meta=meta, header=header)


# --------------------------------------------------------------------------- discovery
def find_psf_product(cube_path: Path | str, directory: Path | str) -> PsfCubeProduct:
    """The ``*_psfcube.fits`` in ``directory`` matching the cube's instrument configuration."""
    from astropy.io import fits

    from .masks import matches_instrument

    cube_hdr = fits.getheader(cube_path)
    candidates = sorted(Path(directory).glob(f"*_{PSFCUBE_SUFFIX}.fits"))
    for f in candidates:
        if matches_instrument(cube_hdr, fits.getheader(f)):
            return PsfCubeProduct.read(f)
    from .masks import INSTRUMENT_KEYS

    config = {k: cube_hdr.get(k) for k in INSTRUMENT_KEYS if cube_hdr.get(k) is not None}
    raise FileNotFoundError(
        f"no *_{PSFCUBE_SUFFIX}.fits matching {config} in {directory} ({len(candidates)} product(s) present)"
    )


def psf_from_stage(cube_path: Path, ctx: RunContext, stage: str | None) -> PsfCubeProduct | None:
    """The PSF product of another stage for this cube, or None when the stage is absent.

    Mirrors :func:`jwstflow.masks.mask_from_stage`: None (with a warning) when
    ``stage`` is empty or not in the workflow; raises when the stage exists
    but holds no matching product (declare ``depends_on`` on the consumer).
    """
    if not stage:
        return None
    try:
        psf_dir = ctx.dir_of(stage)
    except KeyError:
        log.warning("stage %r is not part of this workflow; no PSF product available", stage)
        return None
    try:
        return find_psf_product(cube_path, psf_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc}; run the {stage!r} stage first (and declare depends_on: [{stage}])") from exc
