"""The PSF-cube product: slices in memory, on disk, resampled, convolved -- and how to find one.

The file layout and the design choices are described in :mod:`jwstflow.psf`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..steps.base import RunContext

log = logging.getLogger(__name__)

#: Product suffix of PSF cubes (``*_psfcube.fits``).
PSFCUBE_SUFFIX = "psfcube"


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

    def meta_value(self, key: str) -> Any:
        value = self.meta.get(key)
        return value[0] if isinstance(value, tuple) else value

    @property
    def frame(self) -> str | None:
        """Orientation of the slices: 'ideal' (aperture ideal frame) or 'sky', from JWFPSFFR."""
        value = self.meta_value("JWFPSFFR")
        return str(value) if value is not None else None

    @property
    def position_angle_deg(self) -> float | None:
        """Recorded sky PA [deg E of N] of the aperture ideal +y axis (JWFPSFPA), if known."""
        try:
            return float(self.meta_value("JWFPSFPA"))
        except (TypeError, ValueError):
            return None

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

    def resampled(self, pixelscale_arcsec: float, npix: int, *, to_sky: bool = False,
                  rotation_deg: float | None = None, order: int = 3) -> PsfCubeProduct:
        """This library on another grid: scale change and rotation in *one* interpolation.

        The intended consumer path for model comparison: generate the library
        once (distance-independent), then resample it onto each model's
        angular grid -- e.g. ``psf.resampled(model_pixel_scale(d), 256,
        to_sky=True)`` for a trial distance ``d`` -- and convolve. ``to_sky``
        rotates slices stored in the aperture ideal frame to north-up/
        east-left using the recorded position angle (products already in the
        'sky' frame pass through unrotated; without a recorded frame/PA, pass
        ``rotation_deg`` explicitly). ``rotation_deg`` applies the same
        convention as :func:`rotate_cube_to_sky`: the array +y axis ends up
        at that sky position angle.

        Sampling is a cubic spline by default (``order``); small spline
        undershoots are clipped to zero and values are scaled by the
        pixel-area ratio, so each slice's total flux is conserved (up to
        field truncation). Wavelengths, sums bookkeeping and provenance
        travel along; ``JWFPSFPS``/``JWFPSFNP``/``JWFPSFFR`` are updated.
        """
        from scipy.ndimage import map_coordinates

        if to_sky and rotation_deg is not None:
            raise ValueError("give either to_sky or rotation_deg, not both")
        if to_sky:
            if self.frame == "sky":
                theta = 0.0
            elif self.frame == "ideal":
                pa = self.position_angle_deg
                if pa is None:
                    raise ValueError("no recorded position angle (JWFPSFPA); pass rotation_deg explicitly")
                theta = pa
            else:
                raise ValueError("no recorded frame (JWFPSFFR); pass rotation_deg explicitly")
        else:
            theta = float(rotation_deg or 0.0)
        s = float(pixelscale_arcsec) / self.pixelscale_arcsec       # output pixel in input pixels
        cy_in, cx_in = self.centre
        c_out = (int(npix) - 1) / 2.0
        yy, xx = np.indices((int(npix), int(npix)), dtype=float)
        dx, dy = (xx - c_out) * s, (yy - c_out) * s
        # inverse mapping of "rotate array +y to PA theta" (see rotate_cube_to_sky; pinned by a test)
        rad = np.deg2rad(theta)
        ct, st = np.cos(rad), np.sin(rad)
        in_x = cx_in + ct * dx + st * dy
        in_y = cy_in - st * dx + ct * dy
        out = np.empty((len(self.data), int(npix), int(npix)), dtype=float)
        for i, plane in enumerate(self.data):
            resampled = map_coordinates(np.where(np.isfinite(plane), plane, 0.0), [in_y, in_x],
                                        order=order, mode="constant", cval=0.0)
            out[i] = np.clip(resampled, 0.0, None) * s**2           # conserve the integral
        meta = dict(self.meta)
        meta["JWFPSFPS"] = (float(pixelscale_arcsec), "[arcsec/pix] PSF sampling")
        meta["JWFPSFNP"] = (int(npix), "spatial size [pix]")
        if to_sky or rotation_deg is not None:
            meta["JWFPSFFR"] = ("sky" if to_sky else "custom", "orientation after resampling")
        return PsfCubeProduct(out, self.wavelengths.copy(), float(pixelscale_arcsec),
                              meta=meta, header=self.header)

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

    from ..masks import matches_instrument

    cube_hdr = fits.getheader(cube_path)
    candidates = sorted(Path(directory).glob(f"*_{PSFCUBE_SUFFIX}.fits"))
    for f in candidates:
        if matches_instrument(cube_hdr, fits.getheader(f)):
            return PsfCubeProduct.read(f)
    from ..masks import INSTRUMENT_KEYS

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
