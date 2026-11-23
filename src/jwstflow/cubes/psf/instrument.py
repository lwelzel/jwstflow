"""What selects the PSF: the instrument configuration and geometry of an IFU cube, and the widths to expect."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: JWST circumscribed pupil diameter [m] (outer B-segment corners; stpsf constant).
JWST_DIAMETER_M = 6.603464

#: Gaussian sigma [arcsec] of the NIRSpec IFU broadening stpsf applies (half a spaxel).
NIRSPEC_IFU_BROADENING_SIGMA = 0.05

_FWHM_PER_SIGMA = 2.354820045  # 2 sqrt(2 ln 2)
_ARCSEC_PER_RAD = 180.0 / np.pi * 3600.0

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
    aperture_name: str | None = None   # APERNAME: the reference aperture PA_APER belongs to
    v3_yangle_deg: float | None = None  # V3I_YANG: that aperture's V3IdlYAngle, when the header records it

    def v3_position_angle(self) -> float | None:
        """Sky PA [deg E of N] of the telescope V3 axis at the aperture: ``PA_APER - V3IdlYAngle``.

        The angle of any frame given relative to V3 (an stpsf PSF frame,
        :meth:`StpsfEngine.frame_v3_angle`) is added to it. V3IdlYAngle comes
        from the header (``V3I_YANG``, present in calibrated exposures, not in
        level-3 cubes), else from the SIAF entry of ``APERNAME`` (pysiaf, which
        stpsf brings along); None when neither is available.
        """
        if self.position_angle_deg is None:
            return None
        yangle = self.v3_yangle_deg
        if yangle is None and self.aperture_name:
            yangle = siaf_v3_yangle(self.config.instrument, self.aperture_name)
        return None if yangle is None else float(self.position_angle_deg) - float(yangle)


def siaf_v3_yangle(instrument: str, aperture_name: str) -> float | None:
    """V3IdlYAngle [deg] of a SIAF aperture (pysiaf), or None when pysiaf or the aperture is missing.
    The MRS channel apertures carry 0 there: the PA_APER of an MRS cube is the PA of V3."""
    try:
        import pysiaf
    except ImportError:
        return None
    try:
        value = pysiaf.Siaf(instrument)[str(aperture_name).strip()].V3IdlYAngle
    except Exception:  # noqa: BLE001 -- unknown aperture or SIAF version: the caller falls back
        return None
    return None if value is None or not np.isfinite(value) else float(value)


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
    aperture = next((str(h["APERNAME"]).strip() for h in (sci, primary) if h.get("APERNAME")), None)
    yangle = next((float(h["V3I_YANG"]) for h in (sci, primary) if h.get("V3I_YANG") is not None), None)
    return CubeGeometry(IfuConfig.from_header(primary), np.asarray(wave, float), (ny, nx),
                        pixelscale, pa, str(date_obs) if date_obs else None, primary,
                        aperture_name=aperture, v3_yangle_deg=yangle)


def approx_fwhm_arcsec(config: IfuConfig, wavelength_um: np.ndarray | float) -> np.ndarray | float:
    """Approximate as-observed PSF FWHM [arcsec]: MIRI MRS uses the Law et al. 2023
    empirical relation; NIRSpec IFU the diffraction limit (1.025 lambda/D) plus the
    stpsf IFU broadening (sigma = 0.05 arcsec) in quadrature."""
    lam = np.asarray(wavelength_um, dtype=float)
    if config.instrument == "MIRI":
        return mrs_psf_fwhm_arcsec(lam)
    diffraction = diffraction_fwhm_arcsec(lam)
    return np.sqrt(diffraction**2 + (_FWHM_PER_SIGMA * NIRSPEC_IFU_BROADENING_SIGMA) ** 2)


def mrs_psf_fwhm_arcsec(wavelength_um: np.ndarray | float) -> np.ndarray | float:
    """MIRI MRS PSF FWHM (arcsec) versus wavelength, Law et al. 2023: 0.033 lambda + 0.106."""
    return 0.033 * np.asarray(wavelength_um, dtype=float) + 0.106


def diffraction_fwhm_arcsec(wavelength_um: np.ndarray | float) -> np.ndarray | float:
    """Diffraction-limited FWHM [arcsec], 1.025 lambda/D for the JWST circumscribed pupil."""
    return 1.025 * np.asarray(wavelength_um, dtype=float) * 1e-6 / JWST_DIAMETER_M * _ARCSEC_PER_RAD
