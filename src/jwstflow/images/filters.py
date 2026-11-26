"""The filters of JWST's imagers: which band an image was taken in, its wavelengths and its resolution.

The table ``refdata/filters/jwst_imaging_filters.ecsv`` (pivot wavelength, equivalent width,
half-maximum edges of every NIRCam, NIRISS and MIRI imaging filter) is generated from stpsf's
throughput curves by ``scripts/build_filter_table.py``. Lookup order as for the spectral
features: ``$JWSTFLOW_DATA_DIR``, ``<project root>/data``, the package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

#: JWST's primary mirror, the diameter that sets the diffraction limit lambda / D.
TELESCOPE_DIAMETER_M = 6.5
#: FWHM of the diffraction core in units of lambda / D (an Airy core of the hexagonal aperture is
#: within a few percent of 1.03 lambda / D; the imagers' measured FWHM agree to ~10 % from F150W to
#: F2550W -- MIRI F560W / F770W are broadened by the detector by 10-15 %).
FWHM_LAMBDA_OVER_D = 1.03
TABLE = "jwst_imaging_filters.ecsv"


@dataclass(frozen=True)
class Band:
    """One imaging filter."""

    instrument: str          #: NIRCAM | NIRISS | MIRI
    name: str                #: the filter, e.g. F200W (for a NIRCam pupil-wheel narrow band, that band)
    pivot_um: float
    width_um: float          #: equivalent width
    wave_min_um: float       #: half-maximum edges
    wave_max_um: float

    @property
    def fwhm_arcsec(self) -> float:
        """The diffraction-limited FWHM at the pivot wavelength (:func:`diffraction_fwhm_arcsec`)."""
        return diffraction_fwhm_arcsec(self.pivot_um)


def diffraction_fwhm_arcsec(wavelength_um: float) -> float:
    """FWHM of JWST's diffraction core [arcsec] at ``wavelength_um``: 1.03 lambda / D.

    A scale for rules that must follow the resolution of an image (how far apart two peaks must
    be to be two, how wide a neck connects), not a measured PSF.
    """
    import math

    return FWHM_LAMBDA_OVER_D * wavelength_um * 1e-6 / TELESCOPE_DIAMETER_M * math.degrees(1.0) * 3600.0


def data_dir() -> Path:
    """The directory holding ``filters/`` (lookup order in the module docstring)."""
    from ..config.project import find_project_root

    candidates = []
    if os.environ.get("JWSTFLOW_DATA_DIR"):
        candidates.append(Path(os.environ["JWSTFLOW_DATA_DIR"]).expanduser())
    candidates.append(find_project_root(Path.cwd()) / "data")
    candidates.append(Path(__file__).resolve().parents[1] / "refdata")
    for c in candidates:
        if (c / "filters" / TABLE).is_file():
            return c
    raise FileNotFoundError(f"jwstflow's filter table not found (broken install?); looked at {[str(c) for c in candidates]}")


@lru_cache(maxsize=4)
def _bands(directory: str) -> dict[tuple[str, str], Band]:
    from astropy.table import Table

    table = Table.read(Path(directory) / "filters" / TABLE, format="ascii.ecsv")
    return {(str(r["instrument"]).upper(), str(r["filter"]).upper()):
            Band(str(r["instrument"]).upper(), str(r["filter"]).upper(), float(r["pivot_um"]), float(r["width_um"]),
                 float(r["wave_min_um"]), float(r["wave_max_um"])) for r in table}


def all_bands() -> list[Band]:
    """Every filter of the table, blue to red."""
    return sorted(_bands(str(data_dir())).values(), key=lambda b: b.pivot_um)


def band_of(instrument: str | None, filter: str | None, pupil: str | None = None) -> Band | None:
    """The band of an image from its INSTRUME / FILTER / PUPIL keywords, None when unknown.

    Both wheels are looked up; when both hold a filter (NIRCam's F405N in the pupil wheel behind
    F444W) the narrower one defines the band. NIRISS keeps its filters in either wheel
    (CLEAR + F150W), NIRCam names CLEAR in the pupil wheel: whichever wheel holds the filter.
    """
    bands = _bands(str(data_dir()))
    inst = str(instrument or "").strip().upper()
    found = [bands[(inst, str(v).strip().upper())] for v in (filter, pupil)
             if v is not None and (inst, str(v).strip().upper()) in bands]
    return min(found, key=lambda b: b.width_um) if found else None
