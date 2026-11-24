"""Archival photometry and spectra of a target, kept by hand next to its data.

External measurements (broadband photometry from the literature, archival
spectra) are not downloaded by a recipe: they are curated by hand and stored
per target as ECSV files in ``<target>/archival/`` -- next to ``raw/`` and
``mast_reference/``, so a re-reduction never removes them. The QA step
``plot_archival`` draws them over a reduction's spectra.

Two kinds of file, told apart by ``meta['kind']``:

``photometry`` -- one row per measurement:

=============== ============ ==========================================================
column          unit         meaning
=============== ============ ==========================================================
WAVELENGTH      um           reference wavelength of the band
BANDWIDTH       um           full width of the band (optional; NaN where unknown)
FLUX            Jy (F_nu)    flux density; for an upper limit the limit itself
FLUX_ERROR      Jy           1-sigma uncertainty (for an upper limit the 1-sigma noise)
UPPER_LIMIT     bool         True: FLUX is an upper limit at ``meta['upper_limit_sigma']``
REFERENCE       --           citation key of the source (e.g. the bibtex key)
CITATION        --           the source as text (``Author et al. Year``)
NOTE            --           free text (instrument, caveats); optional
=============== ============ ==========================================================

``spectrum`` -- an archival spectrum: WAVELENGTH, FLUX and optionally
FLUX_ERROR, with ``meta['reference']`` / ``meta['citation']``.

Units are read from the columns (any F_nu or F_lambda unit astropy converts
to Jy; wavelengths in any length unit). Every file should say in its
metadata where its numbers come from (``description``, ``source_file``,
``sources``). :func:`photometry_table` builds a valid photometry table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ARCHIVAL_DIR = "archival"
KINDS = ("photometry", "spectrum")
PHOTOMETRY_COLUMNS = ("WAVELENGTH", "FLUX", "FLUX_ERROR", "UPPER_LIMIT")


@dataclass
class ArchivalData:
    """One archival file, converted: wavelengths in um, fluxes in Jy (F_nu)."""

    path: Path
    kind: str
    wavelength: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    upper_limit: np.ndarray
    bandwidth: np.ndarray
    reference: np.ndarray
    citation: np.ndarray
    note: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """What to call the file's data in a legend (a spectrum: its citation)."""
        return str(self.meta.get("citation") or self.meta.get("reference") or self.meta.get("label")
                   or self.path.stem)

    @property
    def upper_limit_sigma(self) -> float | None:
        value = self.meta.get("upper_limit_sigma")
        return float(value) if value is not None else None


def _to_um(column: Any) -> np.ndarray:
    import astropy.units as u

    unit = column.unit or u.um
    return (np.asarray(column, dtype=float) * unit).to_value(u.um)


def _to_jy(column: Any, wavelength_um: np.ndarray) -> np.ndarray:
    import astropy.units as u

    if column.unit is None:
        raise ValueError(f"column {column.name} has no unit (give F_nu, e.g. Jy or mJy)")
    values = np.asarray(column, dtype=float) * column.unit
    return values.to_value(u.Jy, equivalencies=u.spectral_density(wavelength_um * u.um))


def read_archival(path: Path | str) -> ArchivalData:
    """Read and check one archival ECSV file (:mod:`jwstflow.spectra.archival`)."""
    from astropy.table import Table

    path = Path(path)
    tab = Table.read(path, format="ascii.ecsv")
    kind = str(tab.meta.get("kind") or ("photometry" if "UPPER_LIMIT" in tab.colnames else "spectrum"))
    if kind not in KINDS:
        raise ValueError(f"{path.name}: meta kind {kind!r} is not one of {KINDS}")
    needed = PHOTOMETRY_COLUMNS if kind == "photometry" else ("WAVELENGTH", "FLUX")
    missing = [c for c in needed if c not in tab.colnames]
    if missing:
        raise ValueError(f"{path.name}: {kind} needs the columns {list(needed)}; missing {missing}")
    n = len(tab)
    wave = _to_um(tab["WAVELENGTH"])
    flux = _to_jy(tab["FLUX"], wave)
    error = _to_jy(tab["FLUX_ERROR"], wave) if "FLUX_ERROR" in tab.colnames else np.full(n, np.nan)
    upper = np.asarray(tab["UPPER_LIMIT"], dtype=bool) if "UPPER_LIMIT" in tab.colnames else np.zeros(n, bool)
    if upper.any() and tab.meta.get("upper_limit_sigma") is None:
        raise ValueError(f"{path.name}: upper limits need meta upper_limit_sigma (e.g. 3)")
    bandwidth = _to_um(tab["BANDWIDTH"]) if "BANDWIDTH" in tab.colnames else np.full(n, np.nan)

    def text(name: str, default: str = "") -> np.ndarray:
        return np.asarray(tab[name], dtype=str) if name in tab.colnames else np.full(n, default, dtype=object)

    return ArchivalData(path=path, kind=kind, wavelength=wave, flux=flux, error=error, upper_limit=upper,
                        bandwidth=bandwidth, reference=text("REFERENCE", str(tab.meta.get("reference", ""))),
                        citation=text("CITATION", str(tab.meta.get("citation", ""))), note=text("NOTE"),
                        meta=dict(tab.meta))


def find_archival(target_dir: Path | str, patterns: list[str] | tuple[str, ...] | str = (f"{ARCHIVAL_DIR}/*.ecsv",)
                  ) -> list[Path]:
    """The archival files matching ``patterns`` (relative to ``target_dir`` unless absolute), sorted."""
    patterns = [patterns] if isinstance(patterns, str) else list(patterns)
    found: set[Path] = set()
    for pattern in patterns:
        base = Path(pattern) if Path(pattern).is_absolute() else Path(target_dir) / pattern
        found.update(p for p in base.parent.glob(base.name) if p.is_file())
    return sorted(found)


def photometry_table(wavelength_um: Any, flux_mjy: Any, error_mjy: Any, *, upper_limit: Any = None,
                     bandwidth_um: Any = None, reference: Any = None, citation: Any = None, note: Any = None,
                     meta: dict[str, Any] | None = None, upper_limit_sigma: float | None = None) -> Any:
    """A photometry table in the archival layout (fluxes given in mJy, stored in Jy).

    ``meta`` should carry ``target``, ``description`` and the provenance
    (``source_file``, ``sources``); ``upper_limit_sigma`` is required when
    any row is an upper limit.
    """
    import astropy.units as u
    from astropy.table import Table

    n = len(np.atleast_1d(wavelength_um))

    def column(values: Any, default: Any, dtype: Any) -> np.ndarray:
        return np.full(n, default, dtype=dtype) if values is None else np.asarray(values, dtype=dtype)

    upper = column(upper_limit, False, bool)
    tab = Table()
    tab["WAVELENGTH"] = np.asarray(wavelength_um, dtype=float) * u.um
    tab["BANDWIDTH"] = column(bandwidth_um, np.nan, float) * u.um
    tab["FLUX"] = (np.asarray(flux_mjy, dtype=float) * u.mJy).to(u.Jy)
    tab["FLUX_ERROR"] = (np.asarray(error_mjy, dtype=float) * u.mJy).to(u.Jy)
    tab["UPPER_LIMIT"] = upper
    tab["REFERENCE"] = column(reference, "", str)
    tab["CITATION"] = column(citation, "", str)
    tab["NOTE"] = column(note, "", str)
    tab.meta.update({"kind": "photometry", **(meta or {})})
    if upper.any():
        if upper_limit_sigma is None:
            raise ValueError("upper limits need upper_limit_sigma")
        tab.meta["upper_limit_sigma"] = float(upper_limit_sigma)
    return tab


__all__ = ["ARCHIVAL_DIR", "ArchivalData", "find_archival", "photometry_table", "read_archival"]
