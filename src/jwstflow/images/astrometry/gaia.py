"""Gaia DR3 for a field: one query for the footprint of a run, its cache, and positions at an epoch.

:func:`query_gaia` asks the services in order until one answers:

* ``esa`` -- the ESA Gaia archive (``gaiadr3.gaia_source``) through ``astroquery.gaia``;
* ``stsci`` -- the STScI catalogue service that ``tweakreg``'s ``abs_refcat: GAIADR3`` reads
  (``stcal.tweakreg.astrometric_utils.get_catalog``), asked for the catalogue as published
  (no epoch: every source, positions at J2016.0).

Either answer is normalised to the same small table (:data:`COLUMNS`, with units; blanks are NaN),
which :func:`load_or_query_gaia` writes as ``<base>_gaiadr3.ecsv`` with the query in its metadata
(service, centre, radius, time) and reuses while it covers the footprint asked for.
:func:`propagate` moves the positions from the Gaia epoch to another with the proper motions,
linearly in the tangent plane. Parallax is ignored, as ``tweakreg``'s ``abs_refcat`` ignores it:
below a mas for the background stars that make most of a field, but 5-8 mas for members of the
nearby star-forming regions (Taurus, Ophiuchus, Chamaeleon) -- comparable to the errors of a
median over a few Gaia stars.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

GAIA_SUFFIX = "gaiadr3"     # <base>_gaiadr3.ecsv: the Gaia DR3 sources of a run's footprint
#: The reference epoch of Gaia DR3 positions [Julian year].
GAIA_EPOCH = 2016.0
GAIA_NAME = "Gaia DR3"
#: Services asked, in this order, by default.
SERVICES: tuple[str, ...] = ("esa", "stsci")
#: Columns of the normalised table and their units.
COLUMNS: dict[str, str | None] = {
    "source_id": None, "ra": "deg", "dec": "deg", "ra_error": "mas", "dec_error": "mas", "pmra": "mas / yr",
    "pmdec": "mas / yr", "pmra_error": "mas / yr", "pmdec_error": "mas / yr", "parallax": "mas",
    "phot_g_mean_mag": "mag", "ref_epoch": "yr", "astrometric_params_solved": None,
}
#: Seconds a service may take to answer.
TIMEOUT = 120.0


class GaiaQueryError(RuntimeError):
    """No service answered."""


def query_gaia(ra: float, dec: float, radius: float, *, services: tuple[str, ...] | list[str] = SERVICES) -> tuple[Any, str]:
    """The Gaia DR3 sources within ``radius`` [deg] of (``ra``, ``dec``) [deg] from the first of
    ``services`` that answers: ``(table, service)``. Raises :class:`GaiaQueryError` listing why each failed."""
    failures: list[str] = []
    for service in services:
        try:
            query = {"esa": _query_esa, "stsci": _query_stsci}[service]
        except KeyError:
            failures.append(f"{service}: unknown service (known: esa, stsci)")
            continue
        try:
            table = normalise(query(ra, dec, radius))
        except Exception as exc:  # noqa: BLE001 -- network, server or parsing: try the next service
            failures.append(f"{service}: {type(exc).__name__}: {str(exc).strip()[:200]}")
            log.warning("Gaia DR3 from %s failed: %s", service, failures[-1])
            continue
        log.info("Gaia DR3 from %s: %d source(s) within %.2f arcmin of (%.6f, %.6f)", service, len(table), radius * 60.0,
                 ra, dec)
        return table, service
    raise GaiaQueryError("; ".join(failures) or "no service to ask")


def _query_esa(ra: float, dec: float, radius: float) -> Any:
    from astroquery.gaia import Gaia

    columns = ", ".join(COLUMNS)
    adql = (f"SELECT {columns} FROM gaiadr3.gaia_source WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {ra:.8f}, {dec:.8f}, {radius:.8f}))")
    return Gaia.launch_job_async(adql, verbose=False).get_results()


def _query_stsci(ra: float, dec: float, radius: float) -> Any:
    from stcal.tweakreg.astrometric_utils import get_catalog

    return get_catalog(ra, dec, epoch=None, search_radius=radius, catalog="GAIADR3", timeout=TIMEOUT)


def normalise(table: Any) -> Any:
    """The :data:`COLUMNS` of a Gaia DR3 table (any service; column names case-insensitive) as plain
    floats with units, blanks NaN; ``source_id`` int64, ``astrometric_params_solved`` int (0 unknown)."""
    from astropy.table import Table

    names = {c.lower(): c for c in table.colnames}
    missing = [c for c in ("source_id", "ra", "dec") if c not in names]
    if missing:
        raise KeyError(f"the Gaia table has no {missing} column(s): {table.colnames[:12]}...")
    out = Table()
    n = len(table)
    for col, unit in COLUMNS.items():
        values = table[names[col]] if col in names else None
        if col == "source_id":
            out[col] = np.asarray(values, dtype=np.int64)
        elif col == "astrometric_params_solved":
            out[col] = np.zeros(n, np.int16) if values is None else np.asarray(_filled(values, 0), dtype=np.int16)
        else:
            data = np.full(n, np.nan) if values is None else _floats(values)
            if col == "ref_epoch":
                data = np.where(np.isfinite(data), data, GAIA_EPOCH)
            out[col] = data
            if unit:
                out[col].unit = unit
    return out


def load_or_query_gaia(cache: Path, ra: float, dec: float, radius: float, *,
                       services: tuple[str, ...] | list[str] = SERVICES) -> tuple[Any, dict[str, Any]]:
    """Gaia DR3 for a circle [deg]: from ``cache`` (a ``*_gaiadr3.ecsv``) when its query covers the
    circle, else from :func:`query_gaia`, written to ``cache``. Returns ``(table, query)`` -- the
    query as recorded in the cache (``service``, ``ra``, ``dec``, ``radius``, ``queried``) plus
    ``reused``. Raises :class:`GaiaQueryError` when no service answers."""
    from astropy.table import Table

    cache = Path(cache)
    if cache.is_file():
        try:
            table = Table.read(cache, format="ascii.ecsv")
            meta = dict(table.meta.get("gaia_query", {}))
            if covers(meta, ra, dec, radius):
                log.info("Gaia DR3 from the cache %s (%s, %d sources)", cache.name, meta.get("service"), len(table))
                return table, {**meta, "reused": True}
            log.info("the Gaia cache %s does not cover the footprint; querying again", cache.name)
        except Exception as exc:  # noqa: BLE001 -- an unreadable cache is replaced
            log.warning("the Gaia cache %s is unreadable (%s); querying again", cache.name, exc)
    table, service = query_gaia(ra, dec, radius, services=services)
    meta = {"service": service, "catalog": GAIA_NAME, "ra": float(ra), "dec": float(dec), "radius": float(radius),
            "epoch": GAIA_EPOCH, "queried": datetime.now(UTC).isoformat(timespec="seconds"),
            "units": "ra, dec, radius [deg]; positions at ref_epoch [yr]"}
    table.meta["gaia_query"] = meta
    table.write(cache, format="ascii.ecsv", overwrite=True)
    return table, {**meta, "reused": False}


def covers(meta: dict[str, Any], ra: float, dec: float, radius: float) -> bool:
    """Whether a recorded query (``ra``, ``dec``, ``radius`` [deg]) contains the circle asked for."""
    try:
        ra0, dec0, r0 = float(meta["ra"]), float(meta["dec"]), float(meta["radius"])
    except (KeyError, TypeError, ValueError):
        return False
    return angular_distance(ra0, dec0, ra, dec) + radius <= r0 * (1.0 + 1e-9)


def angular_distance(ra1: Any, dec1: Any, ra2: Any, dec2: Any) -> Any:
    """Great-circle distance [deg] (haversine)."""
    a1, d1, a2, d2 = (np.radians(np.asarray(v, float)) for v in (ra1, dec1, ra2, dec2))
    h = np.sin(0.5 * (d2 - d1)) ** 2 + np.cos(d1) * np.cos(d2) * np.sin(0.5 * (a2 - a1)) ** 2
    return np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))))


def propagate(table: Any, epoch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions of the Gaia sources at ``epoch`` [Julian year]: ``(ra, dec, has_pm)`` [deg]. A source
    with a proper motion moves by it over ``epoch - ref_epoch``, linearly in the tangent plane; a
    source without one (two-parameter solution) stays where it was at ``ref_epoch``."""
    ra, dec = _floats(table["ra"]), _floats(table["dec"])
    pmra, pmdec = _floats(table["pmra"]), _floats(table["pmdec"])
    ref_epoch = _floats(table["ref_epoch"]) if "ref_epoch" in table.colnames else np.full(ra.size, GAIA_EPOCH)
    has_pm = np.isfinite(pmra) & np.isfinite(pmdec)
    if not math.isfinite(epoch):
        return ra, dec, has_pm
    dt = epoch - np.where(np.isfinite(ref_epoch), ref_epoch, GAIA_EPOCH)
    shift_ra = np.where(has_pm, pmra * dt, 0.0) / 3.6e6 / np.cos(np.radians(dec))
    shift_dec = np.where(has_pm, pmdec * dt, 0.0) / 3.6e6
    return (ra + shift_ra) % 360.0, dec + shift_dec, has_pm


def julian_year(mjd: float) -> float:
    """Julian epoch [yr] of an MJD (J2000.0 = MJD 51544.5)."""
    return 2000.0 + (float(mjd) - 51544.5) / 365.25


def _floats(values: Any) -> np.ndarray:
    values = _filled(getattr(values, "value", values), np.nan)
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError):      # a text column with blanks (a CSV answer)
        return np.array([_float(v) for v in values], dtype=float)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _filled(values: Any, fill: Any) -> Any:
    if hasattr(values, "filled"):
        try:
            return values.filled(fill)
        except (TypeError, ValueError):
            pass
    return values
