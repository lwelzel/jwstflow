"""Where is the target? Positions come from, in order of preference:

1. explicit coordinates in the workflow (``target_coords`` or a source entry with ra/dec);
2. the observation itself (``TARG_RA``/``TARG_DEC`` of any input file);
3. the name, resolved through Sesame (SIMBAD/NED/VizieR) and cached in the
   target directory so the network is asked once per target.

No positions are ever read from ad-hoc text files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CACHE_NAME = "targets.json"


def parse_coords(ra: Any, dec: Any) -> tuple[float, float]:
    """Degrees from degrees or sexagesimal strings."""
    from astropy.coordinates import SkyCoord

    ra_s, dec_s = str(ra), str(dec)
    unit = ("hourangle", "deg") if ":" in ra_s or "h" in ra_s.lower() else ("deg", "deg")
    c = SkyCoord(ra_s, dec_s, unit=unit, frame="icrs")
    return float(c.ra.deg), float(c.dec.deg)


def resolve_name(name: str, cache_dir: Path | None = None) -> tuple[float, float]:
    """ICRS position of a catalogue object by name (Sesame), cached per target directory."""
    cache = _load_cache(cache_dir)
    key = name.strip().lower()
    if key in cache:
        return float(cache[key]["ra"]), float(cache[key]["dec"])
    from astropy.coordinates import SkyCoord

    try:
        c = SkyCoord.from_name(name)
    except Exception as exc:  # NameResolveError or network problems
        raise RuntimeError(f"could not resolve {name!r} by name: {exc}. Give target_coords: {{ra, dec}} in the workflow") from exc
    cache[key] = {"ra": float(c.ra.deg), "dec": float(c.dec.deg), "resolver": "sesame"}
    _save_cache(cache_dir, cache)
    log.info("resolved %r to RA=%.6f Dec=%.6f (Sesame)", name, c.ra.deg, c.dec.deg)
    return cache[key]["ra"], cache[key]["dec"]


def resolve_source(entry: Any, *, header: Any = None, default_name: str | None = None,
                   default_coords: dict[str, Any] | None = None, cache_dir: Path | None = None) -> tuple[str, float, float]:
    """One source specification -> ``(name, ra_deg, dec_deg)``.

    ``entry`` may be ``"target"`` (the observation's target: explicit workflow
    coordinates, else the header's TARG_RA/TARG_DEC, else the target name),
    a plain name (resolved), or ``{name, ra, dec}`` (explicit position).
    """
    if isinstance(entry, dict):
        name = str(entry.get("name") or default_name or "source")
        if "ra" in entry and "dec" in entry:
            ra, dec = parse_coords(entry["ra"], entry["dec"])
            return name, ra, dec
        ra, dec = resolve_name(name, cache_dir)
        return name, ra, dec
    label = str(entry)
    if label == "target":
        name = default_name or (str(header.get("TARGPROP", "target")) if header is not None else "target")
        if default_coords and "ra" in default_coords and "dec" in default_coords:
            ra, dec = parse_coords(default_coords["ra"], default_coords["dec"])
            return name, ra, dec
        if header is not None and header.get("TARG_RA") is not None and header.get("TARG_DEC") is not None:
            return name, float(header["TARG_RA"]), float(header["TARG_DEC"])
        ra, dec = resolve_name(name, cache_dir)
        return name, ra, dec
    ra, dec = resolve_name(label, cache_dir)
    return label, ra, dec


def _load_cache(cache_dir: Path | None) -> dict[str, Any]:
    if cache_dir is None:
        return {}
    path = Path(cache_dir) / CACHE_NAME
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache_dir: Path | None, cache: dict[str, Any]) -> None:
    if cache_dir is None:
        return
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    (Path(cache_dir) / CACHE_NAME).write_text(json.dumps(cache, indent=2))
