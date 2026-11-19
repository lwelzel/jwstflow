"""Which exposures are combined into one cube, and what to call the group."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ...steps.naming import slug

#: Default exposure grouping: the exposures one ``calwebb_spec3`` cube is built from (the presets'
#: spec3 ``group_by``), science and background apart.
DEFAULT_GROUP_BY: tuple[str, ...] = ("PROGRAM", "OBSERVTN", "GRATING", "FILTER", "BKGDTARG")


def exposure_of(path: str | Path) -> str:
    """The exposure a cal file belongs to: its name without detector and suffix
    (``jw05299004001_03101_00001_nrs1_cal.fits`` -> ``jw05299004001_03101_00001``)."""
    name = Path(path).name
    m = re.match(r"(.+?)_(?:nrs[12]|mirifu\w*|mirimage)(?:_|\.)", name, flags=re.IGNORECASE)
    return m.group(1) if m else Path(path).stem


def group_key(header: Any, keys: list[str] | tuple[str, ...]) -> tuple[Any, ...]:
    """The values of ``keys`` in a primary header, normalised (strings lower-cased)."""
    out = []
    for key in keys:
        val = header.get(str(key).upper())
        out.append(str(val).strip().lower() if isinstance(val, str) else val)
    return tuple(out)


def group_name(header: Any, keys: list[str] | tuple[str, ...]) -> str:
    """A level-3-like base name for a group of exposures, e.g. ``jw05299-o004_nirspec_g235h-f170lp``
    (other grouping keywords are appended; ``_bkg`` for background exposures)."""
    keys = [str(k).upper() for k in keys]
    parts = []
    prog = header.get("PROGRAM")
    head = f"jw{str(prog).strip()}" if prog is not None else "jw"
    if "OBSERVTN" in keys and header.get("OBSERVTN") is not None:
        head += f"-o{str(header['OBSERVTN']).strip()}"
    parts.append(head)
    parts.append(str(header.get("INSTRUME", "nirspec")).strip().lower())
    optic = "-".join(str(header[k]).strip().lower() for k in ("GRATING", "FILTER") if k in keys and header.get(k))
    if optic:
        parts.append(optic)
    for k in keys:
        if k in ("PROGRAM", "OBSERVTN", "GRATING", "FILTER", "BKGDTARG", "INSTRUME"):
            continue
        if header.get(k) is not None:
            parts.append(slug(f"{k.lower()}{str(header[k]).strip().lower()}"))
    if "BKGDTARG" in keys and bool(header.get("BKGDTARG")):
        parts.append("bkg")
    return "_".join(p for p in parts if p)
