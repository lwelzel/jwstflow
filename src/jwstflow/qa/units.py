"""Units of the plotted quantities: everything flux-like is shown in mJy."""

from __future__ import annotations

import numpy as np

from .style import FLUX_LABEL, SB_LABEL

#: 1 MJy/sr expressed in mJy per arcsec^2.
MJY_ARCSEC2_PER_MJY_SR = 1e9 / (np.degrees(1.0) * 3600.0) ** 2


# --------------------------------------------------------------------------- units (rule 2)
def to_mjy(values: np.ndarray, unit: str | None) -> tuple[np.ndarray, str]:
    """Convert flux-like ``values`` to mJy (surface brightness to mJy arcsec^-2).

    ``unit`` is the unit the values carry (a FITS ``TUNIT``/``BUNIT`` string;
    case matters: ``mJy`` vs ``MJy``). Returns ``(converted, axis label)``,
    the label with its unit in square brackets. Unknown units come back
    unchanged, labelled with whatever the product said.
    """
    values = np.asarray(values, dtype=float)
    key = (unit or "").strip().replace(" ", "")
    factors = {"Jy": 1e3, "mJy": 1.0, "uJy": 1e-3, "µJy": 1e-3, "nJy": 1e-6, "MJy": 1e9}
    if key in factors:
        return values * factors[key], FLUX_LABEL
    if key in ("MJy/sr", "MJy.sr-1", "MJysr-1"):
        return values * MJY_ARCSEC2_PER_MJY_SR, SB_LABEL
    if key in ("mJy/arcsec2", "mJy/arcsec^2"):
        return values, SB_LABEL
    if not key:
        return values, ""
    return values, f"[{unit}]"


__all__ = ["MJY_ARCSEC2_PER_MJY_SR", "to_mjy"]
