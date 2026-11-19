"""Which instrument configuration a cube -- or a product derived from one -- belongs to."""

from __future__ import annotations

from typing import Any

#: Header keywords that identify a cube's instrument configuration; a
#: product matches a cube when every keyword present in both headers agrees.
INSTRUMENT_KEYS = ("INSTRUME", "DETECTOR", "GRATING", "FILTER", "CHANNEL", "BAND")


def matches_instrument(header_a: Any, header_b: Any) -> bool:
    """True when every :data:`INSTRUMENT_KEYS` keyword present in both headers agrees."""
    for key in INSTRUMENT_KEYS:
        a, b = header_a.get(key), header_b.get(key)
        if a is not None and b is not None and str(a).strip().lower() != str(b).strip().lower():
            return False
    return True
