"""The comparison of one dither cube with the others: what deviates, and by how much."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..wcs import resample_image


def robust_sigma(values: np.ndarray) -> float:
    """1.4826 x MAD of the finite values (0.0 when there are none)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def deviation_from_others(plane: np.ndarray, wave_k: float, this_wcs: Any, others: list[dict[str, Any]],
                          ) -> tuple[np.ndarray, np.ndarray, int]:
    """Compare one cube plane with the other dithers at the same wavelength.

    Each other cube's nearest plane (within one plane step) is resampled onto
    this cube's grid. Returns ``(dev, ref, n)``: the *signed deviation with the
    smallest magnitude* over the others (a bad dither deviates from every
    other one; a good dither agrees with at least one), their nanmedian as a
    reference image, and how many others contributed. ``n == 0`` when no
    other dither covers the wavelength.
    """
    import warnings

    ny, nx = plane.shape
    stack = []
    for other in others:
        wave = other["wave"]
        k = int(np.argmin(np.abs(wave - wave_k)))
        step = float(np.median(np.abs(np.diff(wave)))) if wave.size > 1 else np.inf
        if abs(wave[k] - wave_k) > step:
            continue
        stack.append(resample_image(other["data"][k], other["wcs"], this_wcs, (ny, nx)))
    if not stack:
        nan = np.full((ny, nx), np.nan)
        return nan, nan, 0
    imgs = np.stack(stack)
    devs = plane[None] - imgs
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*All-NaN.*")
        absdev = np.where(np.isfinite(devs), np.abs(devs), np.inf)
        idx = np.argmin(absdev, axis=0)
        dev = np.take_along_axis(devs, idx[None], axis=0)[0]
        ref = np.nanmedian(imgs, axis=0)
    return dev, ref, len(stack)
