"""Image operations cube steps share: NaN-aware smoothing, dilation, reducing a mask to one clean blob."""

from __future__ import annotations

import numpy as np


def nan_smooth(img: np.ndarray, sigma: float) -> np.ndarray:
    """NaN-aware Gaussian smoothing (weights = finite pixels)."""
    from scipy.ndimage import gaussian_filter

    if sigma <= 0:
        return img
    finite = np.isfinite(img)
    v = gaussian_filter(np.where(finite, img, 0.0), sigma)
    w = gaussian_filter(finite.astype(float), sigma)
    return np.where(w > 0.05, v / np.maximum(w, 1e-12), np.nan)


def dilate(mask: np.ndarray, n: int) -> np.ndarray:
    """Binary dilation by ``n`` pixels (``n <= 0`` returns the mask unchanged)."""
    from scipy.ndimage import binary_dilation

    return binary_dilation(mask, iterations=n) if n > 0 else mask


def clean_blob(mask: np.ndarray, anchor: np.ndarray | None = None) -> np.ndarray:
    """Close small gaps, fill holes, and keep only structure connected to `anchor`
    (or, without an anchor, the largest connected component)."""
    from scipy.ndimage import binary_closing, binary_fill_holes, label

    if not mask.any():
        return mask
    m = binary_fill_holes(binary_closing(mask, structure=np.ones((3, 3), bool)))
    labels, n = label(m, structure=np.ones((3, 3), int))
    if n <= 1:
        return m
    if anchor is not None and anchor.any():
        keep = np.unique(labels[anchor & (labels > 0)])
    else:
        keep = [int(np.argmax(np.bincount(labels[labels > 0].ravel())))]
    return np.isin(labels, [k for k in keep if k > 0])
