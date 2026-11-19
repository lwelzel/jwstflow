"""The test: each pixel next to the gap against the other exposures at its sky position and wavelength."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .coverage import CoverageGrid


def cross_exposure_deviation(samples: list[dict[str, Any]], tested: list[np.ndarray], grid: CoverageGrid, *,
                             footprint: float = 0.075, dispersion: float, rel_tol: float = 0.2,
                             neighbours: int = 64, chunk: int = 50_000) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each tested pixel against the other exposures at its sky position and wavelength.

    Every other exposure contributes one value: its pixels within
    ``footprint`` arcsec on the sky and one ``dispersion`` step in wavelength,
    weighted by a tent in both (in wavelength that is the linear
    interpolation between the two pixels bracketing the tested one). The
    reference is the median over the exposures; the deviation

        z = (value - ref) / sqrt(err**2 + err_ref**2 + (rel_tol * ref)**2)

    with ``err_ref`` = 1.2533 x the median error of the exposures / sqrt(n).
    Returns per cal file ``(z, n)``: NaN and 0 for pixels not tested.
    """
    from scipy.spatial import cKDTree

    exposures = sorted({s["exposure"] for s in samples})
    cols = {k: [] for k in ("east", "north", "wave", "value", "err", "exp")}
    for s in samples:
        east, north = CoverageGrid._offsets(s["ra"], s["dec"], grid.ra0, grid.dec0)
        cols["east"].append(east)
        cols["north"].append(north)
        for k in ("wave", "value", "err"):
            cols[k].append(np.asarray(s[k], float))
        cols["exp"].append(np.full(s["wave"].size, exposures.index(s["exposure"])))
    a = {k: np.concatenate(v) if v else np.empty(0) for k, v in cols.items()}
    a["exp"] = a["exp"].astype(int)
    usable = np.isfinite(a["value"]) & np.isfinite(a["err"]) & (a["err"] > 0)
    pts = np.column_stack([a["east"] / footprint, a["north"] / footprint, a["wave"] / dispersion])
    out = [(np.full(s["wave"].size, np.nan), np.zeros(s["wave"].size, dtype=int)) for s in samples]
    if not usable.any():
        return out
    idx_usable = np.flatnonzero(usable)
    tree = cKDTree(pts[usable])
    n_exp = len(exposures)
    offsets = np.cumsum([0] + [s["wave"].size for s in samples])
    for f, (s, sel) in enumerate(zip(samples, tested, strict=True)):
        which = offsets[f] + np.flatnonzero(sel & usable[offsets[f]:offsets[f + 1]])
        for lo in range(0, which.size, chunk):
            q = which[lo:lo + chunk]
            dist, nb = tree.query(pts[q], k=min(neighbours, idx_usable.size), distance_upper_bound=1.5)
            dist, nb = np.atleast_2d(dist), np.atleast_2d(nb)
            valid = np.isfinite(dist)
            g = idx_usable[np.where(valid, nb, 0)]
            dsp = np.hypot(a["east"][g] - a["east"][q, None], a["north"][g] - a["north"][q, None]) / footprint
            dwl = np.abs(a["wave"][g] - a["wave"][q, None]) / dispersion
            w = np.clip(1.0 - dsp, 0, None) * np.clip(1.0 - dwl, 0, None) * valid * (a["exp"][g] != a["exp"][q, None])
            flat = (np.arange(q.size)[:, None] * n_exp + a["exp"][g]).ravel()
            size = q.size * n_exp
            sw = np.bincount(flat, w.ravel(), minlength=size).reshape(q.size, n_exp)
            swv = np.bincount(flat, (w * a["value"][g]).ravel(), minlength=size).reshape(q.size, n_exp)
            swe = np.bincount(flat, (w * a["err"][g]).ravel(), minlength=size).reshape(q.size, n_exp)
            have = sw > 0.25
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.filterwarnings("ignore", message=".*All-NaN.*")
                vals = np.where(have, swv / np.where(have, sw, 1.0), np.nan)
                errs = np.where(have, swe / np.where(have, sw, 1.0), np.nan)
                n = have.sum(axis=1)
                ref = np.nanmedian(vals, axis=1)
                err_ref = 1.2533 * np.nanmedian(errs, axis=1) / np.sqrt(np.clip(n, 1, None))
                z = (a["value"][q] - ref) / np.sqrt(a["err"][q] ** 2 + err_ref ** 2 + (rel_tol * ref) ** 2)
            local = q - offsets[f]
            out[f][0][local] = np.where(n > 0, z, np.nan)
            out[f][1][local] = n
    return out
