"""A bright neighbour's PSF in an IFU cube: where it is, how bright it is plane by plane, what is left.

The fitting core of ``subtract_psf_neighbour`` (:mod:`.steps`), on plain
arrays: no files, no step. A cube holds a faint target and, near it, a point
source far brighter than the target (HH 48 SW next to the edge-on disk
HH 48 NE: 10-100x brighter in the mid-infrared, 2.3" away). Its PSF -- the
library of ``psf_cube`` placed at its position on the cube's grid
(:meth:`PsfCubeProduct.placed`) -- is fitted to every plane over the field
minus a protected region around the target::

    data(x, y) = a * P(x - x0, y - y0) + c [+ gx (x - xt) + gy (y - yt)]

``a`` is the neighbour's total flux in the plane (in cube units times one
spaxel: the library slices are normalised to all of the light), ``c`` a
constant and the optional plane a gradient through the target position --
both background, fitted alongside, never subtracted. Weights are
``1 / (ERR^2 + (f a P)^2)``: ``f`` (``model_error``) is the fractional
uncertainty of the PSF model, which keeps the core of the neighbour -- where
a model is least exact and the data have most signal -- from setting the
amplitude alone; residuals beyond ``clip`` sigma are left out and the fit is
repeated. A neighbour whose core is outside the field is fitted from the
part of its wing that is inside.

The position comes first: the cube's position of the neighbour, refined --
when its core is in the field -- by a centroid and then a PSF fit of x0, y0
on block-stacked images (:func:`refine_position`).

Diagnostics: the residual over the fit region, and the **control ring** --
the annulus around the neighbour at the target's distance, minus the
target's protected region: what the subtraction leaves there is what it
leaves at the target, and it must be consistent with zero
(:func:`ring_residual`). :func:`radial_profile` gives the azimuthal mean
around the neighbour before and after.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .product import PsfCubeProduct

log = logging.getLogger(__name__)

#: Background terms the fit can carry alongside the PSF (never subtracted).
BACKGROUND_TERMS = {"none": 0, "constant": 1, "plane": 3}


# --------------------------------------------------------------------------- models per plane
def interpolation_nodes(nodes_um: np.ndarray, wave_um: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every plane the two library slices that bracket its wavelength and the weight of the upper one
    (clamped at the ends of the library)."""
    nodes = np.asarray(nodes_um, dtype=float)
    wave = np.asarray(wave_um, dtype=float)
    if len(nodes) == 1:
        zero = np.zeros(len(wave), dtype=int)
        return zero, zero, np.zeros(len(wave))
    upper = np.clip(np.searchsorted(nodes, wave), 1, len(nodes) - 1)
    lower = upper - 1
    share = np.clip((wave - nodes[lower]) / (nodes[upper] - nodes[lower]), 0.0, 1.0)
    return lower, upper, share


def plane_models(placed: np.ndarray, nodes_um: np.ndarray, wave_um: np.ndarray) -> np.ndarray:
    """(nw, ny, nx): the placed library slices interpolated linearly to every plane's wavelength."""
    lower, upper, share = interpolation_nodes(nodes_um, wave_um)
    return ((1.0 - share)[:, None, None] * placed[lower] + share[:, None, None] * placed[upper]).astype("f8")


# --------------------------------------------------------------------------- the fit
@dataclass
class PlaneFit:
    """The fit of every plane (arrays of length nw; NaN where a plane could not be fitted).

    ``amplitudes`` holds one column per model component: the PSF first, then
    any broadened copies (the halo terms); each is a total flux [cube units
    x spaxel], because every component is normalised to all of the light.
    """

    amplitudes: np.ndarray     #: (nw, ncomp) total flux in each model component
    errors: np.ndarray         #: (nw, ncomp) their 1-sigma errors, scaled by sqrt(reduced chi2) where that exceeds 1
    const: np.ndarray          #: background constant [cube units] (zero without one)
    gradient: np.ndarray       #: (nw, 2) background gradient along x, y [cube units / spaxel] (zero without a plane)
    chi2: np.ndarray           #: chi2 over the spaxels fitted
    nfit: np.ndarray           #: spaxels fitted (after clipping)
    nclip: np.ndarray          #: spaxels clipped
    nparams: int               #: free parameters per plane
    origin: tuple[float, float]  #: (x, y) the gradient is measured from: the target
    total: np.ndarray = None   #: the neighbour's total flux: the sum over the components
    total_err: np.ndarray = None  #: its error (with the covariances of the components)

    @property
    def amplitude(self) -> np.ndarray:
        """The PSF component alone."""
        return self.amplitudes[:, 0]

    @property
    def reduced_chi2(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.chi2 / np.maximum(self.nfit - self.nparams, 1)

    def model(self, components: list[np.ndarray]) -> np.ndarray:
        """(nw, ny, nx): the neighbour -- the sum of the components at their fitted amplitudes."""
        amps = np.nan_to_num(self.amplitudes)
        return sum(amps[:, i, None, None] * c for i, c in enumerate(components))

    def background(self, shape: tuple[int, int]) -> np.ndarray:
        """(nw, ny, nx): the fitted background terms (constant and plane)."""
        yy, xx = np.indices(shape, dtype=float)
        x0, y0 = self.origin
        c = np.nan_to_num(self.const)[:, None, None]
        g = np.nan_to_num(self.gradient)
        return c + g[:, 0, None, None] * (xx - x0) + g[:, 1, None, None] * (yy - y0)


def fit_planes(data: np.ndarray, err: np.ndarray, components: np.ndarray | list[np.ndarray], fit_mask: np.ndarray, *,
               background: str = "constant", origin: tuple[float, float] = (0.0, 0.0), model_error: float = 0.0,
               clip: float = 5.0, iterations: int = 3, min_pixels: int = 10, chunk: int = 256) -> PlaneFit:
    """Fit ``sum_i a_i * component_i + background`` to every plane over ``fit_mask`` (ny, nx), weighted by ERR.

    ``components`` is one (nw, ny, nx) model -- the PSF placed at the
    neighbour -- or a list of them (the PSF, then its broadened copies or
    the radial terms of :func:`radial_tents`; a (1, ny, nx) component is the
    same in every plane). Each is normalised to unit flux, so the sum of
    the amplitudes is the neighbour's flux.
    All planes are solved at once through their normal equations. Each
    iteration re-weights with the model error ``model_error * model`` added
    to ERR in quadrature and leaves out the spaxels more than ``clip`` sigma
    off the previous fit (never more than a fifth of a plane's). Planes are
    solved ``chunk`` at a time.
    """
    if background not in BACKGROUND_TERMS:
        raise ValueError(f"background must be one of {sorted(BACKGROUND_TERMS)}, not {background!r}")
    comps = list(components) if isinstance(components, (list, tuple)) else [components]
    nw = len(data)
    if nw > chunk:   # bounded memory: the planes are independent
        parts = [fit_planes(data[a:a + chunk], err[a:a + chunk], [c if len(c) == 1 else c[a:a + chunk] for c in comps],
                            fit_mask,
                            background=background, origin=origin, model_error=model_error, clip=clip,
                            iterations=iterations, min_pixels=min_pixels, chunk=chunk)
                 for a in range(0, nw, chunk)]
        return PlaneFit(**{name: np.concatenate([getattr(q, name) for q in parts])
                           for name in ("amplitudes", "errors", "const", "gradient", "chi2", "nfit", "nclip",
                                        "total", "total_err")},
                        nparams=parts[0].nparams, origin=parts[0].origin)
    comps = [np.asarray(c, dtype="f8") for c in comps]
    d = np.asarray(data, dtype="f8")
    e = np.asarray(err, dtype="f8")
    nw, ny, nx = d.shape
    usable = np.isfinite(d) & np.isfinite(e) & (e > 0) & np.asarray(fit_mask, bool)[None]
    for c in comps:
        usable &= np.isfinite(c)
    yy, xx = np.indices((ny, nx), dtype=float)
    x0, y0 = origin
    columns = list(comps)
    nc = len(comps)
    if background in ("constant", "plane"):
        columns.append(np.ones((1, ny, nx)))
    if background == "plane":
        columns += [(xx - x0)[None], (yy - y0)[None]]
    p = len(columns)
    d0 = np.where(usable, d, 0.0)
    var_base = np.where(usable, e, 1.0) ** 2
    keep = usable.copy()
    theta = np.full((nw, p), np.nan)
    cov = np.full((nw, p, p), np.nan)
    neighbour = np.zeros_like(d0)
    n_iter = max(1, int(iterations))
    for it in range(n_iter):
        var = var_base + (model_error * neighbour) ** 2
        w = np.where(keep, 1.0 / var, 0.0)
        normal = np.empty((nw, p, p))
        rhs = np.empty((nw, p))
        for i in range(p):
            wi = w * columns[i]
            rhs[:, i] = np.einsum("kyx,kyx->k", wi, d0)
            for j in range(i, p):
                normal[:, i, j] = normal[:, j, i] = np.einsum("kyx,kyx->k", wi, np.broadcast_to(columns[j], d0.shape))
        n_ok = keep.sum(axis=(1, 2))
        with np.errstate(invalid="ignore", over="ignore"):
            good = (n_ok >= max(min_pixels, p + 1)) & np.isfinite(normal).all(axis=(1, 2))
            good[good] &= np.abs(np.linalg.det(normal[good])) > 0
        theta[:] = np.nan
        cov[:] = np.nan
        if good.any():
            cov[good] = np.linalg.pinv(normal[good])
            theta[good] = np.einsum("kij,kj->ki", cov[good], rhs[good])
        t = np.nan_to_num(theta)
        neighbour = np.abs(sum(t[:, i, None, None] * columns[i] for i in range(nc)))
        fit = sum(t[:, i, None, None] * columns[i] for i in range(p))
        if it == n_iter - 1:
            break
        sigma = np.sqrt(var_base + (model_error * neighbour) ** 2)
        new_keep = usable & (np.abs(d0 - fit) <= clip * sigma)
        # never clip more than a fifth of a plane: a model gone wrong must show in chi2, not vanish
        too_many = new_keep.sum(axis=(1, 2)) < 0.8 * usable.sum(axis=(1, 2))
        new_keep[too_many] = usable[too_many]
        keep = new_keep
    sigma = np.sqrt(var_base + (model_error * neighbour) ** 2)
    resid = np.where(keep, (d0 - fit) / sigma, 0.0)
    chi2 = (resid**2).sum(axis=(1, 2))
    nfit = keep.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.sqrt(np.maximum(chi2 / np.maximum(nfit - p, 1), 1.0))
        errors = np.sqrt(np.clip(np.diagonal(cov, axis1=1, axis2=2)[:, :nc], 0, None)) * scale[:, None]
        total_var = cov[:, :nc, :nc].sum(axis=(1, 2))
        total_err = np.sqrt(np.clip(total_var, 0, None)) * scale
    bad = ~np.isfinite(theta[:, 0])
    const = theta[:, nc] if background in ("constant", "plane") else np.zeros(nw)
    gradient = theta[:, nc + 1:nc + 3] if background == "plane" else np.zeros((nw, 2))
    amps = np.where(bad[:, None], np.nan, theta[:, :nc])
    return PlaneFit(amplitudes=amps, errors=np.where(bad[:, None], np.nan, errors),
                    const=np.where(bad, np.nan, const), gradient=np.where(bad[:, None], np.nan, gradient),
                    chi2=np.where(bad, np.nan, chi2), nfit=nfit, nclip=usable.sum(axis=(1, 2)) - nfit, nparams=p,
                    origin=(float(x0), float(y0)), total=amps.sum(axis=1), total_err=np.where(bad, np.nan, total_err))


def broadened(psf: PsfCubeProduct, sigma_arcsec: float) -> PsfCubeProduct:
    """The library convolved with a circular Gaussian of ``sigma_arcsec``: a halo component of the
    same total flux (what the PSF model misses at large radii, scattered light), placed and fitted
    like the PSF itself."""
    from scipy.ndimage import gaussian_filter

    sigma_pix = float(sigma_arcsec) / psf.pixelscale_arcsec
    data = np.stack([gaussian_filter(np.nan_to_num(plane), sigma_pix, mode="constant") for plane in psf.data])
    meta = dict(psf.meta)
    meta["JWFPSFHS"] = (float(sigma_arcsec), "[arcsec] Gaussian halo sigma")
    return type(psf)(data, psf.wavelengths.copy(), psf.pixelscale_arcsec, meta=meta, header=psf.header)


# --------------------------------------------------------------------------- the position
def block_stack(data: np.ndarray, err: np.ndarray, n_blocks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``n_blocks`` images: the nan-median of consecutive runs of planes, their error (the median ERR
    over sqrt of the planes) and the index range of each block, (n, 2)."""
    nw = len(data)
    edges = np.linspace(0, nw, max(1, int(n_blocks)) + 1).round().astype(int)
    images, errors, ranges = [], [], []
    import warnings

    for a, b in pairwise(edges):
        if b <= a:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            images.append(np.nanmedian(data[a:b], axis=0))
            n = np.isfinite(data[a:b]).sum(axis=0)
            errors.append(np.nanmedian(err[a:b], axis=0) * 1.2533 / np.sqrt(np.maximum(n, 1)))
        ranges.append((a, b))
    return np.asarray(images), np.asarray(errors), np.asarray(ranges)


def centroid(image: np.ndarray, x0: float, y0: float, radius: float, *, iterations: int = 5) -> tuple[float, float]:
    """First moment of the background-subtracted image within ``radius`` [pix] of (x0, y0), iterated
    from the given position. Returns the start when nothing positive is found there."""
    ny, nx = image.shape
    yy, xx = np.indices((ny, nx), dtype=float)
    finite = np.isfinite(image)
    if not finite.any():
        return float(x0), float(y0)
    floor = float(np.nanmedian(image))
    x, y = float(x0), float(y0)
    for _ in range(iterations):
        near = finite & (np.hypot(xx - x, yy - y) <= radius)
        w = np.where(near, np.clip(image - floor, 0.0, None), 0.0)
        total = float(w.sum())
        if total <= 0:
            break
        x_new, y_new = float((w * xx).sum() / total), float((w * yy).sum() / total)
        if np.hypot(x_new - x, y_new - y) < 1e-3:
            x, y = x_new, y_new
            break
        x, y = x_new, y_new
    return x, y


def fit_position(image: np.ndarray, error: np.ndarray, render: Callable[[float, float], np.ndarray],
                 x0: float, y0: float, fit_mask: np.ndarray, *, max_shift: float, model_error: float = 0.0,
                 background: str = "constant", anchor: tuple[float, float] | None = None) -> tuple[float, float, float]:
    """The position (x, y) [pix] of the source on one image by least squares: ``render(x, y)`` draws the
    PSF there, amplitude and background are solved linearly at every trial; the search starts at
    (x0, y0) and stays within ``max_shift`` pixels of ``anchor`` (default: the start). Returns (x, y, chi2)."""
    from scipy.optimize import minimize

    ax, ay = anchor if anchor is not None else (x0, y0)

    def chi2(params: np.ndarray) -> float:
        x, y = float(params[0]), float(params[1])
        if np.hypot(x - ax, y - ay) > max_shift:
            return 1e30
        fit = fit_planes(image[None], error[None], render(x, y)[None], fit_mask, background=background,
                         origin=(x0, y0), model_error=model_error, iterations=2, clip=np.inf)
        value = float(fit.chi2[0])
        return value if np.isfinite(value) else 1e30

    result = minimize(chi2, np.array([x0, y0], float), method="Nelder-Mead",
                      options={"xatol": 2e-3, "fatol": 1e-3, "initial_simplex": np.array(
                          [[x0, y0], [x0 + 0.3, y0], [x0, y0 + 0.3]], float)})
    return float(result.x[0]), float(result.x[1]), float(result.fun)


def core_coverage(model: np.ndarray, finite: np.ndarray, x0: float, y0: float, radius: float) -> float:
    """Share of the PSF's light within ``radius`` [pix] of (x0, y0) that falls on finite spaxels of
    the grid (0 when the core is outside the field)."""
    ny, nx = model.shape
    # the core on a grid that extends past the field: what falls outside the array is light lost
    grid_r = np.hypot(*np.meshgrid(np.arange(nx) - x0, np.arange(ny) - y0))
    near = grid_r <= radius
    total_in_grid = float(model[near].sum())
    if total_in_grid <= 0:
        return 0.0
    area_full = np.pi * radius**2
    area_in_grid = max(float(near.sum()), 1.0)
    # light of the core in the array, corrected for the part of the disk that lies beyond the array
    full = total_in_grid * max(area_full / area_in_grid, 1.0) if area_in_grid < 0.5 * area_full else total_in_grid
    return float(model[near & finite].sum()) / full


# --------------------------------------------------------------------------- diagnostics
def ring_mask(shape: tuple[int, int], centre: tuple[float, float], radius: float, width: float) -> np.ndarray:
    """Spaxels whose centres lie within ``width`` [pix] of the circle of ``radius`` [pix] about ``centre``."""
    yy, xx = np.indices(shape, dtype=float)
    r = np.hypot(xx - centre[0], yy - centre[1])
    return np.abs(r - radius) <= width


def ring_residual(residual: np.ndarray, err: np.ndarray, ring: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per plane: the mean residual over the ring's finite spaxels, its formal error
    (sqrt(sum ERR^2) / n) and the number of spaxels."""
    r = np.where(ring[None] & np.isfinite(residual) & np.isfinite(err), residual, np.nan)
    n = np.isfinite(r).sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(r, axis=(1, 2)) / n
        e = np.sqrt(np.nansum(np.where(np.isfinite(r), err, np.nan) ** 2, axis=(1, 2))) / n
    mean[n == 0] = np.nan
    e[n == 0] = np.nan
    return mean, e, n


def radial_profile(image: np.ndarray, centre: tuple[float, float], edges: np.ndarray,
                   mask: np.ndarray | None = None) -> np.ndarray:
    """Azimuthal mean of the finite pixels of ``image`` (inside ``mask``) in the annuli ``edges`` [pix]
    about ``centre``; NaN for empty annuli."""
    yy, xx = np.indices(image.shape, dtype=float)
    r = np.hypot(xx - centre[0], yy - centre[1])
    ok = np.isfinite(image) & (mask if mask is not None else True)
    idx = np.digitize(r[ok], edges) - 1
    values = image[ok]
    valid = (idx >= 0) & (idx < len(edges) - 1)
    sums = np.bincount(idx[valid], weights=values[valid], minlength=len(edges) - 1)
    counts = np.bincount(idx[valid], minlength=len(edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


# --------------------------------------------------------------------------- the wing profile
# The PSF model of the neighbour holds too little light at a few FWHM (on HH 48 SW, PSF fitted alone:
# +20-90 % at 0.3-1.0" in MRS ch1, +5-60 % at 0.3-1.6" in ch2-ch3, within 10-20 % in ch4 -- scattered
# light and a broader PSF than stpsf's); fitted to the core it leaves a ring of positive residual
# around the neighbour. The wing profile is that residual's
# radial profile per unit PSF amplitude, measured on block-stacked residuals over the fitted region
# (all azimuths but the target's), smooth in wavelength, and folded into the model the planes are
# fitted with: a * (PSF + wing).
def radial_nodes(fwhm_pix: float, r_max_pix: float, *, start: float = 1.0, factor: float = 1.4) -> np.ndarray:
    """Radii [pix] of the wing profile's nodes: from ``start`` FWHM outwards in steps of ``factor``,
    the last within ``r_max_pix``."""
    first = max(float(start) * float(fwhm_pix), 1.0)
    if r_max_pix <= first:
        return np.zeros(0)
    n = int(np.floor(np.log(r_max_pix / first) / np.log(factor))) + 1
    return first * float(factor) ** np.arange(n)


def radial_tents(shape: tuple[int, int], x: float, y: float, nodes: np.ndarray, r_max_pix: float) -> np.ndarray:
    """(K, ny, nx): one tent per node about (x, y) -- rising linearly in radius from the node before (the
    first from ``nodes[0] / factor``, so nothing changes inside) and falling to the next (the last to
    ``r_max_pix``) -- each normalised to unit sum over the grid. Weighted and summed they are a
    piecewise-linear radial profile (:func:`fit_wing`)."""
    nodes = np.asarray(nodes, dtype=float)
    yy, xx = np.indices(shape, dtype=float)
    r = np.hypot(xx - x, yy - y)
    if not len(nodes):
        return np.zeros((0, *shape))
    inner = nodes[0] / (nodes[1] / nodes[0] if len(nodes) > 1 else 1.4)
    edges = np.concatenate([[inner], nodes, [max(float(r_max_pix), nodes[-1] + 1.0)]])
    out = np.empty((len(nodes), *shape))
    for k in range(len(nodes)):
        lo, mid, hi = edges[k], edges[k + 1], edges[k + 2]
        tent = np.clip(np.minimum((r - lo) / (mid - lo), (hi - r) / (hi - mid)), 0.0, None)
        out[k] = tent / max(float(tent.sum()), 1e-30)
    return out


def fit_wing(residual: np.ndarray, error: np.ndarray, tents: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """The weights (K,) of the tents that best describe one residual image over ``fit_mask`` (weighted
    least squares): the radial profile of what the model left. NaN when it cannot be solved."""
    ok = fit_mask & np.isfinite(residual) & np.isfinite(error) & (error > 0)
    if not len(tents) or ok.sum() < len(tents) + 2:
        return np.full(len(tents), np.nan)
    a = tents[:, ok].T / error[ok, None]
    used = np.abs(a).sum(axis=0) > 0
    coeffs = np.full(len(tents), np.nan)
    if not used.any():
        return coeffs
    solution, *_ = np.linalg.lstsq(a[:, used], residual[ok] / error[ok], rcond=None)
    coeffs[used] = solution
    coeffs[~used] = 0.0                       # a tent with no fitted spaxel adds nothing
    return coeffs


def wing_shape(coeffs: np.ndarray, block_waves: np.ndarray, wave: np.ndarray, tents: np.ndarray) -> np.ndarray:
    """(nw, ny, nx): the wing profile per unit PSF amplitude in every plane -- the block weights
    ``coeffs`` (nb, K) interpolated linearly in wavelength (held at the end blocks) -- times the tents."""
    good = np.isfinite(coeffs).all(axis=1)
    if not good.any() or not len(tents):
        return np.zeros((len(wave), *tents.shape[1:]))
    bw, c = np.asarray(block_waves, float)[good], np.asarray(coeffs, float)[good]
    per_plane = np.stack([np.interp(wave, bw, c[:, k]) for k in range(c.shape[1])], axis=1)   # (nw, K)
    return np.einsum("wk,kyx->wyx", per_plane, tents)


# --------------------------------------------------------------------------- the procedure
@dataclass
class Located:
    """Where the neighbour is: the position adopted for the fit and how it was found."""

    x: float                   #: adopted position [0-based cube pixels]
    y: float
    given: tuple[float, float]  #: the position the step was given
    refined: bool              #: fitted on the cube (core in the field), else the given position
    coverage: float            #: share of the core's light (within one FWHM) on finite spaxels
    blocks: dict[str, np.ndarray]  #: per block of planes: WAVELENGTH, X, Y (fit), XC, YC (centroid), CHI2


def slice_at(psf: PsfCubeProduct, wavelength_um: float) -> PsfCubeProduct:
    """A one-slice library: the PSF at ``wavelength_um`` interpolated between the bracketing slices,
    keeping their sums (all of the light)."""
    return type(psf)(psf.at(wavelength_um, normalized=False)[None], np.array([float(wavelength_um)]),
                     psf.pixelscale_arcsec, meta=dict(psf.meta), header=psf.header)


def locate(data: np.ndarray, err: np.ndarray, wave: np.ndarray, psf: PsfCubeProduct, given: tuple[float, float], *,
           pixel_arcsec: float, rotation_deg: float, exclude: np.ndarray, fwhm_pix: float, max_shift: float,
           n_blocks: int = 8, model_error: float = 0.05, sampling: str = "point", refine: bool = True,
           min_coverage: float = 0.5) -> Located:
    """The neighbour's position on the cube.

    The PSF at the middle wavelength is placed at the ``given`` position;
    when at least ``min_coverage`` of its light within one FWHM falls on
    finite spaxels the core is in the field and the position is fitted --
    on ``n_blocks`` block-median images, each started from the centroid
    within one FWHM (when that lies within ``max_shift`` pixels of the given
    position), the PSF fitted within three FWHM of it, off ``exclude`` --
    and the median of the block positions adopted. Otherwise (a neighbour
    outside the field, or ``refine`` off) the given position stands.
    """
    shape = data.shape[1:]
    finite = np.isfinite(data).any(axis=0)
    gx, gy = (float(v) for v in given)
    mid = float(np.nanmedian(wave))
    core = slice_at(psf, mid).placed(shape, gx, gy, pixel_arcsec, rotation_deg=rotation_deg, sampling=sampling)[0]
    coverage = core_coverage(core, finite, gx, gy, max(fwhm_pix, 1.0))
    empty = {k: np.zeros(0) for k in ("WAVELENGTH", "X", "Y", "XC", "YC", "CHI2")}
    if not refine or coverage < min_coverage:
        return Located(gx, gy, (gx, gy), False, coverage, empty)
    images, errors, ranges = block_stack(data, err, n_blocks)
    yy, xx = np.indices(shape, dtype=float)
    rows: dict[str, list[float]] = {k: [] for k in empty}
    for image, error, (a, b) in zip(images, errors, ranges):
        lam = float(np.nanmedian(wave[a:b]))
        one = slice_at(psf, lam)

        def render(x: float, y: float, one: PsfCubeProduct = one) -> np.ndarray:
            return one.placed(shape, x, y, pixel_arcsec, rotation_deg=rotation_deg, sampling=sampling)[0]

        cx, cy = centroid(np.where(exclude, np.nan, image), gx, gy, max(fwhm_pix, 1.5))
        start = (cx, cy) if np.hypot(cx - gx, cy - gy) <= max_shift else (gx, gy)
        near = (np.hypot(xx - start[0], yy - start[1]) <= 3.0 * max(fwhm_pix, 1.0)) & ~exclude & np.isfinite(image)
        if near.sum() < 10:
            continue
        fx, fy, c2 = fit_position(image, error, render, start[0], start[1], near, max_shift=max_shift,
                                  model_error=model_error, anchor=(gx, gy))
        for key, value in zip(empty, (lam, fx, fy, cx, cy, c2)):
            rows[key].append(float(value))
    table = {k: np.asarray(v, dtype=float) for k, v in rows.items()}
    if not len(table["X"]):
        return Located(gx, gy, (gx, gy), False, coverage, table)
    return Located(float(np.median(table["X"])), float(np.median(table["Y"])), (gx, gy), True, coverage, table)


def library_reach(psf: PsfCubeProduct, shape: tuple[int, int], x: float, y: float, pixel_arcsec: float) -> np.ndarray:
    """Spaxels the library reaches from a source at (x, y): within the circle inscribed in its field
    (less two library pixels for the spline). Beyond it the placed PSF is zero."""
    radius = ((psf.data.shape[-1] - 1) / 2.0 - 2.0) * psf.pixelscale_arcsec / pixel_arcsec
    yy, xx = np.indices(shape, dtype=float)
    return np.hypot(xx - x, yy - y) <= radius


def required_field(shape: tuple[int, int], finite: np.ndarray, x: float, y: float, pixel_arcsec: float) -> float:
    """The library field [arcsec] that reaches every finite spaxel from a source at (x, y):
    twice the largest distance, plus a spaxel."""
    yy, xx = np.indices(shape, dtype=float)
    r = np.hypot(xx - x, yy - y)[finite]
    return 2.0 * (float(r.max()) + 1.0) * pixel_arcsec if r.size else 0.0


def components(psfs: list[PsfCubeProduct], shape: tuple[int, int], x: float, y: float, wave: np.ndarray, *,
               pixel_arcsec: float, rotation_deg: float, sampling: str = "point") -> list[np.ndarray]:
    """Each library (the PSF, its broadened copies) placed at (x, y) and interpolated to every plane:
    (nw, ny, nx) per library, the share of all of the source's light per spaxel."""
    return [plane_models(lib.placed(shape, x, y, pixel_arcsec, rotation_deg=rotation_deg, sampling=sampling),
                         lib.wavelengths, wave) for lib in psfs]


def fill_amplitudes(amplitudes: np.ndarray, *, smooth: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """The amplitudes (nw, ncomp) to subtract: planes whose fit failed take the linear interpolation of
    their neighbours (flagged), and a running median over ``smooth`` planes (odd; 1 = per plane) is
    applied. Returns (amplitudes, filled)."""
    from scipy.ndimage import median_filter

    amps = np.array(amplitudes, dtype=float, copy=True)
    filled = ~np.isfinite(amps).all(axis=1)
    idx = np.arange(len(amps))
    if filled.all():
        return np.zeros_like(amps), filled
    for j in range(amps.shape[1]):
        good = np.isfinite(amps[:, j])
        amps[~good, j] = np.interp(idx[~good], idx[good], amps[good, j])
        if smooth > 1:
            amps[:, j] = median_filter(amps[:, j], size=int(smooth) | 1, mode="nearest")
    return amps, filled


__all__ = [
    "BACKGROUND_TERMS",
    "Located",
    "PlaneFit",
    "block_stack",
    "broadened",
    "centroid",
    "components",
    "core_coverage",
    "fill_amplitudes",
    "fit_planes",
    "fit_position",
    "fit_wing",
    "interpolation_nodes",
    "library_reach",
    "locate",
    "plane_models",
    "radial_nodes",
    "radial_profile",
    "radial_tents",
    "required_field",
    "ring_mask",
    "ring_residual",
    "slice_at",
    "wing_shape",
]
