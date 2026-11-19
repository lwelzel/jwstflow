"""Measurements on cubes that share one spatial grid: footprints, regions, region spectra, image shifts.

What a cube stitch measures in an overlap is measured over a *region* of
spaxels that both segments cover, summed into a spectrum per segment -- the
1-D ratio measurement (:mod:`jwstflow.spectra.overlaps`) then applies as it
stands. The same planes, collapsed to an image per segment, tell whether the
two segments put the source at the same place: the two images are registered
against each other over one window around the bright core they share.
"""

from __future__ import annotations

import warnings

import numpy as np

#: A spaxel belongs to the footprint of a cube when it carries data in this fraction of the
#: planes that carry any.
FOOTPRINT_FRACTION = 0.9
#: The bright core of an image (or of a pair of images) is what stands above this fraction of its peak excess.
CORE_LEVEL = 0.5
#: The window two images are registered in reaches this many spaxels beyond their shared bright core.
WINDOW_MARGIN = 3
#: The rim of a field: its spaxels within this many spaxels of where the coverage ends. A bright
#: piece of the field that reaches it is a source the field cuts.
EDGE_MARGIN = 2


def footprint(valid: np.ndarray, min_fraction: float = FOOTPRINT_FRACTION) -> np.ndarray:
    """Spaxels ``(ny, nx)`` that carry data in at least ``min_fraction`` of the planes of
    ``valid`` (nw, ny, nx) that carry any data at all."""
    planes = valid.any(axis=(1, 2))
    if not planes.any():
        return np.zeros(valid.shape[1:], dtype=bool)
    return valid[planes].mean(axis=0) >= min_fraction


def collapse(data: np.ndarray) -> np.ndarray:
    """NaN-aware mean of planes ``(n, ny, nx)``; NaN where no plane has a value."""
    if data.shape[0] == 0:
        return np.full(data.shape[1:], np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings("ignore", message=".*(Mean of empty slice|All-NaN).*")
        return np.nanmean(data, axis=0)


def smooth3(image: np.ndarray, region: np.ndarray) -> np.ndarray:
    """The 3x3 median of ``image`` over the spaxels of ``region`` (NaN outside it): what a
    single deviant spaxel cannot set. Bright parts of a field are found on it, never on
    the image itself -- a hot spaxel is not a source."""
    masked = np.where(region & np.isfinite(image), image, np.nan)
    padded = np.pad(masked, 1, constant_values=np.nan)
    ny, nx = image.shape
    stack = np.stack([padded[dy:dy + ny, dx:dx + nx] for dy in range(3) for dx in range(3)])
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings("ignore", message=".*All-NaN.*")
        return np.where(np.isfinite(masked), np.nanmedian(stack, axis=0), np.nan)


def levels(smooth: np.ndarray) -> tuple[float, float]:
    """``(floor, peak)`` of a smoothed image (:func:`smooth3`): its median -- the level of the
    field: zero for a background-subtracted cube, the sky for any other -- and its maximum."""
    values = smooth[np.isfinite(smooth)]
    if not values.size:
        return float("nan"), float("nan")
    return float(np.median(values)), float(values.max())


def rim(field: np.ndarray, margin: int = EDGE_MARGIN) -> np.ndarray:
    """The spaxels of ``field`` within ``margin`` spaxels of its outer edge (holes inside it have no rim)."""
    from scipy import ndimage

    filled = np.pad(ndimage.binary_fill_holes(field), 1)             # the edge of the array is an edge of the field
    depth = ndimage.distance_transform_cdt(filled, metric="chessboard")[1:-1, 1:-1]
    return field & (depth <= int(margin))


def source_region(image_a: np.ndarray, image_b: np.ndarray, common: np.ndarray, fraction: float,
                  min_spaxels: int = 9) -> np.ndarray:
    """The part of the common field where the source stands out, judged on both images alike.

    The mean of the two images (each segment's planes around the crossover),
    smoothed with a 3x3 median, is compared with its own floor and a peak
    over ``common``: the region holds the spaxels above ``floor + fraction *
    (peak - floor)``. Selecting on the mean treats the two segments
    symmetrically, measuring from the floor makes the selection the same
    whether or not the cubes still carry a uniform background, and the
    median keeps single deviant spaxels out.

    Only whole sources count. A bright piece that reaches the rim of the
    field (:func:`rim`) is a source the field cuts: how much of it a band
    holds depends on that band's footprint and pointing, and the rim is
    where a cube is least reliable -- two bands do not agree on it however
    well they are calibrated. Such a piece is left out as a whole, and the
    peak is the brightest spaxel of what remains (a brighter neighbour at
    the edge of the field does not set the threshold for the target). Only
    when everything bright reaches the rim -- a source that fills the field
    -- is the region what stands out clear of the rim. Falls back to all of
    ``common`` when fewer than ``min_spaxels`` spaxels pass (no source to
    speak of).
    """
    from scipy import ndimage

    both = common & np.isfinite(image_a) & np.isfinite(image_b)
    smooth = smooth3(0.5 * (image_a + image_b), both)
    floor, peak = levels(smooth)
    if not (np.isfinite(floor) and np.isfinite(peak) and peak > floor):
        return both
    edge = rim(both)

    def bright(peak: float) -> tuple[np.ndarray, np.ndarray]:
        """What stands above the threshold of this peak, and the pieces of it that reach the rim."""
        with np.errstate(invalid="ignore"):
            above = both & (smooth >= floor + fraction * (peak - floor))
        labelled, _ = ndimage.label(above)
        return above, above & np.isin(labelled, np.unique(labelled[above & edge]))

    region = np.zeros(both.shape, dtype=bool)
    candidates = both & ~edge & np.isfinite(smooth)
    while candidates.any():
        peak = float(smooth[candidates].max())
        if not peak > floor:
            break
        above, cut = bright(peak)
        if not cut[candidates & (smooth == peak)].all():       # the peak is that of a whole source
            region = above & ~cut
            break
        # every spaxel of a cut piece would, as the peak, find itself in a cut piece again (a lower
        # threshold only grows the piece): none of them is the peak of a whole source
        candidates &= ~cut
    if int(region.sum()) < min_spaxels and (both & ~edge & np.isfinite(smooth)).any():
        region = bright(float(smooth[both & ~edge & np.isfinite(smooth)].max()))[0] & ~edge
    return region if int(region.sum()) >= min_spaxels else both


def region_spectrum(data: np.ndarray, err: np.ndarray | None, region: np.ndarray, *, pixel_area_sr: float,
                    correlation: float = 1.0, field: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Flux density and its error of a region, plane by plane: ``(flux, error)`` in Jy for a cube in MJy/sr.

    The sum of the surface brightness over the region times the spaxel solid
    angle. A plane in which any spaxel of the region is missing gets NaN --
    a partial sum is not the flux of the region. The error adds the spaxel
    errors in quadrature, times ``sqrt(correlation)`` for a resampled cube
    whose neighbouring spaxels share source pixels
    (:attr:`~jwstflow.cubes.stitching.regrid.Placement.correlation`).

    With ``field`` (a spaxel mask: the field the region lies in) the flux is
    the *excess over the level of the field*: in every plane the median over
    the field is taken off each spaxel of the region first (planes with less
    than :data:`FOOTPRINT_FRACTION` of the field are NaN). That is what makes
    the fluxes of two segments comparable as a ratio: a constant level --
    sky, or what a sky subtraction left behind, different in every band --
    drops out, and a gain common to source and level stays a factor. The
    uncertainty of the level is part of the error.
    """
    n = data.shape[0]
    if not region.any():
        return np.full(n, np.nan), np.full(n, np.nan)
    values = data[:, region].astype(np.float64)
    complete = np.isfinite(values).all(axis=1)
    scale = pixel_area_sr * 1e6
    total = np.where(np.isfinite(values), values, 0.0).sum(axis=1)
    level_variance = np.zeros(n)
    n_region = int(region.sum())
    if field is not None:
        around = data[:, field].astype(np.float64)
        enough = np.isfinite(around).mean(axis=1) >= FOOTPRINT_FRACTION
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.filterwarnings("ignore", message=".*All-NaN.*")
            level = np.where(enough, np.nanmedian(around, axis=1), np.nan)
        total = total - n_region * level
        complete &= enough
        if err is not None:
            field_errors = err[:, field].astype(np.float64)
            counts = np.isfinite(around).sum(axis=1).clip(1)
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.filterwarnings("ignore", message=".*(Mean of empty slice|All-NaN).*")
                # the median of N values is sqrt(pi / 2) times noisier than their mean
                level_variance = 1.5708 * np.nanmean(field_errors**2, axis=1) / counts * correlation
            level_variance = np.where(np.isfinite(level_variance), level_variance, 0.0)
    flux = np.where(complete, total, np.nan) * scale
    if err is None:
        return flux, np.full(n, np.nan)
    errors = err[:, region].astype(np.float64)
    variance = np.where(np.isfinite(errors), errors, 0.0) ** 2
    error = np.where(complete, np.sqrt(variance.sum(axis=1) * correlation + n_region**2 * level_variance), np.nan) * scale
    return flux, error


def shared_window(image_a: np.ndarray, image_b: np.ndarray, field: np.ndarray, *, within: np.ndarray | None = None,
                  level: float = CORE_LEVEL, margin: int = WINDOW_MARGIN) -> np.ndarray:
    """The window in which two images of one field are compared: around the bright core they share.

    Each image is taken as its excess over its own field level (on the 3x3
    median); the core is the connected region around the peak of the *mean*
    of the two excesses, above ``level`` of that peak, and the window is the
    core grown by ``margin`` spaxels inside ``field``. The peak is looked for
    in ``within`` (default: all of ``field``) -- the region a ratio is
    measured over, so that both measurements are of one source. One window
    for both images: when each image picked its own brightest spot, two
    segments with another source at the edge of the field (brighter in one
    of them) would be compared on different objects. Empty when there is no
    peak.
    """
    from scipy import ndimage

    excess = []
    for image in (image_a, image_b):
        smooth = smooth3(image, field)
        excess.append(smooth - levels(smooth)[0])
    mean = 0.5 * (excess[0] + excess[1])
    search = np.where(field if within is None else (within & field), mean, np.nan)
    if not np.isfinite(search).any() or not np.nanmax(search) > 0:
        return np.zeros(field.shape, dtype=bool)
    peak_at = np.unravel_index(int(np.nanargmax(search)), search.shape)
    with np.errstate(invalid="ignore"):
        above = mean >= level * mean[peak_at]
    labelled, _ = ndimage.label(above)
    core = labelled == labelled[peak_at]
    return ndimage.binary_dilation(core, iterations=max(int(margin), 1)) & field


def image_shift(image_a: np.ndarray, image_b: np.ndarray, window: np.ndarray) -> tuple[float, float]:
    """Shift ``(dx, dy)`` [pixels] of image ``b`` against image ``a`` over a window, scale and level free.

    The shift ``d`` for which ``b(x) = k a(x - d) + c`` fits best over the
    spaxels of ``window``: both images are moved towards each other by half
    of ``d`` (cubic-spline interpolation, so neither is the one that gets
    smoothed), the factor ``k`` and the level ``c`` between them are solved
    for at every trial shift -- neither a calibration difference nor a
    pedestal is taken for a shift -- and ``d`` is found by least squares.

    The fit is robust against what only one image shows. A spaxel that
    differs between the images by more than half of what the source spans in
    the window -- no sub-spaxel shift does that; a hot spaxel does -- takes
    the 3x3 median of its image before the interpolation (which would spread
    it) and has no say, nor have its neighbours; smaller deviations lose
    their weight as the fit is repeated. Meant for shifts of up to a spaxel
    or so; NaN when the window holds too little to tell.
    """
    from scipy import ndimage, optimize

    a, b = np.asarray(image_a, dtype=float), np.asarray(image_b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    # spaxels next to missing data would take interpolated fill: leave them out
    use = window & valid & ~ndimage.binary_dilation(~valid, iterations=2)
    if int(use.sum()) < 12:
        return float("nan"), float("nan")
    filled_a = np.where(np.isfinite(a), a, float(np.median(a[valid])))
    filled_b = np.where(np.isfinite(b), b, float(np.median(b[valid])))
    smooth_a, smooth_b = smooth3(a, valid), smooth3(b, valid)
    amplitude = float(np.ptp(smooth_b[use])) or 1.0
    k, c = np.polyfit(smooth_a[use], smooth_b[use], 1)           # on the medians: what a hot spaxel cannot bend
    gross = use & (np.abs(b - (k * a + c)) > 0.5 * amplitude)
    if gross.any():
        in_a = np.abs(a - smooth_a) * abs(k) >= np.abs(b - smooth_b)
        filled_a = np.where(gross & in_a, smooth_a, filled_a)
        filled_b = np.where(gross & ~in_a, smooth_b, filled_b)
        use = use & ~ndimage.binary_dilation(gross, iterations=1)
        if int(use.sum()) < 12:
            return float("nan"), float("nan")

    def residuals(shift: np.ndarray, weights: np.ndarray) -> np.ndarray:
        dx, dy = float(shift[0]), float(shift[1])
        x = ndimage.shift(filled_a, (0.5 * dy, 0.5 * dx), order=3, mode="nearest")[use]
        y = ndimage.shift(filled_b, (-0.5 * dy, -0.5 * dx), order=3, mode="nearest")[use]
        design = np.column_stack([x, np.ones(len(x))]) * weights[:, None]
        coefficients, *_ = np.linalg.lstsq(design, y * weights, rcond=None)
        return (y - coefficients[0] * x - coefficients[1]) * weights

    solution = np.zeros(2)
    weights = np.ones(int(use.sum()))
    for _ in range(4):                                           # reweighted least squares: outliers fade out
        fit = optimize.least_squares(residuals, solution, args=(weights,), method="lm", diff_step=1e-3)
        solution = fit.x
        left = residuals(solution, np.ones(len(weights)))
        scale = max(1.4826 * float(np.median(np.abs(left - np.median(left)))), 1e-6 * amplitude)
        updated = np.sqrt(1.0 / (1.0 + (left / (4.0 * scale)) ** 2))      # Cauchy weights at four robust sigmas
        if np.allclose(updated, weights, atol=1e-3):
            break
        weights = updated
    if not np.all(np.isfinite(solution)) or float(np.hypot(*solution)) > 3.0:
        return float("nan"), float("nan")              # beyond what a fit from zero shift can be trusted with
    return float(solution[0]), float(solution[1])


def pair_shift(planes_a: np.ndarray, planes_b: np.ndarray, field: np.ndarray, *,
               within: np.ndarray | None = None) -> dict[str, object]:
    """Where two segments put the source, from their planes around a crossover.

    ``planes_a`` / ``planes_b`` are the planes ``(n, ny, nx)`` of the two
    segments inside the measurement window in wavelength, on one grid;
    ``field`` is what both cover and ``within`` where the source is looked
    for (the region of the ratio; default: the field). Their collapsed
    images are registered against each other (:func:`image_shift`) over the
    window around the bright core they share (:func:`shared_window`). Returns the shift
    ``dx, dy`` of ``b`` against ``a`` [pixels], its 1-sigma uncertainty
    ``ex, ey`` -- from the two halves of the planes taken alternately, which
    see the same sky and independent noise --, the position ``x, y`` of the
    shared core (the centroid of the mean excess), the ``window`` and, for
    figures, the part of the window in which each segment stands above half
    of its own peak there (``core_a``, ``core_b``). NaN / empty where
    nothing can be measured.
    """
    image_a, image_b = collapse(planes_a), collapse(planes_b)
    window = shared_window(image_a, image_b, field, within=within)
    empty = np.zeros(field.shape, dtype=bool)
    out: dict[str, object] = {"dx": float("nan"), "dy": float("nan"), "ex": float("nan"), "ey": float("nan"),
                              "x": float("nan"), "y": float("nan"), "window": window, "core_a": empty, "core_b": empty}
    if not window.any():
        return out
    out["dx"], out["dy"] = image_shift(image_a, image_b, window)
    if min(planes_a.shape[0], planes_b.shape[0]) >= 4:
        even = image_shift(collapse(planes_a[0::2]), collapse(planes_b[0::2]), window)
        odd = image_shift(collapse(planes_a[1::2]), collapse(planes_b[1::2]), window)
        out["ex"], out["ey"] = abs(even[0] - odd[0]) / 2.0, abs(even[1] - odd[1]) / 2.0   # the error of the mean of two halves
    weights = np.zeros(field.shape)
    for key, image in (("core_a", image_a), ("core_b", image_b)):
        smooth = smooth3(image, field)
        excess = np.where(window, smooth - levels(smooth)[0], np.nan)
        if np.isfinite(excess).any() and np.nanmax(excess) > 0:
            with np.errstate(invalid="ignore"):
                out[key] = window & (excess >= CORE_LEVEL * np.nanmax(excess))
            weights += np.where(np.isfinite(excess), np.clip(excess / np.nanmax(excess), 0.0, None), 0.0)
    if weights.sum() > 0:
        yy, xx = np.indices(field.shape)
        out["x"], out["y"] = float((weights * xx).sum() / weights.sum()), float((weights * yy).sum() / weights.sum())
    return out
