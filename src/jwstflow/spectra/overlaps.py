"""Overlaps of neighbouring spectral segments: the flux ratio around a crossover, chained scales.

Every stitch -- of 1-D spectra (:mod:`jwstflow.spectra.stitching`) or of IFU cubes
(:mod:`jwstflow.cubes.stitching`) -- rests on the same two measurements, kept here so
that both make them the same way:

* :func:`measure_overlap_ratio` -- the flux ratio of two overlapping segments in a
  small wavelength window around their crossover, with its 1-sigma uncertainty
  (an :class:`OverlapRatio`);
* :func:`chain_scales` -- the multiplicative scale of every segment that puts a chain
  of such ratios onto one reference segment, with the accumulated uncertainty;

and :func:`overlap_midpoints`, the default crossovers.

A *segment* is a dict with a ``label`` and finite, wavelength-ordered ``WAVELENGTH`` /
``FLUX`` / ``FLUX_ERROR`` arrays.
"""

from __future__ import annotations

import logging
from itertools import pairwise
from typing import Any, NamedTuple

import numpy as np

log = logging.getLogger(__name__)


class OverlapRatio(NamedTuple):
    """One neighbour-pair flux-ratio measurement (the result of :func:`measure_overlap_ratio`).

    ``ratio`` is the flux ratio ``b/a`` (NaN when not measurable) and
    ``error`` its 1-sigma uncertainty, propagated from the FLUX_ERROR samples
    of the measurement region (NaN when the ratio is; falls back to the
    scatter of the samples when neither segment carries usable errors).
    ``window`` is the wavelength region [um] the measurement actually used --
    the ``ratio_window_frac`` window around the crossover, clipped to the
    overlap (widened back to the whole overlap when the window holds too few
    samples) -- or None when the segments do not overlap. ``n`` counts the
    samples measured.
    """

    ratio: float
    error: float
    window: tuple[float, float] | None
    n: int

    @classmethod
    def unmeasurable(cls, window: tuple[float, float] | None = None, n: int = 0) -> OverlapRatio:
        return cls(float("nan"), float("nan"), window, n)


def overlap_midpoints(ranges: list[tuple[float, float]]) -> list[float]:
    """Default crossovers of wavelength-ordered segments with these ``(min, max)`` ranges [um]:
    the midpoint of each overlap -- or of the gap, when neighbours do not overlap."""
    return [0.5 * (float(b[0]) + float(a[1])) for a, b in pairwise(ranges)]


def measure_overlap_ratio(a: dict[str, Any], b: dict[str, Any], *, crossover: float,
                          ratio_window_frac: float = 0.05, min_overlap_points: int = 5,
                          min_overlap_snr: float = 3.0) -> OverlapRatio:
    """Flux ratio ``b/a`` around the crossover, with its uncertainty (:class:`OverlapRatio`).

    The measurement region is the wavelength window of full width
    ``ratio_window_frac * crossover`` centred on the crossover, clipped
    to the overlap -- the factor is estimated where it is applied,
    instead of over the whole overlap whose far ends carry the largest
    calibration differences. When the window holds fewer than
    ``min_overlap_points`` finite samples it is widened back to the
    whole overlap (with a warning); when even that holds too few, the
    ratio is unmeasurable (NaN).

    The ratio is the ratio of the two segments' median fluxes over the
    window samples -- robust against spikes and, unlike a median of
    per-sample ratios, against noise crossing zero. Its 1-sigma
    uncertainty is propagated from the segments' FLUX_ERROR samples in
    the window (the error of each median, combined in quadrature as
    relative errors); segments without usable errors fall back to the
    robust scatter of their samples. With ``min_overlap_snr`` both
    medians must also sit that many sigmas above zero -- judged against
    the larger of the propagated and the scatter-based error, so
    underestimated FLUX_ERRORs cannot sneak a near-zero overlap past
    the guard: an overlap whose flux is consistent with zero cannot
    anchor a multiplicative rescaling (dividing by it amplifies the
    redder segments by huge or even negative factors), so such a ratio
    comes back NaN -- the chain then leaves that link unscaled, with a
    warning here.

    ``a`` and ``b`` are segment dicts (``label`` plus finite, wavelength-ordered
    ``WAVELENGTH``/``FLUX``/``FLUX_ERROR`` arrays), ``a`` the bluer one. Samples of
    ``a`` whose FLUX is NaN are left out, which is how callers mask wavelength
    ranges (emission lines, band edges) out of the measurement.
    """
    lo, hi = float(b["WAVELENGTH"].min()), float(a["WAVELENGTH"].max())
    if hi <= lo:
        log.warning("no overlap between %s and %s (%.3f-%.3f um); ratio not measured",
                    a["label"], b["label"], lo, hi)
        return OverlapRatio.unmeasurable()
    half = 0.5 * ratio_window_frac * float(crossover)
    window = (max(lo, float(crossover) - half), min(hi, float(crossover) + half))

    def finite_pairs(wlo: float, whi: float) -> tuple[np.ndarray, ...]:
        inside = (a["WAVELENGTH"] >= wlo) & (a["WAVELENGTH"] <= whi)
        wave = a["WAVELENGTH"][inside]
        a_flux, a_err = a["FLUX"][inside], a["FLUX_ERROR"][inside]
        b_flux = np.interp(wave, b["WAVELENGTH"], b["FLUX"])
        b_err = np.interp(wave, b["WAVELENGTH"], b["FLUX_ERROR"])
        ok = np.isfinite(a_flux) & np.isfinite(b_flux)
        return a_flux[ok], a_err[ok], b_flux[ok], b_err[ok]

    samples = finite_pairs(*window)
    if len(samples[0]) < min_overlap_points and window != (lo, hi):
        log.warning("ratio window %.3f-%.3f um around the %s/%s crossover holds %d finite sample(s) "
                    "(< %d); widening to the whole overlap %.3f-%.3f um",
                    *window, a["label"], b["label"], len(samples[0]), min_overlap_points, lo, hi)
        window = (lo, hi)
        samples = finite_pairs(*window)
    a_flux, a_err, b_flux, b_err = samples
    n = len(a_flux)
    if n < min_overlap_points:
        log.warning("overlap between %s and %s (%.3f-%.3f um) holds %d finite sample(s) (< %d); "
                    "ratio not measured", a["label"], b["label"], lo, hi, n, min_overlap_points)
        return OverlapRatio.unmeasurable(window, n)
    med_a, scatter_a = _median_and_error(a_flux)
    med_b, scatter_b = _median_and_error(b_flux)
    prop_a = _median_error_from_flux_errors(a_err)
    prop_b = _median_error_from_flux_errors(b_err)
    err_a = prop_a if prop_a is not None else scatter_a
    err_b = prop_b if prop_b is not None else scatter_b
    guard_a, guard_b = max(err_a, scatter_a), max(err_b, scatter_b)
    if min_overlap_snr > 0 and not (med_a > min_overlap_snr * guard_a and med_b > min_overlap_snr * guard_b):
        log.warning("ratio window %.3f-%.3f um between %s and %s: flux consistent with zero "
                    "(%.3g +- %.3g vs %.3g +- %.3g); ratio not measurable, that link is not rescaled",
                    *window, a["label"], b["label"], med_a, guard_a, med_b, guard_b)
        return OverlapRatio.unmeasurable(window, n)
    with np.errstate(all="ignore"):
        ratio = med_b / med_a
        error = abs(ratio) * float(np.hypot(err_a / med_a, err_b / med_b))
    return OverlapRatio(float(ratio), float(error), window, n)


def chain_scales(labels: list[str], ratios: list[OverlapRatio], reference: str | None) -> tuple[list[float], list[float]]:
    """Chain the neighbour ratios so the reference segment keeps scale 1 (error 0).

    ``reference`` is a label, ``'shortest'`` or ``'longest'`` (None: the
    reddest segment). Each link multiplies (bluewards) or divides (redwards)
    by its ratio; the relative uncertainties of the ratios crossed on the way
    add in quadrature, so a segment far from the reference carries the
    combined uncertainty of every measurement between them. Unmeasurable
    links -- a ratio that is not finite or not positive -- stay at factor 1
    and contribute no uncertainty (the warning was already given where the
    ratio was measured). Returns the scales and their 1-sigma uncertainties,
    one per segment.
    """
    if reference in (None, "longest"):
        ref = len(labels) - 1
    elif reference == "shortest":
        ref = 0
    elif reference in labels:
        ref = labels.index(reference)
    else:
        raise ValueError(f"reference segment {reference!r} is not among {labels} (nor 'shortest'/'longest')")
    scales = [1.0] * len(labels)
    rel_var = [0.0] * len(labels)

    def link(r: OverlapRatio) -> tuple[float, float]:
        if not (np.isfinite(r.ratio) and r.ratio > 0):        # a flux scale is positive; anything else is no measurement
            return 1.0, 0.0
        return r.ratio, (r.error / r.ratio) ** 2 if np.isfinite(r.error) else 0.0

    for i in range(ref - 1, -1, -1):  # bluewards of the reference
        factor, var = link(ratios[i])
        scales[i] = scales[i + 1] * factor
        rel_var[i] = rel_var[i + 1] + var
    for i in range(ref, len(labels) - 1):  # redwards of the reference
        factor, var = link(ratios[i])
        scales[i + 1] = scales[i] / factor
        rel_var[i + 1] = rel_var[i] + var
    return scales, [abs(s) * float(np.sqrt(v)) for s, v in zip(scales, rel_var)]


def mask_windows(wavelength: np.ndarray, flux: np.ndarray, windows: list[tuple[float, float]] | None,
                 *, within: tuple[float, float] | None = None) -> np.ndarray:
    """``flux`` with the samples inside any of the wavelength ``windows`` set to NaN -- what
    takes spectral features out of :func:`measure_overlap_ratio` (which skips NaN samples of
    its bluer segment). When the windows cover every sample of the range ``within`` (the
    overlap; default: all samples) the flux comes back unmasked: better a contaminated
    ratio than none."""
    if not windows:
        return flux
    masked = np.zeros(len(wavelength), dtype=bool)
    for lo, hi in windows:
        masked |= (wavelength >= lo) & (wavelength <= hi)
    scope = np.ones(len(wavelength), dtype=bool) if within is None else (wavelength > within[0]) & (wavelength < within[1])
    if not masked.any() or (scope.any() and masked[scope].all()):
        return flux
    return np.where(masked, np.nan, flux)


def _median_and_error(values: np.ndarray) -> tuple[float, float]:
    """Median of ``values`` and the robust (MAD-based) uncertainty of that median."""
    med = float(np.median(values))
    mad_sigma = 1.4826 * float(np.median(np.abs(values - med)))
    return med, 1.2533 * mad_sigma / np.sqrt(len(values))


def _median_error_from_flux_errors(errors: np.ndarray) -> float | None:
    """Uncertainty of the median of N samples, propagated from their per-sample errors.

    The error of the mean is ``sqrt(mean(sigma_i^2) / N)``; the median of a
    Gaussian sample is ``sqrt(pi/2) = 1.2533`` times noisier. Samples without
    a usable error are left out of the typical sigma but still count toward
    N (they are part of the median); None when no sample carries one.
    """
    errors = np.asarray(errors, dtype=float)
    usable = errors[np.isfinite(errors) & (errors > 0)]
    if not len(usable):
        return None
    return 1.2533 * float(np.sqrt(np.mean(usable**2) / len(errors)))
