"""Line plots: mid-point steps that break at gaps, uncertainty bands, reference
lines and windows, and logarithmic axes with plain-number ticks.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .style import CMAP, LW, MAIN_COLOR, REF_COLOR, ROLE_COLORS, _defer

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- lines (rules 3 + 6)
def line_colors(n: int) -> list[Any]:
    """Colors for an *ordered* family of ``n`` lines (bands or segments by wavelength,
    slits, ...): black when there is only one, otherwise ``n`` colors sampled from
    :data:`CMAP`. Lines that differ in *role* take :data:`ROLE_COLORS` instead."""
    if n <= 1:
        return [MAIN_COLOR]
    import cmasher as cmr

    return list(cmr.take_cmap_colors(CMAP, n, cmap_range=(0.15, 0.80)))


def break_gaps(x: Any, *ys: Any, factor: float = 4.0) -> tuple[np.ndarray, ...]:
    """Insert a NaN sample into every gap of the abscissa, so lines and bands break there.

    A gap is a step of ``x`` more than ``factor`` times the typical step on
    *both* sides of it (the larger of the two five-sample medians) -- a
    detector gap or a masked stretch that was dropped from the table. A
    change of sampling (the junction of two gratings) is not a gap: the step
    there matches one of its sides. Returns ``(x, *ys)`` with two extra
    samples per gap (every y NaN), one typical step inside each edge -- so
    that in a mid-point step plot the samples on either side keep their own
    bin width instead of reaching into the gap; inputs come back as float
    arrays unchanged when there is none.
    """
    x = np.asarray(x, dtype=float)
    arrays = [np.asarray(y, dtype=float) for y in ys]
    if x.size < 4:
        return (x, *arrays)
    dx = np.abs(np.diff(x))
    at: list[int] = []
    fill: list[float] = []
    for i in np.nonzero(np.isfinite(dx))[0]:
        left, right = dx[max(0, i - 5):i], dx[i + 1:i + 6]
        left, right = left[np.isfinite(left)], right[np.isfinite(right)]
        step_left = float(np.median(left)) if left.size else 0.0
        step_right = float(np.median(right)) if right.size else 0.0
        typical = max(step_left, step_right)
        if typical > 0 and dx[i] > factor * typical:
            sign = 1.0 if x[i + 1] > x[i] else -1.0
            at += [i + 1, i + 1]
            fill += [x[i] + sign * (step_left or typical), x[i + 1] - sign * (step_right or typical)]
    if not at:
        return (x, *arrays)
    return (np.insert(x, at, fill), *(np.insert(y, at, np.nan) for y in arrays))


def step(ax: Any, x: Any, y: Any, *, gap_factor: float | None = 4.0, **kwargs: Any) -> Any:
    """The standard line plot: ``ax.step(..., where="mid")`` (rule 3), broken at gaps.

    Non-finite samples break the line as usual; in addition the line is not
    drawn across gaps of the abscissa (:func:`break_gaps`; ``gap_factor=None``
    switches that off).
    """
    kwargs.setdefault("lw", LW["main"])
    if gap_factor:
        x, y = break_gaps(x, y, factor=gap_factor)
    return ax.step(np.asarray(x), np.asarray(y), where="mid", **kwargs)


def band(ax: Any, x: Any, lower: Any, upper: Any, *, color: Any = MAIN_COLOR, alpha: float = 0.22,
         gap_factor: float | None = 4.0, **kwargs: Any) -> Any:
    """A shaded band between ``lower`` and ``upper`` in the step style of :func:`step`
    (an uncertainty, a range), broken at gaps and non-finite samples like the line it belongs to."""
    if gap_factor:
        x, lower, upper = break_gaps(x, lower, upper, factor=gap_factor)
    kwargs.setdefault("lw", 0)
    return ax.fill_between(np.asarray(x), np.asarray(lower), np.asarray(upper), step="mid",
                           color=color, alpha=alpha, **kwargs)


def error_band(ax: Any, x: Any, y: Any, error: Any, **kwargs: Any) -> Any:
    """The 1-sigma band ``y +- error`` around a line (:func:`band`)."""
    y, error = np.asarray(y, dtype=float), np.asarray(error, dtype=float)
    return band(ax, x, y - error, y + error, **kwargs)


def refline(ax: Any, *, x: float | None = None, y: float | None = None, ls: str = "solid",
            label: str | None = None, **kwargs: Any) -> Any:
    """A reference line (zero, unity, a threshold, a crossover): thin, grey, behind the data."""
    kwargs.setdefault("color", REF_COLOR)
    kwargs.setdefault("lw", LW["reference"])
    kwargs.setdefault("zorder", 1)
    line = ax.axvline(x, ls=ls, **kwargs) if x is not None else ax.axhline(y, ls=ls, **kwargs)
    if label:
        ax.plot([], [], ls=ls, color=kwargs["color"], lw=kwargs["lw"], label=label)
    return line


def span(ax: Any, lo: float, hi: float, *, role: str = "window", alpha: float = 0.18, label: str | None = None,
         **kwargs: Any) -> Any:
    """A shaded wavelength window in the color of its ``role`` (:data:`ROLE_COLORS`), behind the data."""
    color = kwargs.pop("color", ROLE_COLORS[role])
    if role == "anchor":
        alpha = 1.0
    kwargs.setdefault("lw", 0)
    kwargs.setdefault("zorder", 0)
    patch = ax.axvspan(lo, hi, color=color, alpha=alpha, **kwargs)
    if label:
        ax.fill_between([], [], [], color=color, alpha=alpha, lw=0, label=label)
    return patch


def _plain_log_formatter(minor: bool, min_sep: float = 0.05) -> Any:
    from matplotlib.ticker import Formatter

    class PlainLog(Formatter):
        """Plain numbers (10^n outside 1e-3..1e5). Minor ticks are labelled greedily in the
        order 2, 5, 3, ... wherever they stay ``min_sep`` of the axis away from a labelled tick."""

        def __init__(self) -> None:
            self._labelled: set[float] = set()
            self._powers = False

        def text(self, x: float) -> str:
            if x <= 0:
                return ""
            if not self._powers and 1e-3 <= x < 1e5:
                return f"{x:.6g}"
            exponent = int(np.floor(np.log10(x) + 1e-9))
            mantissa = x / 10.0**exponent
            return rf"$10^{{{exponent}}}$" if abs(mantissa - 1) < 1e-6 else rf"${mantissa:g}\times10^{{{exponent}}}$"

        def set_locs(self, locs: Any) -> None:
            self._labelled = set()
            vmin, vmax = sorted(self.axis.get_view_interval())
            # one notation per axis: powers of ten throughout once the view leaves 0.001 .. 10^5
            self._powers = bool(vmin > 0 and (vmin < 1e-3 or vmax >= 1e5))
            if not minor or self._powers:
                return
            if vmin <= 0 or vmax <= vmin:
                return
            span_dex = np.log10(vmax / vmin)
            taken = [np.log10(v) for v in self.axis.get_majorticklocs() if vmin <= v <= vmax]
            candidates = [float(v) for v in locs if vmin <= v <= vmax and v > 0]

            def priority(v: float) -> int:
                mantissa = round(v / 10.0 ** np.floor(np.log10(v) + 1e-9), 6)
                order = [2.0, 5.0, 3.0, 4.0, 6.0, 8.0, 7.0, 9.0]
                return order.index(mantissa) if mantissa in order else len(order)

            for v in sorted(candidates, key=lambda v: (priority(v), v)):
                lv = np.log10(v)
                if all(abs(lv - t) >= min_sep * span_dex for t in taken):
                    taken.append(lv)
                    self._labelled.add(round(v, 12))

        def __call__(self, x: float, pos: Any = None) -> str:
            if minor and round(float(x), 12) not in self._labelled:
                return ""
            return self.text(float(x))

    return PlainLog()


def _plain_log_axis(axis: Any) -> None:
    """Plain-number tick labels on a log axis, now and -- once the view limits are final --
    linearly spaced ticks when the axis spans less than a factor of five."""
    axis.set_major_formatter(_plain_log_formatter(minor=False))
    axis.set_minor_formatter(_plain_log_formatter(minor=True))

    def finalize() -> None:
        from matplotlib.ticker import (
            FuncFormatter,
            MaxNLocator,
            MultipleLocator,
            NullFormatter,
        )

        if axis.get_scale() != "log":
            return
        vmin, vmax = sorted(axis.get_view_interval())
        if vmin <= 0 or np.log10(vmax / vmin) >= 0.7:
            return
        major = MaxNLocator(nbins=9, steps=[1, 2, 2.5, 5, 10])
        ticks = major.tick_values(vmin, vmax)
        axis.set_major_locator(major)
        if len(ticks) > 1:
            spacing = float(ticks[1] - ticks[0])
            mantissa = spacing / 10.0 ** np.floor(np.log10(spacing) + 1e-9)
            axis.set_minor_locator(MultipleLocator(spacing / (4 if abs(mantissa - 2.0) < 1e-6 else 5)))
        axis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.6g}"))
        axis.set_minor_formatter(NullFormatter())

    _defer(axis.axes.figure, finalize, early=True)


def set_wave_scale(ax: Any, scale: str = "log") -> None:
    """The standard wavelength axis of a spectrum: logarithmic by default (rule 9), with
    plain numbers (not powers of ten) as tick labels, placed so that they never crowd."""
    ax.set_xscale(scale)
    if scale == "log":
        _plain_log_axis(ax.xaxis)


def set_flux_scale(ax: Any, scale: str = "linear") -> None:
    """The y axis of a flux-like quantity: ``linear`` or ``log``, the latter with the same
    plain-number tick labels as the wavelength axis (``0.3``, ``1``, ``3`` -- not ``3 x 10^0``)."""
    ax.set_yscale(scale)
    if scale == "log":
        _plain_log_axis(ax.yaxis)
        _defer(ax.figure, lambda: _robust_log_ylim(ax), early=True)


def _robust_log_ylim(ax: Any, *, percentile: float = 0.5, margin: float = 1.6) -> None:
    """Automatic limits of a log y axis that single near-zero samples cannot stretch.

    On a log axis one noise sample close to zero adds decades of empty
    space below the data. Unless the limits were set explicitly, the lower
    limit becomes the ``percentile`` of the positive plotted values (over
    all lines, within the x view) divided by ``margin``, the upper limit
    their maximum times ``margin``.
    """
    if ax.get_yscale() != "log":
        return
    automatic = ax.get_autoscaley_on()
    x_lo, x_hi = sorted(ax.get_xlim())
    lows, highs = [], []
    for line in ax.lines:
        if line.get_gid() == "qafig-feature" or line.get_transform() != ax.transData:
            continue
        x, y = np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)
        if x.shape != y.shape or x.size < 2:
            continue
        ok = np.isfinite(x) & np.isfinite(y) & (y > 0) & (x >= x_lo) & (x <= x_hi)
        if ok.any():
            lows.append(float(np.percentile(y[ok], percentile)))
            highs.append(float(y[ok].max()))
    if not lows:
        # nothing positive to show (an empty or all-zero spectrum): a log axis cannot be drawn at all
        if not any(np.any(np.asarray(line.get_ydata(), dtype=float) > 0) for line in ax.lines):
            log.warning("no positive values to plot on a log axis; drawn on a linear one")
            ax.set_yscale("linear")
        return
    if not automatic or not max(highs) > min(lows):
        return
    ax.set_ylim(min(lows) / margin, max(highs) * margin)
    ax._qafig_auto_y = True   # still ours to extend (headroom for labels)


__all__ = [
    "band",
    "break_gaps",
    "error_band",
    "line_colors",
    "refline",
    "set_flux_scale",
    "set_wave_scale",
    "span",
    "step",
]
