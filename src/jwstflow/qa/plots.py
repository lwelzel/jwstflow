"""The two kinds of QA figure, as classes to build on.

Almost every QA figure is one of two things:

* a **spectrum-like** plot -- a flux-like quantity over wavelength on
  logarithmic axes, its 1-sigma band, the spectral features annotated
  (:class:`Spectrum`);
* a **frame-like** panel -- an image on its pixel grid with pixel and sky
  offsets from one common zero, a colorbar, masks and markers on top
  (:class:`Frame`).

A step builds its figure from these (or from a subclass that adds what its
product needs) instead of repeating the calls; that keeps the figures of all
packages alike. ``docs/qa_figures.md`` shows a worked step for each.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .annotation import DETECTED_NOTE, annotate_features
from .frames import SkyFrame, contour_mask, imshow, marker
from .layout import FEATURE_LABEL_OFFSET
from .lines import error_band, refline, set_flux_scale, set_wave_scale, step
from .style import (
    FIGSIZE,
    LW,
    MAIN_COLOR,
    OVERLAY_COLORS,
    WAVE_LABEL,
    annotate,
    figlegend,
    save,
    subplots,
)


class Spectrum:
    """A spectrum-like figure: lines over wavelength, annotated with the spectral features.

    Stands alone (its own figure) or on an axes of a larger figure (``ax``).
    ``panels`` > 1 stacks further axes under the spectrum on the same
    wavelength axis (a ratio, a residual): ``ax`` is the spectrum, ``axes``
    all of them. Draw with :meth:`line`; :meth:`decorate` labels and scales
    the axes and annotates the features; :meth:`finish` (for a figure of its
    own) does that, sets notes and legend, and saves. Both axes are
    logarithmic unless told otherwise.
    """

    def __init__(self, ax: Any = None, *, xscale: str = "log", yscale: str = "log", panels: int = 1,
                 height_ratios: list[float] | None = None, figsize: tuple[float, float] | None = None) -> None:
        if ax is None:
            self.fig, axes = subplots(panels, 1, sharex=True, squeeze=False,
                                      figsize=figsize or FIGSIZE["spectrum" if panels == 1 else "spectrum2"],
                                      gridspec_kw={"height_ratios": height_ratios} if height_ratios else None)
            self.axes: list[Any] = list(axes[:, 0])
        else:
            self.fig, self.axes = ax.figure, [ax]
        self.ax = self.axes[0]
        self.xscale, self.yscale = xscale, yscale
        self._range: list[tuple[float, float]] = []

    def line(self, wave: Any, values: Any, *, error: Any = None, ax: Any = None, color: Any = MAIN_COLOR,
             label: str | None = None, error_label: str | None = None, context: bool = False, **kwargs: Any) -> None:
        """A mid-point step line (broken at gaps), with its 1-sigma band when ``error`` has any.

        ``context`` marks a line drawn for comparison under the product (an
        input, a segment): thin, and not counted for the annotated range.
        """
        target = self.ax if ax is None else ax
        wave = np.asarray(wave, dtype=float)
        if context:
            kwargs.setdefault("lw", LW["thin"])
        if error is not None:
            error = np.asarray(error, dtype=float)
            if np.isfinite(error).any() and np.nanmax(error) > 0:
                error_band(target, wave, values, error, color=color, label=error_label)
        step(target, wave, values, color=color, label=label, **kwargs)
        if target is self.ax and not context:
            self.cover(wave)

    def cover(self, wave: Any) -> None:
        """Count ``wave`` for the wavelength range the features are annotated over (every line that
        is not ``context`` does so itself)."""
        wave = np.asarray(wave, dtype=float)
        if np.isfinite(wave).any():
            self._range.append((float(np.nanmin(wave)), float(np.nanmax(wave))))

    def crossovers(self, wavelengths: Any) -> None:
        """Dotted vertical lines where a combination switches from one segment to the next."""
        for k, wavelength in enumerate(wavelengths):
            refline(self.ax, x=float(wavelength), ls=":", label="crossover" if k == 0 else None)

    def decorate(self, *, ylabel: str, features: Any = None, label_offset: float = FEATURE_LABEL_OFFSET,
                 photospheric: bool = True, series: str = "labels", follow: tuple[Any, Any] | None = None,
                 ylim: Any = None, wave_range: tuple[float, float] | None = None, detected_only: bool = False) -> None:
        """Label and scale the axes and annotate ``features`` (jwstflow ``features:`` grammar; None
        for no annotation) over the plotted range. ``wave_range`` zooms the wavelength axis.

        The labels follow everything plotted, and the lines of the product
        (black) are searched for peaks. ``follow`` -- ``(wavelength, values)``
        -- names the curve to use instead, for a figure whose product is not
        the curve that carries the lines (a continuum under its spectrum).
        Every line in range is marked; ``detected_only`` marks only the
        lines the spectrum shows (:func:`annotate_features`).
        """
        self.axes[-1].set_xlabel(WAVE_LABEL)
        self.ax.set_ylabel(ylabel)
        if wave_range is not None:
            self.axes[-1].set_xlim(*wave_range)
        set_wave_scale(self.axes[-1], self.xscale)
        set_flux_scale(self.ax, self.yscale)
        if ylim:
            self.ax.set_ylim(*ylim)
        if features not in (None, False) and (self._range or wave_range is not None):
            lo, hi = wave_range or (min(r[0] for r in self._range), max(r[1] for r in self._range))
            annotate_features(self.ax, features, wave_min=lo, wave_max=hi, label_offset=label_offset,
                              photospheric=photospheric, series=series, under=follow, detected_only=detected_only)

    def finish(self, path: Path | str, *, ylabel: str, notes: tuple[str, ...] | list[str] = (), dpi: int = 150,
               **decoration: Any) -> Path:
        """:meth:`decorate` (which takes ``decoration``), then the notes and the legend; save and close.
        A figure that marks only the detected lines says so in its notes."""
        self.decorate(ylabel=ylabel, **decoration)
        detected = decoration.get("detected_only") and decoration.get("features") not in (None, False)
        annotate(self.ax, *notes, *([DETECTED_NOTE] if detected else []))
        figlegend(self.fig)
        return save(self.fig, path, dpi=dpi)


class Frame:
    """A frame-like panel: an image with pixel and sky offsets from one zero, and overlays.

    Stands alone (its own figure) or on an axes of a larger figure (``ax``).
    :meth:`image` draws the image with its colorbar; :meth:`mask` and
    :meth:`mark` add outlines and positions in the overlay colors;
    :meth:`finish` (for a figure of its own) sets notes and legend and saves.
    """

    def __init__(self, ax: Any = None, *, figsize: tuple[float, float] | None = None) -> None:
        if ax is None:
            self.fig, self.ax = subplots(figsize=figsize or FIGSIZE["image"])
        else:
            self.fig, self.ax = ax.figure, ax
        self.sky: SkyFrame | None = None
        self.im: Any = None
        self._overlays = 0

    def image(self, image: Any, *, unit: str | None = None, sky: SkyFrame | None = None, stretch: str = "asinh",
              **kwargs: Any) -> Any:
        """The image (:func:`imshow`: robust limits, ``stretch``, NaN in grey, colorbar) on its two
        sets of axes: pixel offsets and sky offsets from the reference position of ``sky``."""
        self.sky = sky
        self.im = imshow(self.ax, image, unit=unit, sky=sky, stretch=stretch, **kwargs)
        return self.im

    def mask(self, mask: Any, *, label: str | None = None, ls: str | None = None, color: str | None = None) -> None:
        """The outline of a spaxel mask: solid in the first overlay color, the next ones dashed."""
        color = color or OVERLAY_COLORS[self._overlays % len(OVERLAY_COLORS)]
        contour_mask(self.ax, mask, color, ls=ls or ("solid" if self._overlays == 0 else "dashed"), label=label)
        self._overlays += 1

    def mark(self, x: float, y: float, *, label: str | None = None, symbol: str = "+") -> None:
        """A position marker (pixel coordinates); ``symbol`` tells kinds of source apart."""
        marker(self.ax, x, y, label=label, symbol=symbol)

    def note(self) -> str:
        """Where the offsets of this frame are measured from (empty without a sky frame)."""
        return self.sky.note() if self.sky is not None else ""

    def finish(self, path: Path | str, *, notes: tuple[str, ...] | list[str] = (), dpi: int = 150) -> Path:
        """Notes (closing with the reference position of the offsets), legend, save and close."""
        annotate(self.ax, *notes, self.note())
        figlegend(self.fig)
        return save(self.fig, path, dpi=dpi)


__all__ = ["Frame", "Spectrum"]
