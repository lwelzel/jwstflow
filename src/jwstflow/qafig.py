"""The jwstflow QA figure standard.

Every QA figure -- in jwstflow itself and in contributed step packages -- is
built through this module, so all of them share one look. The rules
(``docs/qa_figures.md`` spells them out with examples):

1.  **No titles.** Not on figures, not on subplots. Whatever a title would
    have said goes into the legend: :func:`annotate` adds hand-less entries
    (the product name, counts, parameters) and :func:`figlegend` places one
    combined legend outside the axes.
2.  **Flux-like quantities are plotted in mJy** (surface brightness in
    mJy arcsec^-2). :func:`to_mjy` converts from the units products carry
    (Jy, MJy, MJy/sr, ...) and hands back the axis-label text.
3.  **Line-style plots are drawn with** ``ax.step(..., where="mid")`` --
    spectra are histograms over wavelength bins, not smooth curves.
    :func:`step` wraps it.
4.  **Every image gets a colorbar** whose height exactly matches the image
    axes (:func:`colorbar` -- an inset axes glued to the parent, so the match
    is exact for any aspect ratio).
5.  **Spectral collapses are nan-aware**: :func:`collapse` uses
    ``nanmedian``/``nanmean``, never ``median``/``mean``.
6.  **One palette everywhere**: images use cmasher's ``torch`` colormap
    (:data:`CMAP`); the only/main line of a plot is black (:data:`MAIN_COLOR`);
    families of lines get :func:`line_colors` (sampled from the same
    colormap); overlays on images (contours, apertures, markers) cycle
    through :data:`OVERLAY_COLORS`, chosen for contrast on ``torch``.
7.  **Figures come from QA steps only** (``level = "qa"``), never from data
    steps, so every figure lands in ``qa/<step name>/`` and the directory
    names the step that made it.
8.  **Axis labels always carry their unit in square brackets**:
    ``wavelength [um]``, ``flux density [mJy]``, ``x [pix]``.
9.  **Spectra use a logarithmic wavelength axis by default**
    (:func:`set_wave_scale`, plain numbers as tick labels); steps expose an
    ``xscale`` parameter to switch a figure back to linear.

Matplotlib is imported lazily (through :func:`use_agg`), so importing this
module stays cheap in the scheduler process.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

#: The standard image colormap (cmasher's *torch*; registered on import as ``cmr.torch``).
CMAP = "cmr.torch"
#: Color of the only line of a plot, or of the main product among several.
MAIN_COLOR = "black"
#: Overlay colors (contours, aperture circles, markers) with contrast on :data:`CMAP` images.
OVERLAY_COLORS: tuple[str, ...] = ("cyan", "lime", "deepskyblue", "magenta")
#: Axis-label text for flux densities and surface brightness (rule 2 + rule 8).
FLUX_LABEL = "flux density [mJy]"
SB_LABEL = "surface brightness [mJy arcsec$^{-2}$]"
WAVE_LABEL = "wavelength [um]"

#: 1 MJy/sr expressed in mJy per arcsec^2.
MJY_ARCSEC2_PER_MJY_SR = 1e9 / (np.degrees(1.0) * 3600.0) ** 2


def use_agg() -> Any:
    """Headless matplotlib with the cmasher colormaps registered; returns ``pyplot``.

    Every QA step calls this instead of importing matplotlib itself.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import cmasher  # noqa: F401  (registers the cmr.* colormaps)
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("QA figure steps need `pip install matplotlib cmasher`") from exc


def subplots(nrows: int = 1, ncols: int = 1, *, figsize: tuple[float, float] | None = None,
             **kwargs: Any) -> tuple[Any, Any]:
    """``plt.subplots`` with the standard's constrained layout (needed for outside legends)."""
    plt = use_agg()
    kwargs.setdefault("layout", "constrained")
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)


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


# --------------------------------------------------------------------------- lines (rules 3 + 6)
def line_colors(n: int) -> list[Any]:
    """Colors for a family of ``n`` lines: black when there is only one, otherwise
    ``n`` colors sampled from :data:`CMAP` (consistent across all QA figures)."""
    if n <= 1:
        return [MAIN_COLOR]
    import cmasher as cmr

    return list(cmr.take_cmap_colors(CMAP, n, cmap_range=(0.15, 0.80)))


def step(ax: Any, x: Any, y: Any, **kwargs: Any) -> Any:
    """The standard line plot: ``ax.step(..., where="mid")`` (rule 3)."""
    kwargs.setdefault("lw", 0.8)
    return ax.step(np.asarray(x), np.asarray(y), where="mid", **kwargs)


def set_wave_scale(ax: Any, scale: str = "log") -> None:
    """The standard wavelength axis of a spectrum: logarithmic by default (rule 9),
    with plain numbers (not powers of ten) on both major and minor ticks."""
    from matplotlib.ticker import ScalarFormatter

    ax.set_xscale(scale)
    if scale == "log":
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        minor = ScalarFormatter(useOffset=False)
        minor.set_scientific(False)
        ax.xaxis.set_minor_formatter(minor)
        ax.tick_params(axis="x", which="minor", labelsize=7)


# --------------------------------------------------------------------------- images (rules 4-6)
def collapse(cube: np.ndarray, how: str = "median") -> np.ndarray:
    """Nan-aware collapse of a (nwave, ny, nx) cube along the spectral axis (rule 5)."""
    reducers = {"median": np.nanmedian, "mean": np.nanmean, "sum": np.nansum, "max": np.nanmax}
    data = np.asarray(cube, dtype=float)
    while data.ndim > 3:
        data = data[0]
    if data.ndim == 2:
        return data
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*(empty slice|All-NaN).*")
            return reducers[how](data, axis=0)


def imshow(ax: Any, image: np.ndarray, *, unit: str | None = None, stretch: str = "linear",
           percentiles: tuple[float, float] = (1.0, 99.5), cmap: str = CMAP,
           cbar_label: str | None = None, cbar: bool = True, **kwargs: Any) -> Any:
    """The standard image panel: robust limits, ``origin="lower"``, :data:`CMAP`,
    and a height-matched colorbar (rule 4). Returns the ``AxesImage``.

    ``unit`` converts the data via :func:`to_mjy` and labels the colorbar with
    the result (``cbar_label`` overrides). ``stretch="log"`` uses a ``LogNorm``
    over the positive pixels, so the colorbar still reads in data units.
    """
    from matplotlib.colors import LogNorm

    img, unit_label = to_mjy(image, unit)
    finite = img[np.isfinite(img)]
    with np.errstate(all="ignore"):
        if stretch == "log":
            positive = finite[finite > 0]
            if positive.size:
                vmin, vmax = np.percentile(positive, percentiles)
                vmin = max(vmin, vmax * 1e-5)
            else:
                vmin, vmax = 1e-3, 1.0
            kwargs.setdefault("norm", LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10)))
        elif "norm" not in kwargs:
            lo, hi = np.percentile(finite, percentiles) if finite.size else (0.0, 1.0)
            kwargs.setdefault("vmin", lo)
            kwargs.setdefault("vmax", hi if hi > lo else lo + 1.0)
    kwargs.setdefault("origin", "lower")
    kwargs.setdefault("interpolation", "nearest")
    im = ax.imshow(img, cmap=cmap, **kwargs)
    if cbar:
        colorbar(im, label=cbar_label if cbar_label is not None else unit_label)
    return im


def colorbar(im: Any, *, label: str | None = None, width: float = 0.045, pad: float = 0.02) -> Any:
    """A colorbar exactly as tall as the image axes (an inset axes glued to its right edge)."""
    ax = im.axes
    cax = ax.inset_axes((1.0 + pad, 0.0, width, 1.0))
    cbar = ax.figure.colorbar(im, cax=cax)
    if label:
        cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    return cbar


# --------------------------------------------------------------------------- annotation (rule 1)
def annotate(ax: Any, *texts: str) -> None:
    """Legend-borne annotation instead of a title (rule 1): each ``text`` becomes a
    legend entry without a handle (the product name, counts, parameters, ...)."""
    for text in texts:
        if text:
            ax.plot([], [], linestyle="none", label=str(text))


def figlegend(fig: Any, *, ncol: int | None = None, **kwargs: Any) -> Any:
    """One legend for the whole figure, outside above the axes (rule 1).

    Collects the labelled artists of every axes (duplicates dropped), so
    subplots never need their own legend. Requires the constrained layout
    that :func:`subplots` sets up.
    """
    handles: list[Any] = []
    labels: list[str] = []
    for ax in fig.axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    if not handles:
        return None
    kwargs.setdefault("loc", "outside upper center")
    kwargs.setdefault("ncol", ncol or min(3, len(labels)))
    kwargs.setdefault("frameon", False)
    kwargs.setdefault("fontsize", 8)
    return fig.legend(handles, labels, **kwargs)


def contour_proxy(ax: Any, color: str, *, ls: str = "solid", label: str | None = None) -> Any:
    """A legend handle for a contour set (matplotlib's own contours make poor handles);
    added to ``ax`` so :func:`figlegend` picks it up."""
    from matplotlib.lines import Line2D

    handle = Line2D([], [], color=color, lw=1.2, linestyle=ls, label=label)
    ax.add_line(handle)
    return handle


# --------------------------------------------------------------------------- output
def save(fig: Any, path: Path | str, *, dpi: int = 150) -> Path:
    """Save a figure the standard way (tight bounding box, so outside legends and
    colorbar labels are never clipped) and close it."""
    plt = use_agg()
    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
