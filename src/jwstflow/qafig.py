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
def collapse(cube: np.ndarray, how: str = "median", *, min_coverage: float = 0.0) -> np.ndarray:
    """Nan-aware collapse of a (nwave, ny, nx) cube along the spectral axis (rule 5).

    ``min_coverage`` blanks spaxels whose fraction of finite planes is below
    it: at the edges of an IFU footprint a nanmedian over a handful of planes
    is noise, not signal, and such spaxels otherwise dominate the display
    limits of the collapsed image.
    """
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
            image = reducers[how](data, axis=0)
    if min_coverage > 0:
        image = np.where(np.isfinite(data).mean(axis=0) >= min_coverage, image, np.nan)
    return image


def imshow(ax: Any, image: np.ndarray, *, unit: str | None = None, stretch: str = "linear",
           percentiles: tuple[float, float] = (1.0, 99.5), cmap: str = CMAP,
           cbar_label: str | None = None, cbar: bool = True, **kwargs: Any) -> Any:
    """The standard image panel: robust limits, ``origin="lower"``, :data:`CMAP`,
    and a height-matched colorbar (rule 4). Returns the ``AxesImage``.

    ``unit`` converts the data via :func:`to_mjy` and labels the colorbar with
    the result (``cbar_label`` overrides). ``stretch="log"`` uses a ``LogNorm``
    over the positive pixels, so the colorbar still reads in data units.

    The automatic limits are robust twice over: the upper percentile is
    additionally capped at the brightest *neighbourhood* of the image (the
    maximum of its 3x3 median filter), so isolated hot pixels never set
    ``vmax`` -- on a small image even the 99.9th percentile is essentially
    the single hottest pixel -- while real compact sources, several pixels
    wide, still do. Explicit ``vmin``/``vmax``/``norm`` switch all of it off.
    """
    from matplotlib.colors import LogNorm

    img, unit_label = to_mjy(image, unit)
    finite = img[np.isfinite(img)]
    with np.errstate(all="ignore"):
        if stretch == "log":
            positive = finite[finite > 0]
            if positive.size:
                vmin, vmax = np.percentile(positive, percentiles)
                cap = _robust_max(img)
                if cap is not None and cap > 0:
                    vmax = min(vmax, cap)
                vmin = max(vmin, vmax * 1e-5)
                if not vmax > vmin:
                    vmax = vmin * 10  # (near-)constant image: keep the norm well-formed
            else:
                vmin, vmax = 1e-3, 1.0
            kwargs.setdefault("norm", LogNorm(vmin=vmin, vmax=vmax))
        elif "norm" not in kwargs:
            lo, hi = np.percentile(finite, percentiles) if finite.size else (0.0, 1.0)
            cap = _robust_max(img)
            if cap is not None:
                hi = min(hi, cap)
            kwargs.setdefault("vmin", lo)
            kwargs.setdefault("vmax", hi if hi > lo else lo + 1.0)
    kwargs.setdefault("origin", "lower")
    kwargs.setdefault("interpolation", "nearest")
    im = ax.imshow(img, cmap=cmap, **kwargs)
    if cbar:
        colorbar(im, label=cbar_label if cbar_label is not None else unit_label)
    return im


def _robust_max(image: np.ndarray) -> float | None:
    """The brightest 3x3 *neighbourhood* median of a 2-D image (None when unusable).

    An isolated hot pixel cannot raise it, a real source -- bright over
    several adjacent pixels -- keeps its near-peak value; :func:`imshow` caps
    its automatic ``vmax`` here.
    """
    import warnings

    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or image.size < 9:
        return None
    padded = np.pad(image, 1, constant_values=np.nan)
    ny, nx = image.shape
    stack = np.stack([padded[dy:dy + ny, dx:dx + nx] for dy in range(3) for dx in range(3)])
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings("ignore", message=".*All-NaN.*")
        filtered = np.nanmedian(stack, axis=0)
        if not np.isfinite(filtered).any():
            return None
        return float(np.nanmax(filtered))


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


#: Colors of the spectral-feature annotation (:func:`annotate_features`): gas-line
#: ticks, the emission-band lane (PAH and other emission bands) and the ice-band
#: lane. Muted, colorblind-safe, distinct from the data palette.
FEATURE_COLORS: dict[str, str] = {"line": "0.35", "emission": "#d95f02", "ice": "#7570b3"}
#: Axes-fraction y layout of the annotation lanes (tick/fill bottom, top). All
#: labels hang from :data:`FEATURE_LABEL_Y`, below the lowest lane, so rotated
#: label text never crosses a lane; the class color ties a label to its lane.
FEATURE_LANES: dict[str, tuple[float, float]] = {
    "line": (0.958, 0.990), "emission": (0.915, 0.945), "ice": (0.872, 0.902),
}
FEATURE_LABEL_Y = 0.862


def annotate_features(ax: Any, features: Any, *, wave_min: float, wave_max: float,
                      line_width_um: float = 0.02, resolving_power: float | None = None,
                      max_line_labels: int = 150) -> None:
    """Mark the selected spectral features on a spectrum axes (x = wavelength [um]).

    ``features`` is a jwstflow feature selection (the ``features:`` grammar of
    :mod:`jwstflow.features`; ``"all"`` takes every bundled dataset), restricted
    to ``wave_min``-``wave_max`` (the plotted wavelength range). Three
    annotation classes, each in its own axes-fraction lane near the top so the
    spectrum below stays readable:

    * **gas lines** (``kind == "line"``): a short vertical tick with the line
      label rotated below it (labels are dropped, ticks kept, when more than
      ``max_line_labels`` lines are in range);
    * **emission bands** (``kind == "band"`` outside the ice dataset -- PAH
      bands, hydrocarbon quasi-continua, ...): a shaded wavelength span in the
      lane below the line ticks;
    * **ice bands** (the ``ice_bands`` dataset): a shaded span in a third lane.

    Empty selections draw nothing. One legend proxy per non-empty class is
    added, so :func:`figlegend` names the lanes. Spans crossing the plotted
    range are clipped to it, never widening the axes.
    """
    from .features import select_features

    if features in (None, False) or not np.isfinite([wave_min, wave_max]).all() or wave_max <= wave_min:
        return
    selected = select_features(features, wave_min=wave_min, wave_max=wave_max)
    lines = [f for f in selected if f.kind == "line"]
    bands = [f for f in selected if f.kind != "line"]
    ice = [f for f in bands if f.dataset == "ice_bands"]
    emission = [f for f in bands if f.dataset != "ice_bands"]
    trans = ax.get_xaxis_transform()  # x in data, y in axes fraction
    log_axis = ax.get_xscale() == "log"

    def centre(lo: float, hi: float) -> float:
        return float(np.sqrt(lo * hi)) if log_axis and lo > 0 else 0.5 * (lo + hi)

    def label_text(x: float, text: str, color: str) -> None:
        ax.text(x, FEATURE_LABEL_Y, text, transform=trans, rotation=90, ha="center", va="top",
                fontsize=4.5, color=color, clip_on=True, zorder=5)

    y0, y1 = FEATURE_LANES["line"]
    with_labels = len(lines) <= max_line_labels
    for f in lines:
        w = f.wavelength if f.wavelength is not None else centre(*f.window(line_width_um, resolving_power))
        if not wave_min <= w <= wave_max:
            continue
        ax.plot([w, w], [y0, y1], transform=trans, color=FEATURE_COLORS["line"], lw=0.5, zorder=5)
        if with_labels:
            label_text(w, f.label, FEATURE_COLORS["line"])
    if lines and not with_labels:
        log.info("%d gas lines in range (> %d): ticks drawn without labels", len(lines), max_line_labels)
    for group, kind in ((emission, "emission"), (ice, "ice")):
        y0, y1 = FEATURE_LANES[kind]
        color = FEATURE_COLORS[kind]
        for f in group:
            lo, hi = f.window(line_width_um, resolving_power)
            lo, hi = max(lo, wave_min), min(hi, wave_max)
            if hi <= lo:
                continue
            ax.fill_betweenx([y0, y1], lo, hi, transform=trans, color=color, alpha=0.30, lw=0, zorder=4)
            label_text(centre(lo, hi), f.label, color)
    for group, kind, name in ((lines, "line", "gas lines"), (emission, "emission", "emission bands"),
                              (ice, "ice", "ice bands")):
        if group:
            if kind == "line":
                ax.plot([], [], color=FEATURE_COLORS[kind], lw=0.8, label=name)
            else:
                ax.fill_between([], [], [], color=FEATURE_COLORS[kind], alpha=0.30, label=name)


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
