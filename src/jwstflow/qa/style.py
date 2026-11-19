"""The look of every QA figure: palette, typography, axis-label text, the figure
factory, and the legend that replaces titles.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- palette (rule 6)
#: The standard image colormap (cmasher's *torch*; registered on import as ``cmr.torch``).
CMAP = "cmr.torch"
#: Colormap of signed images (differences, residuals), used with ``imshow(..., center=0)``.
DIVERGING_CMAP = "cmr.fusion_r"
#: Color of pixels without data (NaN) in every image -- a grey no colormap contains.
BAD_COLOR = "0.80"
#: Color of the only line of a plot, or of the main product among several.
MAIN_COLOR = "black"
#: Color of the input / context a main product is drawn over (e.g. the flux under its continuum).
CONTEXT_COLOR = "0.62"
#: Reference lines (zero, unity, thresholds, crossovers): thin and recessive.
REF_COLOR = "0.55"
#: Categorical accents for marks on a white background, in this fixed order
#: (Okabe-Ito; adjacent pairs stay distinguishable under color-vision deficiency).
ACCENTS: tuple[str, ...] = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9")
#: Overlay colors (contours, aperture circles, markers) with contrast on :data:`CMAP` images.
OVERLAY_COLORS: tuple[str, ...] = ("cyan", "lime", "magenta", "white")
#: Colors of the spectral-feature classes (:func:`annotate_features`): gas lines,
#: emission bands (PAH and other emission bands), ice bands and the absorption lines
#: of the stellar photosphere. Muted, colorblind-safe, distinct from :data:`ACCENTS`.
FEATURE_COLORS: dict[str, str] = {"line": "0.35", "emission": "#d95f02", "ice": "#7570b3",
                                  "photosphere": "#1b9e77"}
#: Marks with a meaning keep one color in every figure.
ROLE_COLORS: dict[str, str] = {
    "main": MAIN_COLOR,            # the product
    "context": CONTEXT_COLOR,      # its input, drawn underneath
    "reference": REF_COLOR,        # zero / unity / threshold / crossover lines
    "model": ACCENTS[1],           # a model or fit over the data
    "background": "#8c498c",       # sky / background estimates (the middle of the standard colormap)
    "excluded": ACCENTS[4],        # samples left out of a fit
    "censored": ACCENTS[2],        # censored / lower-limit samples
    "bridged": ACCENTS[5],         # interpolated or prior-bridged stretches
    "window": ACCENTS[3],          # a measurement window (shaded)
    "anchor": "0.86",              # anchor / fit stretches (shaded, behind the data)
    "ice": FEATURE_COLORS["ice"],
    "emission": FEATURE_COLORS["emission"],
    "line": FEATURE_COLORS["line"],
    "photosphere": FEATURE_COLORS["photosphere"],
}

# --------------------------------------------------------------------------- typography (rule 12)
#: Font sizes [pt] of every QA figure. ``base`` is the axis-label size; steps use the
#: helpers of this module (which apply these) instead of passing ``fontsize``.
FONT: dict[str, float] = {"base": 9.0, "tick": 8.0, "legend": 8.0, "note": 7.0, "feature": 5.5, "comb": 4.8}
#: Line widths [pt]: the main line, context/secondary lines, reference lines, overlays on images.
LW: dict[str, float] = {"main": 0.9, "thin": 0.6, "reference": 0.6, "overlay": 1.2}
#: The matplotlib rcParams of the standard, applied by :func:`use_agg`.
STYLE: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.size": FONT["base"],
    "axes.labelsize": FONT["base"],
    "axes.titlesize": FONT["base"],
    "xtick.labelsize": FONT["tick"],
    "ytick.labelsize": FONT["tick"],
    "legend.fontsize": FONT["legend"],
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.8,
    "axes.unicode_minus": True,
    "lines.linewidth": LW["main"],
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "image.origin": "lower",
    "image.interpolation": "nearest",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 100,
}
#: Standard figure sizes [inch]: a spectrum over the full width, a spectrum with a
#: second panel below, a single image panel with its colorbar.
FIGSIZE: dict[str, tuple[float, float]] = {
    "spectrum": (11.0, 4.0), "spectrum2": (11.0, 6.5), "image": (5.8, 5.0), "square": (5.2, 5.0),
}

# --------------------------------------------------------------------------- labels (rule 8)
#: Micrometre as mathtext, for labels and legend text.
UM = r"$\mathrm{\mu m}$"


def label(quantity: str, unit: str | None = None) -> str:
    """An axis label in the standard form ``quantity [unit]`` (rule 8); no brackets without a unit."""
    return f"{quantity} [{unit}]" if unit else quantity


def um(value: float, fmt: str = ".2f") -> str:
    """A wavelength with its unit for legend and annotation text: ``um(4.05)`` -> ``4.05 µm`` (mathtext)."""
    return f"{value:{fmt}} {UM}"


def um_range(lo: float, hi: float, fmt: str = ".2f") -> str:
    """A wavelength range with its unit: ``4.05-4.06 µm`` (en dash, mathtext unit)."""
    return f"{lo:{fmt}}\N{EN DASH}{hi:{fmt}} {UM}"


def times(value: float, fmt: str = ".3f") -> str:
    r"""A multiplicative factor for legend text: ``times(1.021)`` -> ``$\times$ 1.021``."""
    return rf"$\times$ {value:{fmt}}"


def pm(value: float, error: float, fmt: str = ".3f", efmt: str = ".2g") -> str:
    r"""A value with its 1-sigma uncertainty: ``1.021 $\pm$ 0.0016``."""
    return rf"{value:{fmt}} $\pm$ {error:{efmt}}"


#: Axis labels name a quantity by its symbol (rule 16): ``F_nu [mJy]``, not "flux density".
WAVE_LABEL = label(r"$\lambda$", UM)
FLUX_LABEL = label(r"$F_\nu$", "mJy")
FLUX_ERROR_LABEL = label(r"$\sigma_{F_\nu}$", "mJy")
SB_LABEL = label(r"$I_\nu$", r"mJy arcsec$^{-2}$")
X_PIX_LABEL = label("$x$", "pix")
Y_PIX_LABEL = label("$y$", "pix")
DRA_LABEL = label(r"$\Delta\alpha\,\cos\delta$", "arcsec")
DDEC_LABEL = label(r"$\Delta\delta$", "arcsec")
DX_LABEL = label(r"$\Delta x$", "arcsec")
DY_LABEL = label(r"$\Delta y$", "arcsec")
#: Pixel axes of a frame that also has sky offsets: offsets in pixels from the same zero.
DX_PIX_LABEL = label(r"$\Delta x$", "pix")
DY_PIX_LABEL = label(r"$\Delta y$", "pix")
TAU_LABEL = r"$\tau$"
SNR_LABEL = r"$F_\nu\,/\,\sigma_{F_\nu}$"
SIGMA_LABEL = r"$\Delta F_\nu\,/\,\sigma_{F_\nu}$"
SIGMA_BAND_LABEL = r"$\pm 1\sigma$"


def use_agg() -> Any:
    """Headless matplotlib in the standard's style, cmasher colormaps registered; returns ``pyplot``.

    Every QA step calls this (directly or through :func:`subplots` /
    :func:`figure`) instead of importing matplotlib itself. It applies
    :data:`STYLE` to the process-wide rcParams -- QA figures are made in
    worker processes that do nothing else, so one global style is what
    keeps them consistent.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import cmasher  # noqa: F401  (registers the cmr.* colormaps)
        import matplotlib.pyplot as plt

        matplotlib.rcParams.update(STYLE)
        return plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("QA figure steps need `pip install matplotlib cmasher`") from exc


def subplots(nrows: int = 1, ncols: int = 1, *, figsize: tuple[float, float] | None = None,
             **kwargs: Any) -> tuple[Any, Any]:
    """``plt.subplots`` in the standard's style and constrained layout (needed for outside legends)."""
    plt = use_agg()
    kwargs.setdefault("layout", "constrained")
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or FIGSIZE["spectrum"], **kwargs)
    _pad_layout(fig)
    return fig, axes


def figure(figsize: tuple[float, float] | None = None, **kwargs: Any) -> Any:
    """``plt.figure`` in the standard's style and constrained layout, for hand-built grids
    (``fig.add_gridspec`` / ``fig.subplot_mosaic``)."""
    plt = use_agg()
    kwargs.setdefault("layout", "constrained")
    fig = plt.figure(figsize=figsize or FIGSIZE["spectrum"], **kwargs)
    _pad_layout(fig)
    return fig


def _pad_layout(fig: Any) -> None:
    """The standard's spacing between panels (and between a colorbar label and the next panel)."""
    engine = fig.get_layout_engine()
    if engine is not None and hasattr(engine, "set"):
        try:
            engine.set(w_pad=0.10, h_pad=0.07, wspace=0.02, hspace=0.02)
        except (TypeError, ValueError):   # not the constrained engine
            pass


# --------------------------------------------------------------------------- annotation (rule 1)
def annotate(ax: Any, *texts: str) -> None:
    """Note lines instead of a title (rule 1): the product name, counts, parameters.

    The notes of all axes of a figure are collected and :func:`figlegend`
    sets them as the first, handle-less column of the legend.
    """
    fig = ax.figure
    notes = fig.__dict__.setdefault("_qafig_notes", [])
    for text in texts:
        if text and str(text) not in notes:
            notes.append(str(text))


def figlegend(fig: Any, *, ncol: int | None = None, **kwargs: Any) -> Any:
    """One legend for the whole figure, outside above the axes (rule 1).

    Collects the labelled artists of every axes (duplicates dropped), so
    subplots never need their own legend, and puts the notes recorded with
    :func:`annotate` in front of them as a column of plain text. The number
    of columns is chosen so that the legend fits the figure width (``ncol``
    fixes the number of *mark* columns); notes too long for it are wrapped.
    Requires the constrained layout that :func:`subplots` sets up.
    """
    from matplotlib.lines import Line2D

    handles: list[Any] = []
    labels: list[str] = []
    for ax in fig.axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    notes = list(fig.__dict__.get("_qafig_notes", []))
    if not handles and not notes:
        return None
    fontsize = float(kwargs.setdefault("fontsize", FONT["legend"]))
    char = 0.56 * fontsize / 72.0                 # mean advance of the sans font [inch]
    slot = 4.2 * fontsize / 72.0                  # handle + paddings + column gap [inch]
    available = 0.96 * fig.get_figwidth()
    note_chars = max((len(_plain(n)) for n in notes), default=0)

    def mark_width(k: int) -> float:
        if not labels:
            return 0.0
        rows = -(-len(labels) // k)
        cols = [labels[i * rows:(i + 1) * rows] for i in range(k)]
        return sum(slot + char * max((len(_plain(t)) for t in col), default=0) for col in cols if col)

    k = max(1, min(int(ncol) if ncol else 4, len(labels))) if labels else 0
    if notes:
        min_notes = slot + char * min(note_chars, 48)
        while k > 1 and mark_width(k) + min_notes > available:
            k -= 1
        room = int(max(available - mark_width(k) - slot, 0.4 * available) / char) if k else int(available / char)
        notes = [line for n in notes
                 for line in (textwrap.wrap(n, max(room, 24), break_long_words=False, break_on_hyphens=False)
                              if len(_plain(n)) > room else [n])]
    else:
        while k > 1 and mark_width(k) > available:
            k -= 1
    blank = Line2D([], [], linestyle="none")
    rows = max(-(-len(labels) // k) if k else 0, 1)
    entries_h: list[Any] = []
    entries_l: list[str] = []
    if notes:
        note_rows = notes[:rows - 1] + ["\n".join(notes[rows - 1:])] if len(notes) > rows else notes
        note_rows = note_rows + [""] * (rows - len(note_rows))
        entries_h += [blank] * rows
        entries_l += note_rows
    entries_h += handles + [blank] * (rows * k - len(handles))
    entries_l += labels + [""] * (rows * k - len(labels))
    kwargs.setdefault("loc", "outside upper center")
    kwargs.setdefault("ncol", k + (1 if notes else 0))
    kwargs.setdefault("frameon", False)
    legend = fig.legend(entries_h, entries_l, **kwargs)
    if notes:
        for text in legend.get_texts()[:rows]:
            text.set_color("0.25")
    return legend


def _plain(text: str) -> str:
    """Rough rendered text of a label (mathtext markup stripped), for width estimates."""
    import re

    return re.sub(r"\\[a-zA-Z]+|[${}^_]", "", str(text))


# --------------------------------------------------------------------------- deferred layout
def _defer(fig: Any, func: Any, *, early: bool = False) -> None:
    """Run ``func()`` when the figure is saved (:func:`finalize`): ``early`` callbacks before
    the layout is computed (axis scales), the others after it (label placement)."""
    fig.__dict__.setdefault("_qafig_early" if early else "_qafig_late", []).append(func)


def finalize(fig: Any) -> None:
    """Carry out the layout decisions that need the final geometry: tick placement of narrow
    log axes, spreading of feature labels and callouts, headroom above the data.
    :func:`save` calls it; call it yourself only when saving a figure some other way."""
    early = fig.__dict__.pop("_qafig_early", [])
    late = fig.__dict__.pop("_qafig_late", [])
    for func in early:
        func()
    if late:
        fig.draw_without_rendering()
        for func in late:
            func()


# --------------------------------------------------------------------------- output
def save(fig: Any, path: Path | str, *, dpi: int = 150) -> Path:
    """Save a figure the standard way and close it: the deferred layout is carried out
    (:func:`finalize`), then the figure is written with a tight bounding box, so outside
    legends, colorbars and their labels are never clipped."""
    plt = use_agg()
    path = Path(path)
    finalize(fig)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


__all__ = [
    "ACCENTS",
    "BAD_COLOR",
    "CMAP",
    "CONTEXT_COLOR",
    "DDEC_LABEL",
    "DIVERGING_CMAP",
    "DRA_LABEL",
    "DX_LABEL",
    "DX_PIX_LABEL",
    "DY_LABEL",
    "DY_PIX_LABEL",
    "FEATURE_COLORS",
    "FIGSIZE",
    "FLUX_ERROR_LABEL",
    "FLUX_LABEL",
    "FONT",
    "LW",
    "MAIN_COLOR",
    "OVERLAY_COLORS",
    "REF_COLOR",
    "ROLE_COLORS",
    "SB_LABEL",
    "SIGMA_BAND_LABEL",
    "SIGMA_LABEL",
    "SNR_LABEL",
    "STYLE",
    "TAU_LABEL",
    "UM",
    "WAVE_LABEL",
    "X_PIX_LABEL",
    "Y_PIX_LABEL",
    "annotate",
    "figlegend",
    "figure",
    "finalize",
    "label",
    "pm",
    "save",
    "subplots",
    "times",
    "um",
    "um_range",
    "use_agg",
]
