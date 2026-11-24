"""Frame-like panels: images on their pixel grid with sky offsets, robust display
limits and stretches, colorbars, and overlays.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from .lines import _plain_log_axis
from .style import (
    BAD_COLOR,
    CMAP,
    DDEC_LABEL,
    DIVERGING_CMAP,
    DRA_LABEL,
    DX_LABEL,
    DX_PIX_LABEL,
    DY_LABEL,
    DY_PIX_LABEL,
    FONT,
    LW,
    X_PIX_LABEL,
    Y_PIX_LABEL,
    use_agg,
)
from .units import to_mjy

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- sky geometry (rule 11)
@dataclass(frozen=True)
class SkyFrame:
    """How the pixels of an image map onto sky offsets -- what the arcsec axes of an image panel need.

    ``x0, y0`` is the reference pixel (0-based) the offsets are measured
    from, ``sx, sy`` the offset [arcsec] per pixel along +x and +y. For an
    image aligned with the sky (``north_up``; +y within a degree of North)
    these are the RA and Dec offsets, ``sx`` negative when East is to the
    left; for any other orientation they are offsets along the image axes
    and ``north`` / ``east`` give the sky directions as pixel unit vectors
    (:func:`compass` draws them).
    """

    x0: float
    y0: float
    sx: float
    sy: float
    north_up: bool = False
    north: tuple[float, float] | None = None
    east: tuple[float, float] | None = None
    ra: float | None = None
    dec: float | None = None
    reference: str = "image centre"

    @classmethod
    def from_wcs(cls, wcs: Any, shape: tuple[int, ...], *, ref_world: tuple[float, float] | None = None,
                 reference: str = "") -> SkyFrame:
        """From a celestial ``astropy.wcs.WCS`` and the image ``shape`` (ny, nx).

        ``ref_world`` (RA, Dec in degrees) sets the reference position --
        typically the target; it falls back to the image centre when absent
        or outside the image.
        """
        ny, nx = int(shape[-2]), int(shape[-1])
        x0, y0, name = (nx - 1) / 2.0, (ny - 1) / 2.0, "image centre"
        if ref_world is not None and np.isfinite(ref_world).all():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    px, py = (float(v) for v in wcs.all_world2pix(ref_world[0], ref_world[1], 0))
                except Exception:  # noqa: BLE001 -- a WCS that cannot invert leaves the centre
                    px = py = np.nan
            if -0.5 <= px <= nx - 0.5 and -0.5 <= py <= ny - 0.5:
                x0, y0, name = px, py, reference or "reference position"
        matrix = np.asarray(wcs.pixel_scale_matrix, dtype=float) * 3600.0   # arcsec / pix; rows: (xi, eta)
        sx_mag, sy_mag = float(np.hypot(*matrix[:, 0])), float(np.hypot(*matrix[:, 1]))
        inverse = np.linalg.inv(matrix)
        north, east = inverse @ np.array([0.0, 1.0]), inverse @ np.array([1.0, 0.0])
        north, east = north / np.hypot(*north), east / np.hypot(*east)
        aligned = abs(matrix[0, 1]) < np.tan(np.radians(1.0)) * sy_mag and abs(matrix[1, 0]) < np.tan(np.radians(1.0)) * sx_mag
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ra, dec = (float(v) for v in wcs.all_pix2world(x0, y0, 0))
        if aligned:
            return cls(x0, y0, float(matrix[0, 0]), float(matrix[1, 1]), True, tuple(north), tuple(east), ra, dec, name)
        return cls(x0, y0, sx_mag, sy_mag, False, tuple(north), tuple(east), ra, dec, name)

    @classmethod
    def from_header(cls, header: Any, shape: tuple[int, ...], *, primary: Any = None,
                    ref_world: tuple[float, float] | None = None, reference: str = "") -> SkyFrame | None:
        """From a FITS header with a celestial WCS (a cube's SCI header works: the spectral
        axis is dropped); None when the header has none.

        Without ``ref_world`` the target position (``TARG_RA``/``TARG_DEC``
        of ``primary``, or of ``header``) is the reference when it falls
        inside the image.
        """
        from ..cubes.wcs import celestial_wcs_from_header

        if header is None or "CTYPE1" not in header or "CTYPE2" not in header:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                wcs = celestial_wcs_from_header(header)   # the two celestial axes only: a tabular wavelength axis is no obstacle
            except Exception as exc:  # noqa: BLE001 -- a broken WCS only costs the sky axes
                log.debug("no usable celestial WCS: %s", exc)
                return None
        if wcs.naxis != 2:
            return None
        if ref_world is None:
            for hdr in (primary, header):
                if hdr is not None and hdr.get("TARG_RA") is not None and hdr.get("TARG_DEC") is not None:
                    ref_world, reference = (float(hdr["TARG_RA"]), float(hdr["TARG_DEC"])), reference or "target"
                    break
        return cls.from_wcs(wcs, shape, ref_world=ref_world, reference=reference)

    @classmethod
    def from_scale(cls, pixscale_arcsec: float, shape: tuple[int, ...], *, north_up: bool = False,
                   reference: str = "image centre") -> SkyFrame:
        """From a pixel scale alone (no WCS): offsets from the image centre along the image
        axes -- or, with ``north_up``, sky offsets with East to the left."""
        ny, nx = int(shape[-2]), int(shape[-1])
        s = float(pixscale_arcsec)
        return cls((nx - 1) / 2.0, (ny - 1) / 2.0, -s if north_up else s, s, north_up, reference=reference)

    @classmethod
    def from_fits(cls, path: Any, shape: tuple[int, ...], *, ext: str = "SCI") -> SkyFrame | None:
        """From the image extension ``ext`` of a FITS product (None without it or its WCS)."""
        from astropy.io import fits

        with fits.open(path) as hdul:
            if ext not in hdul:
                return None
            return cls.from_header(hdul[ext].header, shape, primary=hdul[0].header)

    @property
    def labels(self) -> tuple[str, str]:
        """The (x, y) arcsec axis labels: RA/Dec offsets when sky-aligned, image-axis offsets otherwise."""
        return (DRA_LABEL, DDEC_LABEL) if self.north_up else (DX_LABEL, DY_LABEL)

    def offsets(self, x: Any, y: Any) -> tuple[Any, Any]:
        """Pixel coordinates -> offsets [arcsec] from the reference pixel."""
        return (np.asarray(x, dtype=float) - self.x0) * self.sx, (np.asarray(y, dtype=float) - self.y0) * self.sy

    def note(self) -> str:
        """The legend line that states where the offsets are measured from."""
        pixel = f"pixel ({self.x0:.1f}, {self.y0:.1f})"
        if self.ra is None or self.dec is None:
            return f"zero of both axes: {self.reference}, {pixel}"
        return (rf"zero of both axes: {self.reference}, $\alpha$ = {self.ra:.5f}$\degree$, "
                rf"$\delta$ = {self.dec:+.5f}$\degree$, {pixel}")


def _offset_axis(axis: Any, zero: float) -> None:
    """Ticks of a pixel axis as offsets from ``zero``: round numbers of pixels either side of it."""
    from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MaxNLocator

    class Shifted(MaxNLocator):
        def tick_values(self, vmin: float, vmax: float) -> Any:
            return super().tick_values(vmin - zero, vmax - zero) + zero

    axis.set_major_locator(Shifted(nbins=6, steps=[1, 2, 5, 10]))
    axis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v - zero:.6g}".replace("-", "\N{MINUS SIGN}")))
    axis.set_minor_locator(AutoMinorLocator())


def sky_axes(ax: Any, frame: SkyFrame | None, *, sides: str = "lbtr") -> None:
    """Give a frame both coordinate systems with **one common zero** (rule 14): offsets in
    pixels at the bottom/left, offsets on the sky in arcsec at the top/right, both measured
    from the reference position of ``frame`` (the target, else the image centre) -- so the
    zeros of the two sets of axes coincide. ``frame.note()`` states where that zero is.

    ``sides`` names the sides that get tick labels and an axis label (the letters of
    left, bottom, top, right); the others keep their ticks only -- for the inner
    panels of a grid. Without a ``frame`` (no celestial WCS: a detector image) the axes are
    plain pixel coordinates. The image itself stays in pixel coordinates, so contours,
    apertures and markers are drawn in pixels. Call before :func:`colorbar`
    (``imshow(..., sky=frame)`` does both in the right order), so that the colorbar is
    placed beyond the right-hand axis.
    """
    ax.tick_params(labelbottom="b" in sides, labelleft="l" in sides)
    if frame is None:
        ax.set(xlabel=X_PIX_LABEL if "b" in sides else "", ylabel=Y_PIX_LABEL if "l" in sides else "")
        return
    _offset_axis(ax.xaxis, frame.x0)
    _offset_axis(ax.yaxis, frame.y0)
    ax.set(xlabel=DX_PIX_LABEL if "b" in sides else "", ylabel=DY_PIX_LABEL if "l" in sides else "")
    ax.tick_params(which="both", top=False, right=False)
    top = ax.secondary_xaxis("top", functions=(lambda x: (x - frame.x0) * frame.sx,
                                               lambda a: a / frame.sx + frame.x0))
    right = ax.secondary_yaxis("right", functions=(lambda y: (y - frame.y0) * frame.sy,
                                                   lambda a: a / frame.sy + frame.y0))
    top.tick_params(which="both", direction="in", labeltop="t" in sides)
    right.tick_params(which="both", direction="in", labelright="r" in sides)
    if "t" in sides:
        top.set_xlabel(frame.labels[0])
    if "r" in sides:
        right.set_ylabel(frame.labels[1])
    ax._qafig_right_axis = 0.50 if "r" in sides else 0.0   # [inch] room the colorbar leaves for it
    if not frame.north_up and frame.north is not None:
        compass(ax, frame)


def compass(ax: Any, frame: SkyFrame, *, size: float = 0.12) -> None:
    """North and East arrows in the lower-right corner of an image that is not aligned with the sky."""
    import matplotlib.patheffects as pe

    if frame.north is None or frame.east is None:
        return
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    length = size * min(abs(x_hi - x_lo), abs(y_hi - y_lo))
    origin = np.array([x_hi - 1.6 * length * np.sign(x_hi - x_lo), y_lo + 1.6 * length * np.sign(y_hi - y_lo)])
    outline = [pe.withStroke(linewidth=2.0, foreground="black")]
    for direction, name in ((frame.north, "N"), (frame.east, "E")):
        tip = origin + length * np.asarray(direction)
        ax.annotate("", xy=tip, xytext=origin, annotation_clip=False,
                    arrowprops={"arrowstyle": "-|>", "color": "white", "lw": 1.0, "path_effects": outline})
        ax.text(*(origin + 1.35 * length * np.asarray(direction)), name, color="white", fontsize=FONT["note"],
                ha="center", va="center", path_effects=outline)
    ax.set(xlim=(x_lo, x_hi), ylim=(y_lo, y_hi))


# --------------------------------------------------------------------------- images (rules 4, 5, 11)
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
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*(empty slice|All-NaN).*")
        image = reducers[how](data, axis=0)
    finite = np.isfinite(data).mean(axis=0)
    image = np.where(finite > 0, image, np.nan)   # nansum of nothing is 0, not a measurement
    if min_coverage > 0:
        image = np.where(finite >= min_coverage, image, np.nan)
    return image


def robust_limits(image: np.ndarray, percentiles: tuple[float, float] = (1.0, 99.5), *,
                  positive: bool = False) -> tuple[float, float]:
    """Display limits of an image that neither NaNs nor isolated hot pixels can set.

    The ``percentiles`` of the finite pixels (of the positive ones with
    ``positive``, for a log stretch), the upper one additionally capped at
    the brightest *neighbourhood* of the image (the maximum of its 3x3
    median filter): on a small image even the 99.9th percentile is
    essentially the single hottest pixel, while a real compact source is
    several pixels wide. Always returns a well-formed pair (``hi > lo``);
    ``(0, 1)`` for an image without usable pixels.
    """
    img = np.asarray(image, dtype=float)
    values = img[np.isfinite(img)]
    if positive:
        values = values[values > 0]
    if not values.size:
        return (1e-3, 1.0) if positive else (0.0, 1.0)
    lo, hi = (float(v) for v in np.percentile(values, percentiles))
    cap = _robust_max(img)
    if cap is not None and (cap > 0 or not positive):
        hi = min(hi, cap)
    if positive:
        lo = max(lo, hi * 1e-5)
        if not hi > lo:
            hi = lo * 10.0   # (near-)constant image: keep the norm well-formed
    elif not hi > lo:
        hi = lo + 1.0
    return lo, hi


def _robust_max(image: np.ndarray) -> float | None:
    """The brightest 3x3 *neighbourhood* median of a 2-D image (None when unusable).

    An isolated hot pixel cannot raise it, a real source -- bright over
    several adjacent pixels -- keeps its near-peak value; :func:`robust_limits`
    caps its upper limit here.
    """
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


#: The image stretches of :func:`imshow`. ``asinh`` is the default choice for sky images:
#: linear around the background (zero and negative pixels stay visible), logarithmic above.
STRETCHES = ("linear", "sqrt", "asinh", "log")
#: Width of the linear part of the ``asinh`` stretch, as a fraction of the display range.
ASINH_LINEAR_FRACTION = 0.03


def stretch_norm(image: np.ndarray, stretch: str = "linear", percentiles: tuple[float, float] = (1.0, 99.5), *,
                 vmin: float | None = None, vmax: float | None = None, center: float | None = None) -> Any:
    """The matplotlib norm of a ``stretch`` (:data:`STRETCHES`) over robust limits.

    ``vmin``/``vmax`` override the automatic limits (:func:`robust_limits`)
    one by one; ``center`` makes the limits symmetric about that value (for
    signed images shown with :data:`DIVERGING_CMAP`). Use it directly to
    share one norm between several panels: ``imshow(ax, img, norm=norm)``.
    """
    from matplotlib.colors import AsinhNorm, LogNorm, Normalize, PowerNorm

    if stretch not in STRETCHES:
        raise ValueError(f"unknown stretch {stretch!r}; one of {STRETCHES}")
    lo, hi = robust_limits(image, percentiles, positive=(stretch == "log"))
    lo, hi = (lo if vmin is None else float(vmin)), (hi if vmax is None else float(vmax))
    if center is not None:
        half = max(abs(hi - center), abs(center - lo)) or 1.0
        lo, hi = center - half, center + half
    if not hi > lo:
        hi = lo * 10.0 if stretch == "log" and lo > 0 else lo + 1.0
    if stretch == "log":
        return LogNorm(vmin=lo, vmax=hi)
    if stretch == "asinh":
        return AsinhNorm(linear_width=ASINH_LINEAR_FRACTION * (hi - lo), vmin=lo, vmax=hi)
    if stretch == "sqrt":
        return PowerNorm(0.5, vmin=lo, vmax=hi)
    return Normalize(vmin=lo, vmax=hi)


def colormap(cmap: Any = None) -> Any:
    """A colormap (default :data:`CMAP`) with NaN pixels in :data:`BAD_COLOR` (rule 5)."""
    import matplotlib

    use_agg()
    cm = matplotlib.colormaps[cmap or CMAP] if isinstance(cmap or CMAP, str) else cmap
    return cm.with_extremes(bad=BAD_COLOR)


def imshow(ax: Any, image: np.ndarray, *, unit: str | None = None, stretch: str = "linear",
           percentiles: tuple[float, float] = (1.0, 99.5), cmap: Any = None, center: float | None = None,
           cbar_label: str | None = None, cbar: bool = True, sky: SkyFrame | None = None,
           sides: str | None = "lbtr", **kwargs: Any) -> Any:
    """The standard image panel. Returns the ``AxesImage``.

    * ``unit`` converts the data via :func:`to_mjy` and labels the colorbar
      with the result (``cbar_label`` overrides).
    * ``stretch`` is one of :data:`STRETCHES` over robust display limits
      (:func:`robust_limits`: percentiles of the finite pixels, hot pixels
      cannot set the upper limit). Explicit ``vmin``/``vmax`` replace the
      automatic limits one by one, an explicit ``norm`` replaces all of it.
    * NaN pixels are drawn in :data:`BAD_COLOR`; under a log stretch zero
      and negative pixels take the lowest color (they are data, not gaps).
    * ``sky`` (a :class:`SkyFrame`) gives the panel its two sets of axes
      (:func:`sky_axes`): pixel offsets at the bottom/left, arcsec offsets
      at the top/right, from one zero. ``sides`` names the sides that are
      labelled (default all four; fewer for the inner panels of a grid);
      ``sides=None`` leaves the axes to the caller, for panels that are
      not frames on the sky (thumbnails, a detector with a wavelength axis).
    * the colorbar is as tall as the image, with pointed ends where data
      exceed the limits (``cbar=False`` for panels sharing a neighbour's).
    """
    img, unit_label = to_mjy(image, unit)
    if "norm" not in kwargs:
        kwargs["norm"] = stretch_norm(img, stretch, tuple(percentiles), vmin=kwargs.pop("vmin", None),
                                      vmax=kwargs.pop("vmax", None), center=center)
    norm = kwargs["norm"]
    shown = img
    if type(norm).__name__ == "LogNorm" and norm.vmin is not None:
        with np.errstate(invalid="ignore"):
            shown = np.where(img < norm.vmin, norm.vmin, img)   # <= 0 is data: lowest color, not a gap
    kwargs.setdefault("origin", "lower")
    kwargs.setdefault("interpolation", "nearest")
    im = ax.imshow(shown, cmap=colormap(cmap or (DIVERGING_CMAP if center is not None else CMAP)), **kwargs)
    if sides is not None:
        sky_axes(ax, sky, sides=sides)
    if cbar:
        colorbar(im, label=cbar_label if cbar_label is not None else unit_label, data=img)
    return im


def colorbar(im: Any, *, label: str | None = None, width: float = 0.13, pad: float | None = None,
             data: np.ndarray | None = None, extend: str | None = None) -> Any:
    """A colorbar exactly as tall as the image axes (rule 4), ``width`` inches wide, ``pad``
    inches to the right of them (default: just clear of the axes, or of their arcsec axis).

    ``extend`` (``neither``/``min``/``max``/``both``) is worked out from
    ``data`` (default: the image's own array) -- the bar gets a pointed end
    wherever data lie beyond the display limits.
    """
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory

    ax = im.axes
    fig = ax.figure
    if pad is None:
        pad = 0.07 + float(getattr(ax, "_qafig_right_axis", 0.0))
    if extend is None:
        values = np.asarray(im.get_array() if data is None else data, dtype=float)
        values = values[np.isfinite(values)]
        below = bool(values.size and im.norm.vmin is not None and values.min() < im.norm.vmin)
        above = bool(values.size and im.norm.vmax is not None and values.max() > im.norm.vmax)
        extend = {(False, False): "neither", (True, False): "min", (False, True): "max", (True, True): "both"}[below, above]
    # x in inches from the right edge of the axes, y in axes fraction: a fixed physical width
    x_inches = fig.dpi_scale_trans + ScaledTranslation(1.0, 0.0, ax.transAxes)
    cax = ax.inset_axes((pad, 0.0, width, 1.0), transform=blended_transform_factory(x_inches, ax.transAxes))
    cbar = fig.colorbar(im, cax=cax, extend=extend, extendfrac=0.03)
    cbar.ax.tick_params(which="both", direction="out", labelsize=FONT["tick"])
    if type(im.norm).__name__ == "LogNorm":
        _plain_log_axis(cbar.ax.yaxis)
    elif type(im.norm).__name__ == "AsinhNorm":
        from matplotlib.ticker import FuncFormatter

        cbar.set_ticks(_asinh_ticks(im.norm))
        cbar.minorticks_off()
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.6g}".replace("-", "\N{MINUS SIGN}")))
    if label:
        cbar.set_label(label)
    return cbar


def panel_note(ax: Any, text: str, *, loc: str = "upper left") -> Any:
    """A short label inside a *line-plot* panel of a multi-panel figure (which pair, which band,
    a measured value): dark text on a translucent white box in one of the four corners.
    What is specific to one panel goes here; what is common to all goes into the legend."""
    vertical, horizontal = loc.split()
    x = {"left": 0.03, "right": 0.97}[horizontal]
    y = {"upper": 0.96, "lower": 0.04}[vertical]
    return ax.text(x, y, text, transform=ax.transAxes, color="0.15", fontsize=FONT["note"],
                   ha=horizontal, va={"upper": "top", "lower": "bottom"}[vertical], zorder=6,
                   bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.8, "lw": 0})


def _asinh_ticks(norm: Any, *, min_sep: float = 0.075) -> list[float]:
    """Colorbar ticks of an asinh stretch: zero, the decades and -- where the bar has room --
    their 5- and 2-multiples, none closer than ``min_sep`` of the bar to another."""
    vmin, vmax = float(norm.vmin), float(norm.vmax)
    top = max(abs(vmin), abs(vmax))
    if not np.isfinite(top) or top <= 0:
        return [vmin, vmax]
    candidates: list[tuple[int, float, float]] = [(0, 0.0, 0.0)] if vmin <= 0.0 <= vmax else []
    high = int(np.floor(np.log10(top)))
    for exponent in range(high, high - 7, -1):
        for rank, mantissa in ((1, 1.0), (2, 5.0), (3, 2.0)):
            for sign in (1.0, -1.0):
                value = sign * mantissa * 10.0**exponent
                if vmin <= value <= vmax:
                    candidates.append((rank, -abs(value), value))
    kept: list[tuple[float, float]] = []
    for _rank, _order, value in sorted(candidates):
        position = float(norm(value))
        if all(abs(position - p) >= min_sep for p, _ in kept):
            kept.append((position, value))
    return sorted(v for _, v in kept) or [vmin, vmax]


def tag(ax: Any, text: str, *, loc: str = "upper left") -> Any:
    """A short label inside an image panel (a wavelength, a dither): white on a translucent
    dark box, in one of the four corners -- the in-panel counterpart of a subplot title."""
    vertical, horizontal = loc.split()
    x = {"left": 0.04, "right": 0.96}[horizontal]
    y = {"upper": 0.96, "lower": 0.04}[vertical]
    return ax.text(x, y, text, transform=ax.transAxes, color="white", fontsize=FONT["note"],
                   ha=horizontal, va={"upper": "top", "lower": "bottom"}[vertical],
                   bbox={"boxstyle": "round,pad=0.25", "fc": "black", "alpha": 0.55, "lw": 0})


def contour_mask(ax: Any, mask: np.ndarray, color: str, *, ls: str = "solid", label: str | None = None) -> None:
    """The outline of a boolean spaxel mask over an image, with its legend handle.
    An empty mask draws nothing (and gets no legend entry)."""
    mask = np.asarray(mask)
    if not mask.any():
        return
    ax.contour(mask.astype(float), levels=[0.5], colors=color,
               linewidths=LW["overlay"] if ls == "solid" else 1.0, linestyles=ls)
    if label:
        contour_proxy(ax, color, ls=ls, label=label)


def contour_proxy(ax: Any, color: str, *, ls: str = "solid", label: str | None = None) -> Any:
    """A legend handle for a contour set (matplotlib's own contours make poor handles);
    added to ``ax`` so :func:`figlegend` picks it up."""
    from matplotlib.lines import Line2D

    handle = Line2D([], [], color=color, lw=LW["overlay"], linestyle=ls, label=label)
    ax.add_line(handle)
    return handle


def marker(ax: Any, x: float, y: float, *, label: str | None = None, symbol: str = "+") -> Any:
    """A position marker on an image (a source, the target): a white ``symbol`` (a plus by default;
    ``x`` for a second kind of source) with a dark outline."""
    import matplotlib.patheffects as pe

    return ax.plot([x], [y], symbol, color="white", ms=9 if symbol == "+" else 7, mew=1.2, ls="none", label=label,
                   path_effects=[pe.withStroke(linewidth=2.6, foreground="black")])[0]


__all__ = [
    "ASINH_LINEAR_FRACTION",
    "STRETCHES",
    "SkyFrame",
    "collapse",
    "colorbar",
    "colormap",
    "compass",
    "contour_mask",
    "contour_proxy",
    "imshow",
    "marker",
    "panel_note",
    "robust_limits",
    "sky_axes",
    "stretch_norm",
    "tag",
]
