"""The layout of the feature annotation of a spectrum axes.

:func:`~jwstflow.qa.annotation.annotate_features` only records what to mark;
:class:`FeatureLayout` places it when the figure is saved, in phases that each
do one thing: find what stands out of the spectrum, turn the lines into labels,
make room for the band rows, place the labels along the spectrum, set the flux
range, draw.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .labels import (
    UNIDENTIFIED_IN_BAND,
    UNIDENTIFIED_SIGMA,
    _band_group,
    _band_text,
    _line_label,
    _member_span,
    _photosphere_color,
    _photosphere_label,
    _series,
    _unidentified_label,
    detect_lines,
    species_color,
    tex_label,
)
from .placement import _pack_bands, _spread
from .style import FONT, REF_COLOR

log = logging.getLogger(__name__)

#: The constant factor between the observation and the end of a line's stem: stems stop this
#: factor above the top of an emission line (below the bottom of an absorption line).
FEATURE_LABEL_OFFSET = 0.85
#: An annotated figure stays wider than tall: at most this height / width, and when its strips
#: need more it is widened (up to this width in inches) instead of squeezing the data area.
FEATURE_MAX_ASPECT = 0.62
FEATURE_MAX_WIDTH = 20.0
#: A line that stands out of the spectrum beside it by more than this many (median) label
#: lengths keeps its label on top of itself; the labels of its neighbours dodge left or right of it.
TALL_LINE = 0.5
#: The labels in the plot may take up to this fraction of the axes width; with more the figure is widened.
FEATURE_FILL = 0.8
#: The least height [inch] of an annotated spectrum's data area, whatever legend sits above it.
FEATURE_MIN_AXES_HEIGHT = 2.9
#: A series of lines (an H I series, a branch of an H2 band) with at least this many members
#: in view is drawn as a comb above the plot in the comb layout (``series="combs"``).
COMB_MIN = 4
#: A broad gas band (a forest the catalogue names as a band, not line by line) is seen in the
#: spectrum when at least this many emission peaks that are no catalogue line stand within this
#: many samples of one another inside it. Measured on edge-on disks: a forest of CO, OH or water
#: lines puts 5 to 20 there, noise and stray peaks at most 4. Only a band that is seen explains
#: the peaks inside it, and where only the detected features are marked only such bands are drawn.
FOREST_MIN_PEAKS = 5
FOREST_WINDOW = 300
#: The rows of solid-state bands at the foot of the plot, top to bottom.
BAND_ROW_ORDER = ("H2O ice", "CO2 ice", "CO ice", "other ices", "silicates", "PAH bands")
#: Artists of the annotation that are lines carry this gid, so that they never count as data.
FEATURE_GID = "qafig-feature"


def _strip(ax: Any, offset: float, height: float, *, above: bool) -> Any:
    """A child axes of ``height`` inches glued to the top (``above``) or bottom edge of ``ax``,
    ``offset`` inches away from it, sharing its x axis: the room for labels and band rows."""
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory

    fig = ax.figure
    anchor = fig.dpi_scale_trans + ScaledTranslation(0.0, 1.0 if above else 0.0, ax.transAxes)
    bounds = (0.0, offset, 1.0, height) if above else (0.0, -(offset + height), 1.0, height)
    strip = ax.inset_axes(bounds, transform=blended_transform_factory(ax.transAxes, anchor), sharex=ax, zorder=1)
    strip.set_ylim(0.0, 1.0)
    strip.patch.set_alpha(0.0)
    strip.tick_params(which="both", left=False, right=False, top=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in strip.spines.values():
        spine.set_visible(False)
    return strip


def _data_lines(ax: Any) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    """``(x, y, is_main)`` of the data lines of an axes (annotation artists left out)."""
    from matplotlib.colors import to_rgb

    out = []
    for line in ax.lines:
        if line.get_gid() == FEATURE_GID or line.get_transform() != ax.transData:
            continue
        x, y = np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)
        if x.shape != y.shape or x.size < 2:
            continue
        try:
            main = to_rgb(line.get_color()) == (0.0, 0.0, 0.0)
        except ValueError:
            main = False
        out.append((x, y, main))
    return out


def _window(values: np.ndarray, half: int, func: Any) -> np.ndarray:
    """A running statistic ``func`` over ``2 * half + 1`` samples (edges repeated)."""
    padded = np.pad(values, half, mode="edge")
    return func(np.stack([padded[k:k + values.size] for k in range(2 * half + 1)]), axis=0)


def _merged(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sorted intervals with the overlapping ones merged."""
    out: list[tuple[float, float]] = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


class FeatureLayout:
    """Lays out the feature annotation recorded on ``ax``: every line labelled a constant factor
    above (below) the spectrum at its own wavelength, the broad gas bands in a strip above the
    axes, the solid-state bands in rows at the foot of the plot area.

    All lengths are in display pixels unless a name says otherwise; ``u`` is the flux
    coordinate the layout works in (log10 of the flux on a log axis, the flux itself otherwise).
    """

    def __init__(self, ax: Any) -> None:
        self.ax = ax
        self.record: dict[str, Any] = ax.__dict__.get("_qafig_annotation") or {}
        self.fig = ax.figure
        self.renderer = self.fig.canvas.get_renderer()
        self.x_lo, self.x_hi = sorted(ax.get_xlim())
        self.size = FONT["feature"]
        self.px = self.fig.dpi / 72.0                      # one point in pixels
        self.thickness = 1.30 * self.size * self.px        # what a rotated label takes along x
        self.text_px = 1.25 * self.size * self.px
        # height of the curves of moved labels; least length of a stem
        self.curve_px, self.stem_px = 9.0 * self.px, 6.0 * self.px
        self.pitch_px, self.rect_px = self.text_px + 1.5 * self.px, 0.55 * self.text_px    # band rows
        self.tick_px = 2.5 * self.px
        self.comb_pitch = self.tick_px + 1.25 * FONT["comb"] * self.px + 2.5 * self.px
        self.lines = sorted((e for e in self.record.get("lines", []) if self.x_lo <= e["x"] <= self.x_hi),
                            key=lambda e: e["x"])
        self.bands = [dict(e, lo=max(e["lo"], self.x_lo), hi=min(e["hi"], self.x_hi))
                      for e in self.record.get("bands", []) if e["hi"] > self.x_lo and e["lo"] < self.x_hi]

    def run(self) -> None:
        """Carry the layout out (a no-op without anything to mark, or on an inverted flux axis)."""
        if not self.record or self.ax.yaxis.get_inverted() or not (self.lines or self.bands):
            return
        self.detect()
        self.label_lines()
        self.sort_bands()
        self.make_room()
        self.sample_spectrum()
        # a photospheric line with no room beside the band rows is labelled above the spectrum instead
        below, flipped = self.place(self.stellar_lines, above=False, blocked=_merged(self.band_boxes()))
        above, _ = self.place(sorted(self.inline_gas + flipped, key=lambda e: e["x"]), above=True, blocked=[])
        self.set_flux_range(above, below)
        self.draw_labels(above, below)
        self.draw_combs()
        self.draw_bands()
        self.fig.set_layout_engine("none")     # the strips are placed: later draws must not move anything

    # ------------------------------------------------------------------ coordinates and text
    def to_px(self, x: Any) -> np.ndarray:
        """Wavelengths -> display x."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return self.ax.transData.transform(np.column_stack([x, np.ones_like(x)]))[:, 0]

    def to_x(self, pixel: float) -> float:
        """Display x -> wavelength."""
        return float(self.ax.transData.inverted().transform((pixel, self.box.y0))[0])

    def measure(self, text: str, rotation: float = 0.0, fontsize: float | None = None) -> tuple[float, float]:
        """Width and height of ``text`` as it would be drawn."""
        artist = self.ax.text(0.5, 0.5, text, transform=self.ax.transAxes, rotation=rotation,
                              fontsize=self.size if fontsize is None else fontsize)
        extent = artist.get_window_extent(self.renderer)
        artist.remove()
        return extent.width, extent.height

    def u(self, values: Any) -> Any:
        """Flux -> the coordinate the layout works in."""
        return np.log10(values) if self.log_y else np.asarray(values, dtype=float)

    def on_grid(self, values: np.ndarray, pixel: Any) -> Any:
        """A quantity sampled on the pixel grid, at display x ``pixel``."""
        return np.interp(pixel, self.grid, values)

    # ------------------------------------------------------------------ phases
    def detect(self) -> None:
        """What stands out of the spectrum: orders the labels where they crowd, and finds strong
        peaks no catalogue line explains. Every line in range is marked whether it is seen or not;
        ``detected_only`` restricts the marks to the lines found -- a catalogue line is found when
        a peak (a dip, for the photosphere) stands within its tolerance (:func:`detect_lines`) --
        and the broad gas bands to those whose forest the spectrum shows (:data:`FOREST_MIN_PEAKS`
        peaks no catalogue line is within :data:`FOREST_WINDOW` samples). Only such a band explains
        the peaks inside it, in either figure, and inside it a catalogue line counts as found only
        where its peak stands above the forest around it. Solid-state bands are kept: no peak
        search can tell them."""
        record = self.record
        detected_only = bool(record.get("detected_only", False))
        self.data = _data_lines(self.ax)
        if record.get("under") is not None:
            self.curves = [tuple(np.asarray(v, dtype=float) for v in record["under"])]
        else:
            self.curves = [(d[0], d[1]) for d in ([d for d in self.data if d[2]] or self.data)]
        found = [f for f in (detect_lines(cx, cy) for cx, cy in self.curves) if f is not None]
        self.unidentified: list[float] = []
        if not found:
            if detected_only:      # nothing could be searched, so nothing is known to be there
                log.info("the spectrum is too short to search for lines: none labelled")
                self.lines, self.bands = [], [e for e in self.bands if not self._gas_band(e)]
            return
        peaks = {key: np.concatenate([f[key] for f in found]) for key in found[0]}

        def hits(e: dict[str, Any], key: str) -> np.ndarray:
            """The peaks (dips) that are line ``e``: those within their tolerance of it. A Q-branch
            too is found at its wavelength only -- its catalogue window is a band's width and would
            take the peaks of any forest around it."""
            return np.abs(peaks[f"x_{key}"] - e["x"]) <= peaks[f"tol_{key}"]

        for e in self.lines:
            key = "ab" if e["cls"] == "photosphere" else "em"
            mine = hits(e, key)
            e["strength"] = float(peaks[f"s_{key}"][mine].max()) if mine.any() else 0.0
        unknown = np.ones(peaks["x_em"].size, dtype=bool)
        for e in record["lines"]:
            if e["cls"] != "photosphere":
                unknown &= ~hits(e, "em")
        loose, step = peaks["x_em"][unknown], peaks["step_em"][unknown]

        def forest(e: dict[str, Any]) -> bool:
            """Whether the spectrum shows the forest of gas band ``e``."""
            inside = (loose >= e["lo"]) & (loose <= e["hi"])
            x, reach = loose[inside], FOREST_WINDOW * step[inside]
            return any(int(np.sum((x >= a) & (x <= a + r))) >= FOREST_MIN_PEAKS for a, r in zip(x, reach))

        seen_bands = [e for e in record["bands"] if self._gas_band(e) and forest(e)]
        covered = [(e["lo"], e["hi"]) for e in seen_bands]
        # inside a forest a catalogue line is found only where it stands above the forest around
        # it: a forest peak that happens to fall on a catalogue line is no detection of that line
        loose_s = peaks["s_em"][unknown]
        for e in self.lines:
            if e["cls"] != "photosphere" and e["strength"] > 0 and any(a <= e["x"] <= b for a, b in covered):
                near = np.abs(loose - e["x"]) <= 0.5 * FOREST_WINDOW * step
                if near.any() and e["strength"] <= float(loose_s[near].max()):
                    e["strength"] = 0.0
        for xp, sp, alone in zip(peaks["x_em"], peaks["s_em"], unknown):
            # inside a broad gas band its forest explains ordinary peaks; one that towers over it is flagged too
            need = UNIDENTIFIED_SIGMA * (UNIDENTIFIED_IN_BAND if any(a <= xp <= b for a, b in covered) else 1.0)
            if alone and sp >= need and self.x_lo <= xp <= self.x_hi:
                self.unidentified.append(float(xp))
        seen = sum(e["strength"] > 0 for e in self.lines)
        log.info("%d of %d catalogue lines in view stand out of the spectrum; %d strong peak(s) unidentified",
                 seen, len(self.lines), len(self.unidentified))
        if detected_only:
            self.lines = [e for e in self.lines if e["strength"] > 0]
            drawn = {(e["label"], e["lo"], e["hi"]) for e in seen_bands}
            self.bands = [e for e in self.bands if not self._gas_band(e) or any(
                b[0] == e["label"] and b[1] <= e["lo"] and e["hi"] <= b[2] for b in drawn)]

    @staticmethod
    def _gas_band(entry: dict[str, Any]) -> bool:
        """A broad band of gas emission (a forest named as one band), not a solid-state band or PAH."""
        return entry["cls"] == "emission" and entry["species"] != "PAH"

    def label_lines(self) -> None:
        """Turn the lines into labels: one per line, except that converging members of a series
        share one label for their range and photospheric multiplets one for the multiplet -- each
        member keeps its own stem. In the comb layout the series become combs above the plot."""
        thickness = self.thickness
        for e in self.lines:
            stellar = e["cls"] == "photosphere"
            e["color"] = _photosphere_color(e["species"]) if stellar else species_color(e["species"])
            text = _photosphere_label(e["label"]) if stellar else _line_label(e["label"])
            e["text"] = tex_label(text)
        series: dict[str, list[dict[str, Any]]] = {}
        inline_gas: list[dict[str, Any]] = []
        for e in self.lines:
            if e["cls"] == "photosphere":
                continue
            member = _series(e["label"])
            if member is None:
                inline_gas.append(e)
            else:
                series.setdefault(member[0], []).append(dict(e, tick=member[1]))
        self.combs: dict[str, list[dict[str, Any]]] = {}
        if self.record.get("series", "labels") == "combs":
            self.combs = {name: sorted(members, key=lambda e: e["x"]) for name, members in series.items()
                          if len(members) >= COMB_MIN}
            inline_gas += [e for name, members in series.items() if name not in self.combs for e in members]
        else:
            # every member is labelled at its line; members that lie closer than a label is wide
            # share one label for their range, each with its own stem to it -- unless another
            # labelled line stands between them, whose stem the shared label's stems would cross
            labelled = np.array([e["x"] for e in self.lines if e["cls"] != "photosphere"] + self.unidentified)
            for name, members in series.items():
                members = sorted(members, key=lambda e: e["x"])
                pixels = self.to_px([e["x"] for e in members])
                own = np.isin(labelled, [e["x"] for e in members])
                strangers = labelled[~own]
                runs, run = [], [0]
                for k in range(1, len(members)):
                    between = bool(np.any((strangers > members[k - 1]["x"]) & (strangers < members[k]["x"])))
                    if pixels[k] - pixels[k - 1] < thickness and not between:
                        run.append(k)
                    else:
                        runs.append(run)
                        run = [k]
                runs.append(run)
                for run in runs:
                    if len(run) < 3:
                        inline_gas += [members[k] for k in run]
                        continue
                    span = _member_span([members[k]["tick"] for k in run])
                    text = tex_label(name) + (f"({span})" if name.startswith("H2") else f" {span}")
                    xs = [members[k]["x"] for k in run]
                    inline_gas.append(dict(members[run[0]], x=float(np.sqrt(xs[0] * xs[-1])), members=xs, text=text,
                                           strength=max(members[k].get("strength", 0.0) for k in run)))
        inline_gas += [{"x": xp, "text": _unidentified_label(xp), "label": "?", "species": "?", "cls": "line",
                        "color": REF_COLOR, "strength": UNIDENTIFIED_SIGMA} for xp in self.unidentified]
        inline_gas.sort(key=lambda e: e["x"])
        self.inline_gas = inline_gas
        self.with_labels = len(inline_gas) <= int(self.record.get("max_line_labels", 250))
        for e in self.lines:
            e["width"], e["length"] = self.measure(e["text"], 90)
        for e in inline_gas:
            e["width"], e["length"] = self.measure(e["text"], 90)
        # photospheric lines of one species that stand side by side are one multiplet: one label
        # (named a doublet / triplet where the catalogue says so), a stem from every line
        stellar_lines: list[dict[str, Any]] = []
        for e in sorted((e for e in self.lines if e["cls"] == "photosphere"), key=lambda e: e["x"]):
            last = stellar_lines[-1] if stellar_lines else None
            if last is not None and last["name"] == e["text"] \
                    and float(self.to_px(e["x"])[0] - self.to_px(last["members"][-1])[0]) < 2.5 * thickness:
                last["members"].append(e["x"])
                last["kinds"].append(e.get("notes", ""))
                last["strength"] = max(last.get("strength", 0.0), e.get("strength", 0.0))
            else:
                stellar_lines.append(dict(e, name=e["text"], members=[e["x"]], kinds=[e.get("notes", "")]))
        for e in stellar_lines:
            if len(e["members"]) == 1:
                del e["members"]
                continue
            word = next((w for w in ("doublet", "triplet") if all(w in kind for kind in e["kinds"])), "")
            e["text"] = f"{e['name']} {word}".strip() if word else f"{e['name']} ($\\times${len(e['members'])})"
            e["x"] = float(np.sqrt(e["members"][0] * e["members"][-1]))
            e["width"], e["length"] = self.measure(e["text"], 90)
        self.stellar_lines = stellar_lines

    def sort_bands(self) -> None:
        """Group, color and name the bands: gas emission for the strip above the axes, solid-state
        bands for the rows at the foot of the plot."""
        for e in self.bands:
            e["group"], e["color"] = _band_group(e)
            e["text"] = _band_text(e)
        self.gas_bands = sorted((e for e in self.bands if "gas" in e["group"]), key=lambda e: e["lo"])
        self.solid_bands = sorted((e for e in self.bands if "gas" not in e["group"]),
                                  key=lambda e: (BAND_ROW_ORDER.index(e["group"]), e["lo"]))
        pah = [e for e in self.solid_bands if e["group"] == "PAH bands"]
        for e in pah[1:]:                                                  # the row is named at its first band
            e["text"] = e["text"].replace("PAH ", "")

    def make_room(self) -> None:
        """The strip above the axes (broad gas emission; the combs of the comb layout) and the rows
        of solid-state bands at the foot of the plot area: pack them into rows, and size the figure
        so that the data area keeps its height and every label its place along the wavelength axis."""
        ax, fig, px = self.ax, self.fig, self.px
        box = ax.get_window_extent(self.renderer)
        comb_rows: list[list[tuple[float, float]]] = []
        self.comb_layout: dict[str, dict[str, Any]] = {}
        for name, members in self.combs.items():
            first, last = (float(v) for v in self.to_px([members[0]["x"], members[-1]["x"]]))
            title = tex_label(name)
            width = self.measure(title)[0]
            reach = (3.0 * px + 0.5 * self.measure(members[0]["tick"], fontsize=FONT["comb"])[0],
                     3.0 * px + 0.5 * self.measure(members[-1]["tick"], fontsize=FONT["comb"])[0])
            name_left = first - reach[0] - width >= box.x0
            left, right = (first - reach[0] - width, last + reach[1]) if name_left else (first - reach[0], last + reach[1] + width)
            for r in range(len(comb_rows) + 1):
                if r == len(comb_rows):
                    comb_rows.append([])
                if all(right + 4.0 * px < t0 or left - 4.0 * px > t1 for t0, t1 in comb_rows[r]):
                    comb_rows[r].append((left, right))
                    self.comb_layout[name] = {"row": r, "title": title, "name_left": name_left, "reach": reach}
                    break
        row_px = len(comb_rows) * self.comb_pitch + (2.0 * px if comb_rows else 0.0)
        gas_rows = _pack_bands(self.gas_bands, self.to_px, self.measure, box.x0, box.x1, px)
        solid_rows = _pack_bands(self.solid_bands, self.to_px, self.measure, box.x0, box.x1, px)
        self.top_px = row_px + (gas_rows * self.pitch_px + 3.0 * px if gas_rows else 0.0)
        self.foot_px = solid_rows * self.pitch_px + 4.0 * px if solid_rows else 0.0
        width_in, height_in = fig.get_size_inches()
        extra = (self.top_px + self.foot_px) / fig.dpi
        # every label needs its place along the wavelength axis: a figure with more labels gets wider
        crowd = max(len(self.inline_gas), len(self.stellar_lines)) * self.thickness / FEATURE_FILL
        width_want = min(max(width_in, width_in + (crowd - box.width) / fig.dpi), FEATURE_MAX_WIDTH) \
            if self.with_labels else width_in
        if extra > 0 or width_want > width_in:   # the data area keeps its height; the figure never gets taller than wide-ish
            needed = height_in + extra
            width_new = min(max(width_want, needed / FEATURE_MAX_ASPECT), FEATURE_MAX_WIDTH)
            fig.set_size_inches(width_new, min(needed, FEATURE_MAX_ASPECT * width_new))
        self.top = _strip(ax, 0.0, self.top_px / fig.dpi, above=True) if self.top_px else None
        fig.draw_without_rendering()
        box = ax.get_window_extent(self.renderer)
        deficit = FEATURE_MIN_AXES_HEIGHT * fig.dpi - box.height
        if deficit > 1.0:       # a tall legend must not squeeze the spectrum: give the axes its height back
            width_now, height_now = fig.get_size_inches()
            needed = height_now + deficit / fig.dpi
            width_new = min(max(width_now, needed / FEATURE_MAX_ASPECT), FEATURE_MAX_WIDTH)
            fig.set_size_inches(width_new, min(needed, FEATURE_MAX_ASPECT * width_new))
            fig.draw_without_rendering()
            box = ax.get_window_extent(self.renderer)
        self.box = box
        self.full = box.height

    def sample_spectrum(self) -> None:
        """The observation on a pixel grid: its top and bottom at every position along the axis."""
        ax, box = self.ax, self.box
        self.log_y = ax.get_yscale() == "log"
        cx = np.concatenate([d[0] for d in self.data]) if self.data else np.array([])
        cy = np.concatenate([d[1] for d in self.data]) if self.data else np.array([])
        if self.record.get("under") is not None:
            cx, cy = self.curves[0]
        ok = np.isfinite(cx) & np.isfinite(cy) & (cx >= self.x_lo) & (cx <= self.x_hi)
        if self.log_y:
            ok &= cy > 0
        cx, cy = cx[ok], cy[ok]
        self.grid = grid = np.arange(box.x0, box.x1 + 3.0, 3.0)
        low, high = np.full(grid.size, np.inf), np.full(grid.size, -np.inf)
        if cx.size:
            index = np.clip(np.round((self.to_px(cx) - grid[0]) / 3.0).astype(int), 0, grid.size - 1)
            np.minimum.at(low, index, self.u(cy))
            np.maximum.at(high, index, self.u(cy))
        self.have = have = np.isfinite(low)
        v_bottom, v_top = ax.get_ylim()
        if have.any():
            low, high = np.interp(grid, grid[have], low[have]), np.interp(grid, grid[have], high[have])
        else:
            low = high = np.full(grid.size, 0.5 * (float(self.u(v_bottom)) + float(self.u(v_top))))
        self.tops, self.bottoms = _window(high, 1, np.max), _window(low, 1, np.min)
        self.u0, self.u1 = float(self.u(v_bottom)), float(self.u(v_top))

    def band_boxes(self) -> list[tuple[float, float]]:
        """Where the band rows (rectangles and their labels) stand along the axis: no label below
        the spectrum may stand there."""
        px = self.px
        boxes = []
        for e in self.solid_bands:
            a_, b_ = self.to_px([e["lo"], e["hi"]])
            side = e.get("side")
            left = a_ - (e["label_px"] + 3.0 * px if side == "left" else 0.0) - 2.0 * px
            right = b_ + (e["label_px"] + 3.0 * px if side == "right" else 0.0) + 2.0 * px
            boxes.append((float(left), float(right)))
        return boxes

    def place(self, entries: list[dict[str, Any]], *, above: bool,
              blocked: list[tuple[float, float]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Place the labels of ``entries`` above (below) the spectrum; ``blocked`` are stretches of
        the axis no label may stand in. Returns the placed items and the entries that found no room.

        Each label stands a constant factor above (below) the spectrum at its own line. A line
        that towers over its neighbours keeps its label on top of itself, moved just aside where
        the stem of another line would rise inside its column (it gives up its place where no
        spot within a label width is free, and below the spectrum where the band rows stand).
        The others dodge left or right of it, never past it: a label stays between the pinned
        labels on either side of its line (where there is no room, the lower of the two gives up
        its place). Labels that still would touch are moved apart, and only labels whose
        connectors would otherwise cross share a level. So no connector crosses a label or
        another connector.
        """
        if not entries or not self.with_labels:
            return [], []
        thickness, px, box, grid = self.thickness, self.px, self.box, self.grid
        edge = self.tops if above else self.bottoms
        sign = 1.0 if above else -1.0
        n = len(entries)
        pixels = self.to_px([e["x"] for e in entries])
        priority = np.array([e.get("strength", 0.0) for e in entries])
        level = _window(_window(edge, 6, np.min if above else np.max), 8, np.mean)     # the continuum beside the lines
        rise = sign * (self.on_grid(edge, pixels) - self.on_grid(level, pixels)) / (self.u1 - self.u0) * self.full
        tall = rise > TALL_LINE * float(np.median([e["length"] for e in entries]))
        for i in np.nonzero(tall)[0]:        # (below the spectrum) not where the band rows stand
            if any(a_ - 0.5 * thickness < pixels[i] < b_ + 0.5 * thickness for a_, b_ in blocked):
                tall[i] = False
        stems = [np.atleast_1d(self.to_px(e.get("members") or [e["x"]])) for e in entries]
        stem_x = np.concatenate(stems)
        stem_of = np.concatenate([np.full(len(x), j) for j, x in enumerate(stems)])
        lo_edge, hi_edge = box.x0 + 2.0, box.x1 - 2.0
        offsets = sorted(np.arange(-thickness, thickness + 0.25, 0.5), key=abs)

        def pin() -> dict[int, float]:
            """Where the labels of the tall lines stand: over their own line, or as close to it as
            needed -- a label width from one another, with no stem of another line inside the
            label's column or between the label and its line. A label that finds no such spot
            within a label width of its line gives up its place on top; its line dodges like any
            other."""
            spots: dict[int, float] = {}
            for i in sorted(np.nonzero(tall)[0], key=lambda i: -rise[i]):
                others = stem_x[stem_of != i]
                reach = 0.5 * entries[i].get("width", thickness) + px
                for d in offsets:
                    spot = float(pixels[i]) + d
                    a_, b_ = min(spot, float(pixels[i])), max(spot, float(pixels[i]))
                    if lo_edge + 0.5 * thickness <= spot <= hi_edge - 0.5 * thickness \
                            and all(abs(spot - q) >= thickness for q in spots.values()) \
                            and not any(lo - 0.5 * thickness < spot < hi + 0.5 * thickness for lo, hi in blocked) \
                            and not np.any(np.abs(others - spot) < reach) and not np.any((others > a_) & (others < b_)):
                        spots[int(i)] = spot
                        break
                else:
                    tall[i] = False
            return spots

        half = 0.5 * (thickness + 2.0 * px)      # pitch of two interleaved tiers of labels
        tier: dict[int, int] = {}

        def arrange(members: list[int], a_: float, b_: float) -> np.ndarray | None:
            """Positions of the labels of one stretch: side by side where that keeps them near their
            lines, otherwise in two tiers -- every second label one label length higher, its stem
            rising between its neighbours. None when even that does not fit."""
            wish = pixels[members]
            if len(members) * thickness <= b_ - a_:
                spots = _spread(wish, np.full(len(members), thickness), a_, b_)
                if np.max(np.abs(spots - wish)) <= 1.5 * thickness:
                    return spots
            inset = 0.5 * (thickness - half)
            if len(members) * half > b_ - a_ - 2.0 * inset:
                return None
            spots = _spread(wish, np.full(len(members), half), a_ + inset, b_ - inset)
            run = 0
            for k, i in enumerate(members):      # alternate within every stretch of labels closer than one width
                run = run + 1 if k and spots[k] - spots[k - 1] < thickness else 0
                tier[i] = run % 2
            return spots

        placed: dict[int, float] = {}
        skipped: set[int] = set()        # no room on this side of the spectrum at all
        for _ in range(2 * n + 2):
            placed = pin()            # the tall lines first, each over itself (or just beside it)
            keep = np.nonzero(tall)[0]
            walls = _merged([(p - 0.5 * thickness - px, p + 0.5 * thickness + px) for p in placed.values()] + blocked)
            starts = [lo_edge] + [w[1] for w in walls]
            ends = [w[0] for w in walls] + [hi_edge]
            segments = [(a_, b_) for a_, b_ in zip(starts, ends) if b_ - a_ >= thickness]
            rest = [i for i in range(n) if not tall[i] and i not in skipped]
            if rest and not segments:
                skipped.update(rest)
                break
            homes: dict[tuple[float, float], list[int]] = {}      # free stretch -> the labels placed in it
            demote = None
            for i in rest:
                # a label stays between the tall lines on either side of its own: its curve never
                # passes under a label that stands over its line
                left = max((float(pixels[j]) for j in keep if pixels[j] < pixels[i]), default=-np.inf)
                right = min((float(pixels[j]) for j in keep if pixels[j] > pixels[i]), default=np.inf)
                inside = {k: (max(a_, left), min(b_, right)) for k, (a_, b_) in enumerate(segments)
                          if min(b_, right) - max(a_, left) >= thickness}
                if not inside:     # no room between them: the lower of the two gives up its place on top
                    demote = min((int(j) for j in keep if pixels[j] in (left, right)), key=lambda j: rise[j])
                    break
                k = min(inside, key=lambda k: 0.0 if inside[k][0] <= pixels[i] <= inside[k][1]
                        else min(abs(pixels[i] - inside[k][0]), abs(pixels[i] - inside[k][1])))
                homes.setdefault(inside[k], []).append(i)
            if demote is not None:
                tall[demote] = False
                continue
            again = False
            tier.clear()
            for (a_, b_), members in homes.items():
                spots = arrange(members, a_, b_)
                if spots is None:
                    # no room between two tall lines: the lower of them gives up its place on top;
                    # between walls that cannot move (band rows) the weakest lines change sides
                    beside = [int(i) for i in keep if abs(placed[int(i)] - a_) < thickness or abs(placed[int(i)] - b_) < thickness]
                    if beside:
                        tall[min(beside, key=lambda i: rise[i])] = False
                    else:
                        extra = max(len(members) - int((b_ - a_ - thickness) // half), 1)
                        skipped.update(sorted(members, key=lambda i: priority[i])[:extra])
                    again = True
                    break
                placed.update({i: float(p) for i, p in zip(members, spots)})
            if not again:
                break
        # labels whose connectors overlap along the axis share a level; every other label has its own
        spans = sorted((min([pixels[i], p, *self.to_px(entries[i].get("members", []))]) - 0.5 * thickness,
                        max([pixels[i], p, *self.to_px(entries[i].get("members", []))]) + 0.5 * thickness, i, p)
                       for i, p in placed.items())
        groups: list[list[tuple[float, float, int, float]]] = []
        for item in spans:
            lone = bool(tall[item[2]])
            if groups and not lone and item[0] < max(g[1] for g in groups[-1]) and not tall[groups[-1][-1][2]]:
                groups[-1].append(item)
            else:
                groups.append([item])
        out = []
        for group in groups:
            near = (grid >= min(g[0] for g in group) - 2) & (grid <= max(g[1] for g in group) + 2)
            base = float(edge[near].max() if above else edge[near].min()) if near.any() \
                else float(self.on_grid(edge, group[0][3]))
            # the upper tier of a group stands one (longest lower) label above the lower one
            lift = max((entries[i]["length"] for _a, _b, i, _p in group if not tier.get(i)), default=0.0) + 3.0 * px
            for _a, _b, i, p in group:
                moved = abs(p - pixels[i]) > 0.75 or len(entries[i].get("members", [])) > 1 or bool(tier.get(i))
                out.append({"entry": entries[i], "label": p, "base": base, "curve": self.curve_px if moved else 0.0,
                            "lift": lift if tier.get(i) else 0.0})
        return out, [entries[i] for i in sorted(skipped)]

    def set_flux_range(self, above: list[dict[str, Any]], below: list[dict[str, Any]]) -> None:
        """The flux range: only the room the labels need; the band rows and the labels below the
        spectrum share the foot of the plot. An explicit range is left alone."""
        ax, px, full = self.ax, self.px, self.full
        gap_px = 3.0 * px
        offset = float(self.record.get("offset", FEATURE_LABEL_OFFSET))
        factor = -np.log10(offset) if self.log_y else 0.0       # the constant factor between spectrum and label, in dex
        u0, u1 = self.u0, self.u1
        automatic = ax.get_autoscaley_on() or getattr(ax, "_qafig_auto_y", False)
        if automatic and self.have.any():
            for _ in range(16):
                reach = max(gap_px, factor * full / (u1 - u0))      # that factor in pixels at the current range
                for item in above:
                    k = min((reach + max(item["curve"], self.stem_px) + item["lift"] + item["entry"]["length"] + 5.0 * px) / full, 0.8)
                    u1 = max(u1, u0 + (item["base"] - u0) / (1.0 - k))
                for item in below:
                    k = min((reach + max(item["curve"], self.stem_px) + item["lift"] + item["entry"]["length"] + 5.0 * px) / full, 0.8)
                    u0 = min(u0, (item["base"] - k * u1) / (1.0 - k))
                if self.foot_px:
                    k = min((self.foot_px + 5.0 * px) / full, 0.5)
                    u0 = min(u0, (float(np.min(self.bottoms)) - k * u1) / (1.0 - k))
            ax.set_ylim(10.0**u0 if self.log_y else u0, 10.0**u1 if self.log_y else u1)
            ax._qafig_auto_y = True
        self.u0, self.u1 = u0, u1
        self.gap = max(factor, gap_px / full * (u1 - u0))   # in u: the constant factor, never < 3 pt

    def fraction(self, value: Any) -> Any:
        """The layout's flux coordinate -> axes fraction."""
        return (np.asarray(value, dtype=float) - self.u0) / (self.u1 - self.u0)

    def draw_labels(self, above: list[dict[str, Any]], below: list[dict[str, Any]]) -> None:
        """The labels with their stems: straight where the label stands over its line, a smooth
        curve where it was moved, one stem per line of a shared label."""
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MplPath

        ax, px, full, gap = self.ax, self.px, self.full, self.gap
        main = ax.get_xaxis_transform()
        for items, sign, edge in ((above, 1.0, self.tops), (below, -1.0, self.bottoms)):
            for item in items:
                e = item["entry"]
                dashed = e["cls"] == "photosphere"
                x_label = self.to_x(item["label"])
                base = float(self.fraction(item["base"] + sign * gap))
                knee = base + sign * max(item["curve"], self.stem_px) / full   # every label has a stem to its line
                foot = knee + sign * item["lift"] / full                       # where the label starts (upper tier: higher)
                for x_line in e.get("members") or [e["x"]]:                    # one stem per line of a group
                    tip = float(self.fraction(self.on_grid(edge, float(self.to_px(x_line)[0])) + sign * gap))
                    if item["curve"]:
                        middle = 0.5 * (base + knee)
                        vertices = [(x_line, tip), (x_line, base), (x_line, middle), (x_label, middle), (x_label, knee)]
                        codes = [MplPath.MOVETO, MplPath.LINETO] + [MplPath.CURVE4] * 3
                        if item["lift"]:
                            vertices.append((x_label, foot))
                            codes.append(MplPath.LINETO)
                    else:
                        vertices, codes = [(x_line, tip), (x_line, knee)], [MplPath.MOVETO, MplPath.LINETO]
                    ax.add_patch(PathPatch(MplPath(vertices, codes), transform=main, fc="none", ec=e["color"], lw=0.5,
                                           ls=(0, (3, 2)) if dashed else "solid", capstyle="butt", zorder=4))
                ax.text(x_label, foot + sign * 1.5 * px / full, e["text"], transform=main, rotation=90, ha="center",
                        va="bottom" if sign > 0 else "top", fontsize=self.size, color=e["color"], zorder=5, clip_on=True)
        if not self.with_labels:      # too many to label: mark the lines at the edges of the axes
            for e in self.inline_gas + self.stellar_lines:
                y = (0.97, 1.0) if e["cls"] != "photosphere" else (0.0, 0.03)
                ax.plot([e["x"], e["x"]], y, transform=main, color=e["color"], lw=0.5, zorder=4, gid=FEATURE_GID)

    def draw_combs(self) -> None:
        """The comb layout: a named row per series in the strip above the axes, a tick at every member."""
        top, px, top_px, tick_px = self.top, self.px, self.top_px, self.tick_px
        if top is None or not self.combs:
            return
        blend = top.get_xaxis_transform()
        for name, members in self.combs.items():
            place = self.comb_layout[name]
            color = members[0]["color"]
            y_line = (place["row"] * self.comb_pitch + 1.0 * px + tick_px) / top_px
            xs = [e["x"] for e in members]
            top.plot([xs[0], xs[-1]], [y_line, y_line], transform=blend, color=color, lw=0.6, solid_capstyle="butt",
                     clip_on=False)
            last_right = -np.inf
            for e, pixel in zip(members, self.to_px(xs)):
                top.plot([e["x"], e["x"]], [y_line, y_line - tick_px / top_px], transform=blend, color=color, lw=0.6,
                         solid_capstyle="butt", clip_on=False)
                half = 0.5 * self.measure(e["tick"], fontsize=FONT["comb"])[0]
                if pixel - half > last_right + 1.0 * px:
                    top.text(e["x"], y_line + 1.0 * px / top_px, e["tick"], transform=blend, ha="center", va="bottom",
                             fontsize=FONT["comb"], color=color)
                    last_right = pixel + half
            first, last = (float(v) for v in self.to_px([xs[0], xs[-1]]))
            anchor, ha = (first - place["reach"][0], "right") if place["name_left"] else (last + place["reach"][1], "left")
            top.text(self.to_x(anchor), y_line, place["title"], transform=blend, ha=ha, va="center_baseline",
                     fontsize=self.size, color=color)

    def draw_bands(self) -> None:
        """The band rectangles with their names: gas emission (outlined) in rows above the axes,
        solid-state bands (filled) in rows at the foot of the plot."""
        from matplotlib.patches import Rectangle

        px, top = self.px, self.top
        for target, entries, zone_px, total_px in ((top, self.gas_bands, self.top_px, self.top_px),
                                                   (self.ax, self.solid_bands, self.foot_px, self.full)):
            if target is None or not entries:
                continue
            blend = target.get_xaxis_transform()
            for e in entries:
                a, b = self.to_px([e["lo"], e["hi"]])
                if b - a < 2.0 * px:
                    a, b = 0.5 * (a + b) - px, 0.5 * (a + b) + px
                # rows count down from the top of the strip, or of the foot zone inside the axes
                y = ((total_px if target is top else zone_px) - 1.5 * px - (e["row"] + 0.5) * self.pitch_px) / total_px
                outline = "gas" in e["group"]
                target.add_patch(Rectangle((self.to_x(a), y - 0.5 * self.rect_px / total_px), self.to_x(b) - self.to_x(a),
                                           self.rect_px / total_px, transform=blend, fc="none" if outline else e["color"],
                                           ec=e["color"] if outline else "none", lw=0.9, alpha=1.0 if outline else 0.7,
                                           clip_on=False, zorder=3))
                if e["side"] is None:
                    continue
                anchor, ha = (b + 3.0 * px, "left") if e["side"] == "right" else (a - 3.0 * px, "right")
                target.text(self.to_x(anchor), y, e["text"], transform=blend, ha=ha, va="center_baseline",
                            fontsize=self.size, color=e["color"] if e["group"] != "silicates" else "0.35", zorder=5)


__all__ = [
    "COMB_MIN",
    "FEATURE_FILL",
    "FEATURE_LABEL_OFFSET",
    "FEATURE_MAX_ASPECT",
    "FEATURE_MAX_WIDTH",
    "FEATURE_MIN_AXES_HEIGHT",
    "FOREST_MIN_PEAKS",
    "FOREST_WINDOW",
    "TALL_LINE",
    "FeatureLayout",
]
