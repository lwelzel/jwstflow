"""Tests of the QA figure standard (jwstflow.qa): image scaling, frames, lines and the feature annotation."""

from __future__ import annotations

import re
from itertools import pairwise

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from jwstflow import qa
from jwstflow.qa.frames import _robust_max
from jwstflow.qa.labels import _line_label, _member_span, _series
from jwstflow.qa.placement import _declutter


def field(rng, hot: bool = True) -> np.ndarray:
    """A ch4-like collapse: faint blob over a bright noisy background, NaN
    corners, and (optionally) isolated hot pixels."""
    yy, xx = np.mgrid[:32, :33]
    img = 2.8 + 0.05 * rng.standard_normal((32, 33))
    img += 2.3 * np.exp(-((xx - 16.0) ** 2 + (yy - 15.0) ** 2) / (2 * 1.0**2))
    img[(xx + yy < 10) | (xx + yy > 54)] = np.nan
    if hot:
        img[25, 4] = 40.0
        img[3, 27] = 25.0
    return img


def test_robust_max_ignores_isolated_hot_pixels():
    rng = np.random.default_rng(2)
    img = field(rng)
    assert np.nanmax(img) == 40.0
    cap = _robust_max(img)
    assert cap is not None and cap < 6.0        # the blob's neighbourhood, not the hot pixel
    assert cap > 3.5                            # ... but the blob still stands out of the background


def test_robust_max_keeps_extended_bright_sources():
    img = np.ones((20, 20))
    img[8:13, 8:13] = 50.0                      # a real, several-pixel source
    assert _robust_max(img) == 50.0
    assert _robust_max(np.full((2, 2), np.nan)) is None    # unusable input
    assert _robust_max(np.ones(9)) is None                 # not an image


def test_imshow_log_limits_span_the_real_structure(tmp_path):
    # the ch4 aperture-QA regression: percentiles hit the hot pixels and the
    # old one-decade floor forced vmax to 10 x vmin -- background-dominated
    # images came out black with the source invisible
    rng = np.random.default_rng(3)
    fig, ax = qa.subplots()
    im = qa.imshow(ax, field(rng), stretch="log", percentiles=(5.0, 99.9))
    assert im.norm.vmax < 6.0                   # capped at the blob, not the hot pixel
    assert im.norm.vmax > 3.5                   # ... and not squashed below the blob
    assert 2.0 < im.norm.vmin < 3.0             # the background sits at the bottom
    qa.save(fig, tmp_path / "log.png")


def test_imshow_constant_image_keeps_a_well_formed_norm(tmp_path):
    fig, ax = qa.subplots()
    im = qa.imshow(ax, np.full((10, 10), 5.0), stretch="log")
    assert im.norm.vmax == pytest.approx(10 * im.norm.vmin)   # degenerate fallback
    fig2, ax2 = qa.subplots()
    im2 = qa.imshow(ax2, np.full((10, 10), 5.0))           # linear
    assert im2.norm.vmax > im2.norm.vmin
    qa.save(fig, tmp_path / "const_log.png")
    qa.save(fig2, tmp_path / "const_lin.png")


def test_imshow_linear_limits_are_capped_too(tmp_path):
    img = np.zeros((30, 30))
    img[14:17, 14:17] = 10.0
    img[2, 2] = 1e4
    fig, ax = qa.subplots()
    im = qa.imshow(ax, img, percentiles=(0.0, 100.0))
    assert im.norm.vmax == pytest.approx(10.0)  # the source, not the hot pixel
    qa.save(fig, tmp_path / "lin.png")


def test_imshow_explicit_norm_wins(tmp_path):
    from matplotlib.colors import LogNorm

    fig, ax = qa.subplots()
    im = qa.imshow(ax, field(np.random.default_rng(4)), stretch="log",
                      norm=LogNorm(vmin=1.0, vmax=100.0))
    assert im.norm.vmin == 1.0 and im.norm.vmax == 100.0
    qa.save(fig, tmp_path / "norm.png")


def test_collapse_min_coverage_blanks_edge_spaxels():
    cube = np.ones((20, 8, 8))
    cube[:, 0, 0] = np.nan
    cube[1:, 0, 1] = np.nan      # 1/20 planes finite -> nanmedian of noise
    cube[:10, 0, 2] = np.nan     # exactly half covered
    out = qa.collapse(cube, min_coverage=0.5)
    assert np.isnan(out[0, 0]) and np.isnan(out[0, 1])
    assert out[0, 2] == 1.0 and out[5, 5] == 1.0
    full = qa.collapse(cube)  # default: nothing blanked beyond all-NaN
    assert full[0, 1] == 1.0


def _overlaps(texts, renderer, shrink: float = 0.5) -> int:
    """Number of pairs of visible texts whose boxes (shrunk by a pixel fraction) overlap."""
    boxes = [t.get_window_extent(renderer).shrunk_to_aspect(1.0) if False else t.get_window_extent(renderer) for t in texts]
    count = 0
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if a.x0 + shrink < b.x1 and b.x0 + shrink < a.x1 and a.y0 + shrink < b.y1 and b.y0 + shrink < a.y1:
                count += 1
    return count


def _spectrum(lines=(), dips=(), lo=1.7, hi=14.0, n=6000, seed=1):
    """A noisy flat spectrum with Gaussian emission lines and absorption dips at the given wavelengths."""
    wave = np.geomspace(lo, hi, n)
    flux = 1.0 + 0.004 * np.random.default_rng(seed).standard_normal(n)
    for centre, amplitude in [(c, 3.0) for c in lines] + [(c, -0.3) for c in dips]:
        flux += amplitude * np.exp(-0.5 * ((wave - centre) / (6e-4 * centre)) ** 2)
    return wave, flux


def test_annotate_features_marks_every_line_without_overlaps(tmp_path):
    import matplotlib

    from jwstflow.spectra.features import select_features

    fig, ax = qa.subplots()
    wave, flux = _spectrum(lines=(12.81355, 8.3000), dips=(2.20625,))      # [Ne II], an unknown line, Na I
    qa.step(ax, wave, flux, color=qa.MAIN_COLOR)
    ax.set(xlabel=qa.WAVE_LABEL)
    qa.set_wave_scale(ax)
    qa.set_flux_scale(ax, "log")
    qa.annotate_features(ax, "all", wave_min=1.7, wave_max=14.0)
    assert not ax.get_legend_handles_labels()[1]      # the annotation adds nothing to the legend
    assert not ax.texts and not ax.child_axes         # the layout needs the final geometry: nothing is drawn yet
    qa.finalize(fig)
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    (top,) = ax.child_axes                            # broad gas emission above; everything else is in the plot
    box = ax.get_window_extent(renderer)
    assert top.get_window_extent(renderer).y0 >= box.y1 - 1
    assert ax.spines["bottom"].get_visible() and ax.get_xlabel() == qa.WAVE_LABEL     # one frame, one x axis
    assert ax.get_ylim()[0] > 0                       # a log axis: no negative range under the spectrum
    # every line in range is marked, seen or not
    rotated = [t for t in ax.texts if t.get_rotation() == 90]
    inline = {t.get_text() for t in rotated}
    gas = [f for f in select_features({"gas_lines": "all", "gas_lines_extended": "all"}, wave_min=1.7, wave_max=14.0)
           if f.kind == "line"]
    loners = [f for f in gas if _series(f.label) is None]
    assert len(loners) > 30 and all(qa.tex_label(_line_label(f.label)) in inline for f in loners)
    assert "[Fe II] 5.340" in inline                  # not in this spectrum, marked all the same
    # the members of a series are labelled at their lines too; where they converge they share one label
    assert {"H$_{2}$ 1-0 S(1)", r"H I Br$\gamma$", "H$_{2}$ 0-0 S(3)"} <= inline
    assert any("\N{EN DASH}" in t and t.startswith("H I ") for t in inline)
    marks = [t for t in rotated if t.get_text().startswith("?")]
    assert len(marks) == 1 and marks[0].get_color() == qa.REF_COLOR     # the strong line no catalogue explains,
    assert re.fullmatch(r"\? \[8\.\d{3}\]", marks[0].get_text())          # labelled with its wavelength
    assert abs(float(marks[0].get_text()[3:-1]) - 8.3) < 0.003
    spot = marks[0].get_window_extent(renderer)
    assert abs(0.5 * (spot.x0 + spot.x1) - ax.transData.transform((8.3, 1.0))[0]) < 12
    curve = ax.transData.transform((5.0, 1.0))[1]
    above = [t for t in rotated if t.get_va() == "bottom"]
    below = [t for t in rotated if t.get_va() == "top"]
    assert all(t.get_window_extent(renderer).y0 > curve for t in above)
    assert len(below) > 5 and all(t.get_window_extent(renderer).y1 < curve for t in below)    # photosphere
    assert _overlaps([t for t in ax.texts if t.get_visible()], renderer) == 0     # band labels included
    assert _overlaps(list(top.texts), renderer) == 0
    assert all(box.y0 - 1 <= t.get_window_extent(renderer).y0 and t.get_window_extent(renderer).y1 <= box.y1 + 1
               for t in ax.texts)                     # no label leaves the axes
    # labels follow the spectrum: the one over the strong line stands on top of it, its neighbours do not
    neon = next(t for t in rotated if t.get_text() == "[Ne II] 12.814")
    assert neon.get_color() == qa.SPECIES_COLORS["noble gases"]
    peak = ax.transData.transform((12.81355, 4.0))
    extent = neon.get_window_extent(renderer)
    assert abs(0.5 * (extent.x0 + extent.x1) - peak[0]) < 6 and 0 < extent.y0 - peak[1] < 40
    others = [t.get_window_extent(renderer).y0 for t in above if t is not neon and not t.get_text().startswith("?")]
    assert np.median(others) < peak[1]                # ... they stay down at the continuum
    # the band rows are the foot of the plot; all regions of one ice share a row; formulae, no "ice"
    rects = [p for p in ax.patches if isinstance(p, matplotlib.patches.Rectangle)]
    water = [p for p in rects if p.get_facecolor()[:3] == tuple(matplotlib.colors.to_rgb(qa.BAND_COLORS["H2O ice"]))]
    assert len(rects) > 10 and len(water) >= 3 and len({round(p.get_y(), 6) for p in water}) == 1
    flat = {t.get_text() for t in ax.texts if t.get_rotation() == 0}
    assert "H$_{2}$O" in flat and "CO$_{2}$" in flat and not any("ice" in t for t in flat)
    low = min(ax.transData.transform((x, y))[1] for x, y in zip(wave, flux))
    assert all(ax.transAxes.transform((0, p.get_y() + p.get_height()))[1] < low for p in rects)   # under the spectrum
    width, height = fig.get_size_inches()
    assert width > height and box.height > 0.6 * fig.bbox.height    # wider than tall; the spectrum dominates
    qa.save(fig, tmp_path / "annot.png")


def test_series_as_combs_is_a_separate_layout(tmp_path):
    fig, ax = qa.subplots()
    wave, flux = _spectrum(lo=1.7, hi=5.3)
    qa.step(ax, wave, flux, color=qa.MAIN_COLOR)
    qa.set_wave_scale(ax)
    qa.annotate_features(ax, "all", wave_min=1.7, wave_max=5.3, series="combs")
    qa.finalize(fig)
    fig.draw_without_rendering()
    top = ax.child_axes[0]
    names = {t.get_text() for t in top.texts}
    assert {qa.tex_label("H2 1-0 S"), qa.tex_label("H2 0-0 S"), "H I Br", "H I Pf"} <= names
    assert not any(t.get_text().startswith("H$_{2}$ 1-0 S(") for t in ax.texts)     # no label per member in the plot
    assert len(top.lines) > 80                        # a tick for every member of the series
    assert _overlaps(list(top.texts), fig.canvas.get_renderer()) == 0
    qa.save(fig, tmp_path / "combs.png")


def _annotated(tmp_path, name: str, *, detected_only: bool, lines=(), dips=(), lo=1.7, hi=14.0):
    """A synthetic spectrum annotated with every dataset; the axes after the layout ran."""
    fig, ax = qa.subplots()
    wave, flux = _spectrum(lines=lines, dips=dips, lo=lo, hi=hi)
    qa.step(ax, wave, flux, color=qa.MAIN_COLOR)
    qa.set_wave_scale(ax)
    qa.set_flux_scale(ax, "log")
    qa.annotate_features(ax, "all", wave_min=lo, wave_max=hi, detected_only=detected_only)
    qa.finalize(fig)
    fig.draw_without_rendering()
    qa.save(fig, tmp_path / f"{name}.png")
    return ax


def test_detected_only_labels_what_the_spectrum_shows(tmp_path):
    # [Ne II], H2 0-0 S(3), an unknown line, Na I in absorption -- and a forest of unknown lines
    # inside the CO fundamental band (4.4-5.2 um), none inside the water band at 5.5-7.3 um
    forest = tuple(4.5 + 0.035 * k for k in range(8))
    shown = {"lines": (12.81355, 9.66491, 8.3000, *forest), "dips": (2.20625,)}
    every = _annotated(tmp_path, "every", detected_only=False, **shown)
    found = _annotated(tmp_path, "found", detected_only=True, **shown)

    def texts(ax, rotation):
        return {t.get_text() for t in ax.texts if t.get_rotation() == rotation}

    inline = texts(found, 90)
    assert {"[Ne II] 12.814", "H$_{2}$ 0-0 S(3)"} <= inline              # the lines the spectrum shows
    assert any(t.startswith("Na I") for t in inline)                     # ... the photospheric dip among them
    assert "[Fe II] 5.340" not in inline and "[Fe II] 5.340" in texts(every, 90)     # absent: only in the full figure
    assert len(inline) < 15 < len(texts(every, 90))
    # the unidentified peaks are labelled alike in both figures
    assert {t for t in inline if t.startswith("?")} == {t for t in texts(every, 90) if t.startswith("?")} != set()
    # gas bands: kept where the spectrum shows their forest, dropped elsewhere; solid-state bands kept
    top_found = {t.get_text() for child in found.child_axes for t in child.texts}
    top_every = {t.get_text() for child in every.child_axes for t in child.texts}
    assert any("CO fundamental" in t for t in top_found)
    assert any("$\\nu_{2}$" in t for t in top_every) and not any("$\\nu_{2}$" in t for t in top_found)
    assert texts(found, 0) == texts(every, 0) != set()                   # the ice rows are the same


def test_detected_only_without_a_searchable_spectrum_labels_nothing(tmp_path):
    fig, ax = qa.subplots()
    wave = np.linspace(2.0, 2.4, 60)                 # too short to search for lines
    qa.step(ax, wave, np.ones_like(wave), color=qa.MAIN_COLOR)
    qa.annotate_features(ax, "all", wave_min=2.0, wave_max=2.4, detected_only=True)
    qa.finalize(fig)
    assert not [t for t in ax.texts if t.get_rotation() == 90]


def test_a_peak_identifies_a_line_within_one_sample_never_across_a_gap():
    wave, flux = _spectrum(lines=(3.0, 4.06), lo=2.9, hi=5.2, n=8000)
    keep = (wave < 4.05) | (wave > 4.15)             # a detector gap right of 4.05 um
    wave, flux = wave[keep], flux[keep]
    edge = int(np.searchsorted(wave, 4.05))         # the first sample after the gap
    flux[edge - 2:edge] += 0.5                       # the detector edge rises before the gap: no line
    found = qa.detect_lines(wave, flux)
    assert not np.any(np.abs(found["x_em"] - 4.05) < 0.01)
    step = np.median(np.diff(wave) / wave[1:])
    # the tolerance is one sample or IDENTIFY_VELOCITY, whichever is larger -- also next to the gap
    assert np.all(found["tol_em"] <= np.maximum(1.01 * step * found["x_em"],
                                                qa.IDENTIFY_VELOCITY / 2.998e5 * found["x_em"] * 1.001))
    assert abs(found["x_em"][np.argmin(np.abs(found["x_em"] - 3.0))] - 3.0) <= found["tol_em"].max()


def test_a_shared_label_names_exactly_the_members_with_stems():
    assert _member_span(["25", "24", "23", "22"]) == "22\N{EN DASH}25"
    assert _member_span(["17", "19", "20", "22"]) == "17, 19\N{EN DASH}20, 22"
    assert _member_span([r"$\alpha$", r"$\beta$"]) == "$\\alpha$\N{EN DASH}$\\beta$"     # Greek ticks: first to last


def _stems_through_labels(ax, renderer) -> list[tuple[str, str]]:
    """(label, owner of the stem) for every connector that passes through a label not its own."""
    from matplotlib.patches import PathPatch

    labels = [t for t in ax.texts if t.get_rotation() == 90 and t.get_visible()]
    boxes = [t.get_window_extent(renderer) for t in labels]
    found = []
    for patch in (p for p in ax.patches if isinstance(p, PathPatch)):
        path = patch.get_transform().transform_path(patch.get_path())
        for poly in path.to_polygons(closed_only=False):
            end = poly[-1]
            own = next((i for i, b in enumerate(boxes) if b.x0 - 2 <= end[0] <= b.x1 + 2
                        and min(abs(end[1] - b.y0), abs(end[1] - b.y1)) < 5), None)
            dense = np.concatenate([a + np.linspace(0.0, 1.0, 12)[:, None] * (b - a) for a, b in pairwise(poly)])
            for i, b in enumerate(boxes):
                inside = (dense[:, 0] > b.x0 + 1) & (dense[:, 0] < b.x1 - 1) & (dense[:, 1] > b.y0 + 1) & (dense[:, 1] < b.y1 - 1)
                if i != own and inside.any():
                    found.append((labels[i].get_text(), labels[own].get_text() if own is not None else "?"))
    return found


#: Emission lines of an H2-rich NIRSpec spectrum (HV Tau C, 1.6-2.75 and 3.4-3.6 um): wavelength
#: [um] and peak over continuum. Strong lines with weak ones in their label's column, series with
#: other lines between their members, unknown peaks beside known lines -- where connectors crossed.
CROWDED_LINES = (
    (1.66416, 0.126), (1.67762, 0.227), (1.68792, 0.133), (1.71485, 0.124), (1.74811, 0.648), (1.78811, 0.432),
    (1.79761, 0.151), (1.80038, 0.175), (1.80989, 0.3), (1.83602, 1.873), (1.85265, 0.136), (1.87562, 0.827),
    (1.89186, 0.972), (1.94492, 0.352), (1.95759, 4.564), (2.00432, 0.148), (2.01581, 0.07), (2.03402, 1.621),
    (2.0657, 0.071), (2.07362, 0.572), (2.12193, 5.473), (2.13342, 0.032), (2.15441, 0.215), (2.16629, 0.063),
    (2.20153, 0.114), (2.22331, 1.257), (2.24786, 0.584), (2.38646, 0.181), (2.47556, 0.985), (2.50011, 2.145),
    (2.52823, 0.471), (2.5512, 0.565), (2.55872, 0.214), (2.55991, 0.947), (2.56981, 0.523), (2.58525, 0.185),
    (2.59555, 0.173), (2.60426, 0.495), (2.62723, 2.098), (2.63515, 0.385), (2.65416, 0.387), (2.6795, 0.191),
    (2.68505, 0.102), (2.71039, 0.126), (2.71118, 0.163), (2.72029, 0.273), (2.72702, 0.169), (2.73138, 0.131),
    (3.40433, 0.203), (3.43824, 0.42), (3.45487, 0.042), (3.48612, 0.403), (3.50075, 0.688), (3.534, 0.044),
    (3.54797, 0.172), (3.59252, 0.023),
)


def test_no_connector_passes_through_a_label(tmp_path):
    wave = np.geomspace(1.6, 3.65, 9000)       # NIRSpec-like sampling (~60 km/s), R ~ 2700 lines
    flux = 1.0 + 0.004 * np.random.default_rng(4).standard_normal(wave.size)
    for centre, amplitude in CROWDED_LINES:
        flux += amplitude * np.exp(-0.5 * ((wave - centre) / (1.6e-4 * centre)) ** 2)
    for detected_only in (False, True):
        fig, ax = qa.subplots()
        qa.step(ax, wave, flux, color=qa.MAIN_COLOR)
        qa.set_wave_scale(ax)
        qa.set_flux_scale(ax, "log")
        qa.annotate_features(ax, "all", wave_min=1.6, wave_max=3.65, detected_only=detected_only)
        qa.finalize(fig)
        fig.draw_without_rendering()
        renderer = fig.canvas.get_renderer()
        assert _stems_through_labels(ax, renderer) == []
        assert _overlaps([t for t in ax.texts if t.get_visible()], renderer) == 0
        qa.save(fig, tmp_path / f"crowded_{detected_only}.png")


def test_a_log_axis_without_positive_data_falls_back_to_linear(tmp_path):
    fig, ax = qa.subplots()
    qa.step(ax, np.linspace(1.0, 2.0, 50), np.zeros(50), color=qa.MAIN_COLOR)
    qa.set_wave_scale(ax)
    qa.set_flux_scale(ax, "log")
    qa.save(fig, tmp_path / "empty.png")           # used to raise in the log locator
    assert ax.get_yscale() == "linear"


def test_a_flat_spectrum_on_a_log_axis_gets_a_bounded_number_of_ticks(tmp_path):
    """A (nearly) constant curve leaves a view a few 1e-9 wide before the limits are set; a minor
    tick step fixed from that view asked for billions of ticks once the limits were widened."""
    wave = np.linspace(5.0, 6.5, 200)
    plot = qa.Spectrum(xscale="log", yscale="log")
    plot.line(wave, np.full(200, 1.0) + 1e-9 * np.arange(200) / 200, label="flat")
    plot.decorate(ylabel=qa.FLUX_LABEL)
    qa.finalize(plot.fig)
    lo, hi = plot.ax.get_ylim()
    assert lo < 0.7 and hi > 1.5                                 # the robust limits, set before the ticks were chosen
    for axis in (plot.ax.yaxis, plot.ax.xaxis):
        minor, major = axis.get_minor_locator(), axis.get_major_locator()
        step = getattr(getattr(minor, "_edge", None), "step", None)      # a fixed step must not be finer than the view
        assert step is None or (hi - lo) / step < 1000
        assert 2 <= len(major.tick_values(*sorted(axis.get_view_interval()))) <= 15
        assert len(minor.tick_values(*sorted(axis.get_view_interval()))) <= 100
    assert qa.save(plot.fig, tmp_path / "flat.png").stat().st_size > 5_000


def test_detect_lines_and_typeset_labels():
    wave, flux = _spectrum(lines=(4.05226,), dips=(2.20625,), lo=1.7, hi=5.3)
    found = qa.detect_lines(wave, flux)
    assert found["x_em"].size == 1 and abs(found["x_em"][0] - 4.05226) < found["tol_em"][0]
    assert found["x_ab"].size == 1 and abs(found["x_ab"][0] - 2.20625) < found["tol_ab"][0]
    assert qa.detect_lines(wave[:50], flux[:50]) is None          # too short to search
    assert qa.tex_label("13CO2 ice stretch") == "$^{13}$CO$_{2}$ ice stretch"
    assert qa.tex_label("NH4+") == "NH$_{4}$$^{+}$" and qa.tex_label("H I Br-gamma") == r"H I Br$\gamma$"
    assert qa.tex_label("[Fe II] 5.340") == "[Fe II] 5.340" and qa.tex_label("H2 1-0 S(1)") == "H$_{2}$ 1-0 S(1)"
    assert _series("H I Br 10-4") == ("H I Br", "10") and _series("H2 1-0 Q(3)") == ("H2 1-0 Q", "3")
    assert _series("[Ne II] 12.814") is None
    assert qa.FLUX_LABEL == r"$F_\nu$ [mJy]"                       # symbols, not names, on the axes


def test_labels_of_weak_lines_go_first_where_lines_crowd():
    wish = np.array([10.0, 12.0, 14.0, 60.0])
    positions, kept = _declutter(wish, np.full(4, 8.0), 0.0, 100.0, max_shift=3.0,
                                       priority=np.array([5.0, 0.1, 4.0, 1.0]))
    assert list(kept) == [True, False, True, True]   # the weakest of the crowd, not the isolated one
    assert np.all(np.diff(positions[kept]) >= 8.0 - 1e-9)


def test_species_colors_group_the_lines():
    assert qa.species_color("FeII") == qa.species_color("SIV") == qa.SPECIES_COLORS["metals"]
    assert qa.species_color("NeII") == qa.species_color("ArIII") == qa.SPECIES_COLORS["noble gases"]
    assert qa.species_color("13CO2") == qa.species_color("CO") == qa.SPECIES_COLORS["CO"]
    assert qa.species_color("SiO") == qa.SPECIES_COLORS["other"]       # a molecule outside the groups
    assert _line_label("H2O cold diagnostic 23.8-24.0") == "H2O cold" and _line_label("C2H6 Q-branch") == "C2H6"


def test_annotate_features_respects_the_range_and_none(tmp_path):
    fig, ax = qa.subplots()
    ax.plot([0.5, 0.6], [1.0, 1.0])
    qa.annotate_features(ax, "all", wave_min=0.5, wave_max=0.6)  # nothing lives here
    qa.annotate_features(ax, None, wave_min=1.0, wave_max=30.0)  # no selection: no-op
    qa.finalize(fig)
    assert not ax.texts
    qa.save(fig, tmp_path / "annot_empty.png")


def test_photospheric_lines_are_opt_in_for_masking_but_annotated(tmp_path):
    from jwstflow.spectra.features import DATASETS, select_features

    assert "photospheric_lines" not in DATASETS and "gas_lines_extended" not in DATASETS
    assert {f.dataset for f in select_features("all")} == set(DATASETS)                    # `all` is unchanged
    further = select_features({"gas_lines_extended": "all"})
    assert len(further) > 60 and {"HI", "H2", "HeI", "FeII"} <= {f.species for f in further}
    assert not {f.id for f in further} & {f.id for f in select_features({"gas_lines": "all"})}
    stellar = select_features({"photospheric_lines": "all"}, wave_min=2.0, wave_max=2.4)
    assert {"NaI", "CaI", "CO"} <= {f.species for f in stellar}
    fig, ax = qa.subplots()
    wave = np.linspace(2.0, 2.4, 300)
    qa.step(ax, wave, np.ones_like(wave), color=qa.MAIN_COLOR)
    qa.annotate_features(ax, {"gas_lines": "all"}, wave_min=2.0, wave_max=2.4)
    assert abs(next(f.wavelength for f in stellar if f.id.startswith("nai_2.2062")) - 2.20625) < 2e-5   # NIST Ritz
    assert {"cai_1.9782", "nai_1.8470"} <= {f.id for f in select_features({"photospheric_lines": "all"})}
    fig2, ax2 = qa.subplots()
    qa.step(ax2, wave, np.ones_like(wave), color=qa.MAIN_COLOR)
    qa.annotate_features(ax2, {"gas_lines": "all"}, wave_min=2.0, wave_max=2.4, photospheric=False)
    qa.finalize(fig)
    qa.finalize(fig2)
    assert any(t.get_text() == "Na I doublet" for t in ax.texts)          # 2.2062 + 2.2090: one label, two ticks
    assert not any(t.get_text().startswith("Na I") for t in ax2.texts)
    qa.save(fig, tmp_path / "stellar.png")
    qa.save(fig2, tmp_path / "no_stellar.png")


def test_step_breaks_the_line_at_gaps_but_not_at_a_change_of_sampling():
    x = np.r_[np.arange(1.0, 2.0, 0.01), np.arange(2.5, 3.0, 0.01), np.arange(3.0, 5.0, 0.05)]
    xs, ys = qa.break_gaps(x, np.ones_like(x))
    assert np.isnan(ys).sum() == 2                       # the 2.0-2.5 gap, one NaN inside each edge
    assert np.all(np.diff(xs) > 0)
    gap = xs[np.isnan(ys)]
    assert 1.99 < gap[0] < 2.02 and 2.48 < gap[1] < 2.5
    xs2, ys2 = qa.break_gaps(np.arange(10.0), np.ones(10))
    assert not np.isnan(ys2).any() and xs2.size == 10    # regular sampling: untouched


def test_imshow_shows_missing_data_as_missing_and_adds_sky_axes(tmp_path):
    from astropy.io import fits

    img = field(np.random.default_rng(5), hot=False)
    header = fits.Header({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": 10.0, "CRVAL2": -20.0,
                          "CRPIX1": 17.0, "CRPIX2": 16.0, "CDELT1": 2.8e-5, "CDELT2": 2.8e-5,
                          "PC1_1": -1.0, "PC1_2": 0.0, "PC2_1": 0.0, "PC2_2": 1.0, "TARG_RA": 10.0, "TARG_DEC": -20.0})
    sky = qa.SkyFrame.from_header(header, img.shape)
    assert sky is not None and sky.north_up and sky.sx < 0 < sky.sy       # East to the left
    assert sky.reference == "target" and abs(sky.x0 - 16.0) < 1e-3 and abs(sky.y0 - 15.0) < 1e-3
    assert qa.SkyFrame.from_header(fits.Header(), img.shape) is None    # no WCS: pixel axes only
    fig, ax = qa.subplots(figsize=qa.FIGSIZE["image"])
    im = qa.imshow(ax, img, unit="MJy/sr", stretch="asinh", sky=sky)
    assert tuple(im.cmap.get_bad()[:3]) != tuple(im.cmap(0.0)[:3])        # NaN is not the lowest color
    assert ax.get_xlabel() == qa.DX_PIX_LABEL and len(ax.child_axes) == 3  # two arcsec axes + the colorbar
    # both sets of axes are offsets from one zero: the pixel "0" sits at the reference pixel, where the sky offset is 0
    fig.canvas.draw()
    zero = {t.get_text(): t.get_position()[0] for t in ax.get_xticklabels()}["0"]
    top = ax.child_axes[0]
    assert abs(zero - sky.x0) < 1e-6 and min(abs(top.get_xticks())) == 0.0
    assert abs(top.get_xaxis_transform().transform((0.0, 0))[0] - ax.transData.transform((sky.x0, 0))[0]) < 0.5
    assert "pixel (16.0, 15.0)" in sky.note()
    plain = qa.subplots()[1]
    qa.imshow(plain, img)                                                 # no WCS: plain pixel coordinates
    assert plain.get_xlabel() == qa.X_PIX_LABEL and len(plain.child_axes) == 1
    log_im = qa.imshow(qa.subplots()[1], np.where(img > 3.0, -1.0, img), stretch="log")
    assert np.nanmin(log_im.get_array()) >= log_im.norm.vmin              # <= 0 is data: lowest color, not a gap
    qa.save(fig, tmp_path / "sky.png")


def test_figlegend_puts_the_notes_first_and_fits_the_figure(tmp_path):
    fig, ax = qa.subplots(figsize=qa.FIGSIZE["image"])
    ax.plot([0, 1], [0, 1], label="a line")
    qa.annotate(ax, "product_name_of_some_length_s3d.fits", "a second note")
    legend = qa.figlegend(fig)
    texts = [t.get_text() for t in legend.get_texts()]
    assert texts[0].splitlines() == ["product_name_of_some_length_s3d.fits", "a second note"]   # the notes column
    assert "a line" in texts
    assert ax.get_title() == "" and fig._suptitle is None                 # rule 1: no titles
    qa.save(fig, tmp_path / "legend.png")


def test_short_labels_strip_whole_name_parts_only():
    from jwstflow.qa import short_labels as _short_labels

    names = ["jw01751-o006_t010_nirspec_g235h-f170lp_extended_s1d", "jw01751-o006_t010_nirspec_g395h-f290lp_extended_s1d"]
    assert _short_labels(names) == ["g235h-f170lp", "g395h-f290lp"]
    assert _short_labels(["only_one"]) == ["only_one"]


def test_grid_panels_label_only_the_named_sides(tmp_path):
    sky = qa.SkyFrame.from_scale(0.1, (32, 33))
    fig, axes = qa.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
    qa.Frame(axes[0]).image(field(np.random.default_rng(2), hot=False), sky=sky, cbar=False, sides="lbt")
    qa.Frame(axes[1]).image(field(np.random.default_rng(3), hot=False), sky=sky, sides="btr")
    assert axes[0].get_ylabel() == qa.DY_PIX_LABEL and axes[1].get_ylabel() == ""
    right = [child for child in axes[0].child_axes if child.get_ylabel() or child.yaxis.get_ticks_position() == "right"]
    assert all(not child.get_ylabel() for child in right)                 # ticks, but no label on the inner side
    assert qa.save(fig, tmp_path / "grid.png").exists()


def test_spectrum_and_frame_are_the_two_figures_to_build_on(tmp_path):
    x, y = _spectrum(lines=[4.0523, 2.1218], lo=1.7, hi=5.2, n=3000)
    plot = qa.Spectrum()
    plot.line(x, y * 0.5, color=qa.ROLE_COLORS["context"], context=True, label="input")
    plot.line(x, y, error=np.full_like(y, 0.02), label="product", error_label=qa.SIGMA_BAND_LABEL)
    ax = plot.ax
    png = plot.finish(tmp_path / "spectrum.png", ylabel=qa.FLUX_LABEL, notes=["a spectrum"], features="all")
    assert png.exists() and ax.get_xscale() == "log" and ax.get_yscale() == "log" and ax.get_ylim()[0] > 0
    assert ax.get_xlabel() == qa.WAVE_LABEL and any(t.get_rotation() == 90 for t in ax.texts)   # annotated
    zoom = qa.Spectrum(panels=2, height_ratios=[2, 1])
    zoom.line(x, y)
    zoom.line(x, y / y, ax=zoom.axes[1])
    lower = zoom.axes[1]
    zoom.finish(tmp_path / "zoom.png", ylabel=qa.FLUX_LABEL, wave_range=(3.8, 4.3))
    assert tuple(round(v, 6) for v in lower.get_xlim()) == (3.8, 4.3)

    frame = qa.Frame()
    img = field(np.random.default_rng(4), hot=False)
    frame.image(img, unit="MJy/sr", sky=qa.SkyFrame.from_scale(0.1, img.shape))
    frame.mask(np.isfinite(img) & (img > 3.0), label="source")
    frame.mask(np.isfinite(img), label="footprint")
    frame.mark(16.0, 15.0)
    assert "zero of both axes" in frame.note()
    assert frame.finish(tmp_path / "frame.png", notes=["a frame"]).exists()


def test_the_package_exports_the_figure_classes_and_helpers():
    assert {"Spectrum", "Frame", "SkyFrame", "annotate_features", "read_cube", "to_mjy"} <= set(qa.__all__)
