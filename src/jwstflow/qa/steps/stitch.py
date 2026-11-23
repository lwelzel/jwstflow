"""Figures of stitched spectra (``*_s1dcomb.ecsv``): the combination over its segments,
the overlaps where the scale factors were measured, and the background that was removed."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ... import qa
from ...steps import RunContext, Step
from .spectra import all_spectra_figure

log = logging.getLogger(__name__)


def _read_stitch(path: Path) -> dict[str, Any]:
    """A stitched spectrum in mJy with what its metadata records about the segments."""
    from astropy.table import Table

    tab = Table.read(path)
    flux, ylabel = qa.to_mjy(np.asarray(tab["FLUX"], dtype=float), str(tab["FLUX"].unit or "Jy"))
    meta = tab.meta
    names = [str(n) for n in meta.get("inputs", [])]
    return {
        "wave": np.asarray(tab["WAVELENGTH"], dtype=float), "flux": flux, "ylabel": ylabel, "meta": meta,
        "names": names,
        "labels": [str(s) for s in meta.get("segments", names)],
        "scales": [float(s) for s in meta.get("scales_applied", [1.0] * len(names))],
        "notes": {str(k): str(v) for k, v in (meta.get("scale_notes") or {}).items()},
        "inherited": {str(k): v for k, v in (meta.get("inherited_scales") or {}).items()},
        "crossovers": [float(b) for b in meta.get("crossovers_um", [])],
        "segment_of": np.asarray(tab["SEGMENT"], dtype=str) if "SEGMENT" in tab.colnames
        else np.full(len(tab), "", dtype=object),
    }


class PlotStitch(Step):
    """Comparison figures of a stitched spectrum (``*_s1dcomb.ecsv``): the combination
    in black over its rescaled input segments -- and, whenever the stitch rescaled
    anything, a second figure of the segments *as extracted* (``*_unscaled.png``),
    before the multiplicative scales were applied, so the raw flux offsets between
    neighbouring segments stay visible (``unscaled: false`` turns it off).

    The segment files named in the ECSV metadata are searched in
    ``segments_stage`` (when given), in every stage directory of the run,
    next to the ECSV itself, and in the sibling runs of the target (so a
    combination run stitching across runs still finds its segments);
    segments that are not found any more are simply left out of the figure.
    The wavelength axis is logarithmic unless ``xscale: linear``, the flux
    axis logarithmic too unless ``yscale: linear``. With two or more stitched spectra a
    combined overview figure (``<stage>_all.png``) draws every one on a
    single log-log axis, coloured from the standard colormap. With
    ``features`` the selected spectral features (jwstflow ``features:``
    grammar; ``all`` takes every bundled dataset in range) are annotated on
    every figure (``qa.annotate_features``: every line labelled at its own
    wavelength; strong peaks no catalogue line explains as ``? [wavelength]``),
    a second figure (``*_detected.png``) labels only the lines the stitched
    spectrum shows -- and the gas bands only where it shows their forest
    (``detected: false`` leaves it out) -- and a third (``*_series.png``)
    shows the spectrum with the line series drawn as combs above the plot.
    """

    level = "qa"
    batch = "all"   # all inputs in one task, so the combined overview sees every spectrum
    inputs = ("*_s1dcomb.ecsv",)
    version = "8"   # 7 -> 8: a figure of the detected lines; "? [wavelength]" for unidentified peaks

    def run(self, inputs: list[Path], ctx: RunContext, *, segments_stage: str | None = None,
            unscaled: bool = True, xscale: str = "log", yscale: str = "log",
            features: Any = None, label_offset: float = qa.FEATURE_LABEL_OFFSET,
            photospheric: bool = True, detected: bool = True, dpi: int = 150, **_: Any) -> Iterable[Path]:
        out: list[Path] = []
        combined: list[tuple[np.ndarray, np.ndarray, str, str]] = []
        annotation = {"features": features, "label_offset": label_offset, "photospheric": photospheric}
        for inp in sorted(inputs):
            s = _read_stitch(inp)
            combined.append((s["wave"], s["flux"], inp.stem, s["ylabel"]))
            colors = qa.line_colors(max(len(s["names"]), 2))
            segments = []   # (label, scale, color, wavelength, flux, unit) of every segment still on disk
            for name, label, scale, color in zip(s["names"], s["labels"], s["scales"], colors):
                segment = qa.find_file(name, ctx, segments_stage, inp.parent)
                if segment is None:
                    log.warning("%s: segment %s not found in any stage directory; left out", inp.name, name)
                    continue
                w, f, _, unit = qa.read_spectrum(segment)
                segments.append((label, scale, color, w, f, unit))
            many = len(segments) > 4      # then the legend names the family, the overlap figure the factors

            # the combination over its rescaled segments: the overview with every line labelled at
            # its place and -- with features -- one with only the lines the spectrum shows, and
            # one with the series as combs
            variants = [("labels", "", False)]
            if features not in (None, False):
                variants += [("labels", "_detected", True)] if detected else []
                variants += [("combs", "_series", False)]
            for series, suffix, detected_only in variants:
                plot = qa.Spectrum(xscale=xscale, yscale=yscale)
                for k, (label, scale, color, w, f, unit) in enumerate(segments):
                    plot.line(w, qa.to_mjy(f * scale, unit)[0], color=color, alpha=0.7, context=True,
                              label=(f"{len(segments)} segments" + (", rescaled" if any(abs(sc - 1.0) > 1e-9 for _, sc, *_ in segments) else "")
                                     if k == len(segments) // 2 else None) if many
                              else label + (f" {qa.times(scale)}" if scale != 1.0 else "")
                              + (f" ({s['notes'][label]})" if label in s["notes"] else "")
                              + "".join(f" (incl. {inner} {qa.times(v['factor'])} +- {v['error']:g})"
                                        for inner, v in s["inherited"].get(label, {}).items()))
                plot.line(s["wave"], s["flux"], label="stitched")
                plot.crossovers(s["crossovers"])
                out.append(plot.finish(ctx.output_dir / f"{inp.stem}{suffix}.png", ylabel=s["ylabel"], notes=[inp.name],
                                       series=series, detected_only=detected_only, dpi=dpi, **annotation))

            # the segments as extracted, before the multiplicative rescaling -- only
            # when a scale was actually applied (otherwise it repeats the figure above)
            if unscaled and any(scale != 1.0 for _, scale, *_ in segments):
                plot = qa.Spectrum(xscale=xscale, yscale=yscale)
                ylabel = s["ylabel"]
                for k, (label, _, color, w, f, unit) in enumerate(segments):
                    f, ylabel = qa.to_mjy(f, unit)
                    plot.line(w, f, color=color, context=True, lw=qa.LW["main"],
                              label=(f"{len(segments)} segments" if k == len(segments) // 2 else None) if many else label)
                plot.cover(s["wave"])       # annotated over the range of the combination, like the figure above
                plot.crossovers(s["crossovers"])
                out.append(plot.finish(ctx.output_dir / f"{inp.stem}_unscaled.png", ylabel=ylabel, dpi=dpi,
                                       notes=[f"{inp.name}: segments as extracted (before rescaling)"], **annotation))
        overview = all_spectra_figure(combined, ctx.output_dir / f"{ctx.stage}_all.png", what="stitched spectra",
                                      dpi=dpi, **annotation)
        if overview is not None:
            out.append(overview)
        return out


class PlotStitchOverlaps(Step):
    """Per-overlap figures of a stitched spectrum: where the multiply factors were measured.

    For each ``*_s1dcomb.ecsv`` one figure (``*_overlaps.png``) with a panel
    per neighbouring segment pair, zoomed into their overlap: both segments
    *as extracted* (with their FLUX_ERROR bands), the crossover wavelength
    where the combination switches segments (dotted), and -- shaded -- the
    wavelength window around the crossover from which the stitch measured
    the multiply factor (``ratio_window_frac`` of the crossover wavelength,
    clipped to the overlap; ``ratio_windows_um`` in the ECSV metadata). The
    bluer segment times the measured factor is overlaid dashed, so the
    quality of the factor is visible exactly where it was measured, and a
    note inside each panel names the pair and states the factor with its
    1-sigma uncertainty (the legend explains the marks common to all panels).
    Pairs whose ratio was not measurable (no overlap, too few samples, flux
    consistent with zero) say so in their panel.

    The segment files named in the ECSV metadata are located like
    ``plot_stitch`` does (``segments_stage``, every stage directory, next to
    the ECSV, sibling runs of the target); missing segments leave their
    panel annotated instead of drawn. Panels are zoomed views of a narrow
    wavelength range, so both axes default to linear (``xscale``/``yscale``
    switch them).
    """

    level = "qa"
    batch = "all"
    inputs = ("*_s1dcomb.ecsv",)
    version = "2"

    def run(self, inputs: list[Path], ctx: RunContext, *, segments_stage: str | None = None,
            xscale: str = "linear", yscale: str = "linear", dpi: int = 150, **_: Any) -> Iterable[Path]:
        from astropy.table import Table

        out: list[Path] = []
        for inp in sorted(inputs):
            meta = Table.read(inp).meta
            names = [str(n) for n in meta.get("inputs", [])]
            labels = [str(s) for s in meta.get("segments", names)]
            crossovers = [float(b) for b in meta.get("crossovers_um", [])]
            ratios = meta.get("neighbour_ratios", [None] * len(crossovers))
            ratio_errors = meta.get("neighbour_ratio_errors", [None] * len(crossovers))
            windows = meta.get("ratio_windows_um", [None] * len(crossovers))
            if len(names) < 2 or len(crossovers) != len(names) - 1:
                log.warning("%s: no neighbour-pair metadata; skipped", inp.name)
                continue
            colors = qa.line_colors(len(names)) if len(names) > 1 else qa.line_colors(2)
            segments: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
            for name in names:
                path = qa.find_file(name, ctx, segments_stage, inp.parent)
                if path is None:
                    log.warning("%s: segment %s not found in any stage directory; its panels "
                                "are annotated instead of drawn", inp.name, name)
                    segments.append(None)
                    continue
                w, f, e, unit = qa.read_spectrum(path)
                f, _ = qa.to_mjy(f, unit)
                e, _ = qa.to_mjy(e, unit)
                segments.append((w, f, e))

            n_pairs = len(names) - 1
            ncols = min(3, n_pairs)
            nrows = (n_pairs + ncols - 1) // ncols
            fig, axes = qa.subplots(nrows, ncols, figsize=(max(4.6 * ncols, 7.0), 3.4 * nrows), squeeze=False)
            qa.annotate(axes[0][0], inp.name)
            for i in range(n_pairs):
                ax = axes[i // ncols][i % ncols]
                overlap_panel(ax, i, labels, colors, segments, crossovers[i],
                              None if windows[i] is None else (float(windows[i][0]), float(windows[i][1])),
                              None if ratios[i] is None else float(ratios[i]),
                              None if ratio_errors[i] is None else float(ratio_errors[i]),
                              xscale=xscale, yscale=yscale,
                              applied=[f"{lab} x {float(v[0]):g} +- {float(v[1]) if len(v) > 1 else 0:g} (fixed)"
                                       for lab, v in (meta.get("fixed_scales") or {}).items()
                                       if lab in (labels[i], labels[i + 1])])
                if i % ncols == 0:
                    ax.set_ylabel(qa.FLUX_LABEL)
                if i // ncols == nrows - 1 or i + ncols >= n_pairs:
                    ax.set_xlabel(qa.WAVE_LABEL)
            for i in range(n_pairs, nrows * ncols):
                axes[i // ncols][i % ncols].set_axis_off()
            overlap_legend(axes[0][0], meta.get("ratio_window_frac"))
            qa.figlegend(fig)
            out.append(qa.save(fig, ctx.output_dir / f"{inp.stem}_overlaps.png", dpi=dpi))
        return out


def overlap_panel(ax: Any, i: int, labels: list[str], colors: list[Any],
                  segments: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None],
                  crossover: float, window: tuple[float, float] | None,
                  ratio: float | None, ratio_error: float | None, *,
                  xscale: str = "linear", yscale: str = "linear", applied: list[str] | None = None) -> None:
    """One neighbour pair (segments ``i`` and ``i + 1``) around its overlap: both segments as
    measured -- ``(wavelength, flux, error)`` in mJy, None for a missing one -- with their error
    bands, the crossover, the ratio window, the bluer segment times the measured factor, and a
    note naming the pair and the factor. Stitched spectra and joint cubes draw their overlaps
    with it."""
    seg_a, seg_b = segments[i], segments[i + 1]
    # the region of interest: the overlap (or the gap), padded -- always
    # containing the crossover and the measurement window
    edges = [crossover]
    if seg_a is not None and seg_b is not None:
        edges += [float(seg_b[0].min()), float(seg_a[0].max())]
    if window is not None:
        edges += [window[0], window[1]]
    lo, hi = min(edges), max(edges)
    pad = 0.75 * (hi - lo) or 0.01 * crossover
    view = (lo - pad, hi + pad)

    for seg, label, color in ((seg_a, labels[i], colors[i]), (seg_b, labels[i + 1], colors[i + 1])):
        if seg is None:
            continue
        w, f, e = seg
        show = (w >= view[0]) & (w <= view[1])
        qa.step(ax, w[show], f[show], color=color)
        if (show & np.isfinite(e)).any():
            qa.error_band(ax, w[show], f[show], e[show], color=color)
    if ratio is not None and seg_a is not None and seg_b is not None:
        w, f, _ = seg_a
        inside = (w >= float(seg_b[0].min())) & (w <= float(seg_a[0].max()))
        if inside.any():
            qa.step(ax, w[inside], f[inside] * ratio, color=colors[i], ls="--")
    qa.refline(ax, x=crossover, ls=":")
    if window is not None:
        qa.span(ax, window[0], window[1], role="window")
    measured = (rf"$\times$ {qa.pm(ratio, ratio_error)}" if ratio is not None and ratio_error is not None
                else "ratio not measured" if ratio is None else qa.times(ratio))
    missing = [labels[i + k] for k, seg in ((0, seg_a), (1, seg_b)) if seg is None]
    qa.panel_note(ax, f"{labels[i]} | {labels[i + 1]}: {'measured ' if applied else ''}{measured}"
                     + "".join(f"\napplied: {a}" for a in (applied or []))
                     + (f"\nsegment file missing: {', '.join(missing)}" if missing else ""))
    ax.set_xlim(*view)
    if xscale != "linear":
        qa.set_wave_scale(ax, xscale)
    qa.set_flux_scale(ax, yscale)


def overlap_legend(ax: Any, ratio_window_frac: float | None, *, segments: str = "segments as extracted") -> None:
    """The legend entries all overlap panels share (handles without data, on ``ax``), and the note
    that says how wide the ratio window is. ``segments`` words what the two lines of a panel are."""
    ax.plot([], [], color=qa.MAIN_COLOR, label=f"{segments} (color: segment)")
    ax.fill_between([], [], [], color=qa.MAIN_COLOR, alpha=0.22, lw=0, label=qa.SIGMA_BAND_LABEL)
    ax.plot([], [], color=qa.MAIN_COLOR, ls="--", label=r"bluer segment $\times$ measured factor")
    ax.plot([], [], color=qa.REF_COLOR, lw=qa.LW["reference"], ls=":", label="crossover")
    ax.fill_between([], [], [], color=qa.ROLE_COLORS["window"], alpha=0.18, lw=0, label="ratio window")
    qa.annotate(ax, rf"window: {float(ratio_window_frac):g} $\times$ crossover wavelength, clipped to the overlap"
                if ratio_window_frac is not None else "")


class PlotStitchBackground(Step):
    """Source / background / difference comparison of every stitched spectrum.

    For each ``*_s1dcomb.ecsv`` the background that was removed from the
    combination is reassembled segment by segment: every segment file named
    in the ECSV metadata is looked up (like ``plot_stitch``), its
    BACKGROUND/BKGD_ERROR columns -- the background the extraction measured
    and subtracted (``mrs_extract``'s on/off annulus; ``extract_extended``'s
    in-field background), carried through 1-D processing steps -- are
    interpolated onto the stitched samples owned by that segment and scaled
    by the segment's applied stitch scale, so all three curves share the
    stitch's flux scale. One figure per stitched spectrum
    (``*_bkgcomp.png``): the source *without* background subtraction
    (stitched flux + background), the background estimate (with its error
    band), and the background-subtracted combination itself (black); the
    identity ``source = background + subtracted`` holds sample by sample.
    Crossovers are marked and the selected spectral ``features`` (default:
    every bundled dataset in range) are annotated as on every spectrum-like
    figure.

    Segments whose files are gone or carry no background record (SKYSUB not
    set -- extraction without background estimation, or products from before
    the columns travelled through defringing/cleaning) leave a gap in the
    background curves and are named in the legend annotation; a spectrum
    with no background record at all still gets its figure, showing only the
    combination. The wavelength axis is logarithmic unless ``xscale:
    linear``, the flux axis logarithmic too unless ``yscale: linear`` (backgrounds and
    mid-infrared fluxes span decades; negative excursions of a noisy
    background disappear on a log axis).
    """

    level = "qa"
    batch = "all"
    inputs = ("*_s1dcomb.ecsv",)
    version = "4"   # 3 -> 4: "? [wavelength]" for unidentified peaks, lines identified within one sample / 50 km/s

    def run(self, inputs: list[Path], ctx: RunContext, *, segments_stage: str | None = None,
            xscale: str = "log", yscale: str = "log", features: Any = "all",
            label_offset: float = qa.FEATURE_LABEL_OFFSET, photospheric: bool = True,
            dpi: int = 150, **_: Any) -> Iterable[Path]:
        out: list[Path] = []
        for inp in sorted(inputs):
            s = _read_stitch(inp)
            wave, flux = s["wave"], s["flux"]
            background = np.full(len(wave), np.nan)
            bkg_error = np.full(len(wave), np.nan)
            missing: list[str] = []
            for name, label, scale in zip(s["names"], s["labels"], s["scales"]):
                rows = s["segment_of"] == label
                if not rows.any():
                    continue
                segment = qa.find_file(name, ctx, segments_stage, inp.parent)
                if segment is None:
                    log.warning("%s: segment %s not found in any stage directory; "
                                "background left out", inp.name, name)
                    missing.append(label)
                    continue
                record = qa.read_background(segment)
                if record is None:
                    log.warning("%s: %s carries no background record (SKYSUB not set); "
                                "was its extraction run without background estimation, or its "
                                "1-D processing rerun since the columns travel through?",
                                inp.name, segment.name)
                    missing.append(label)
                    continue
                w, b, be = record
                background[rows] = np.interp(wave[rows], w, b, left=np.nan, right=np.nan) * scale
                bkg_error[rows] = np.interp(wave[rows], w, be, left=np.nan, right=np.nan) * scale

            plot = qa.Spectrum(xscale=xscale, yscale=yscale, figsize=(11.0, 4.8))
            if np.isfinite(background).any():
                plot.line(wave, flux + background, color=qa.ROLE_COLORS["context"], context=True,
                          label="source (background not subtracted)")
                plot.line(wave, background, error=bkg_error, color=qa.ROLE_COLORS["background"], context=True,
                          label="background estimate")
            else:
                log.warning("%s: no segment carries a background record; drawing the "
                            "combination only", inp.name)
            plot.line(wave, flux, label="source \N{MINUS SIGN} background (stitched)")
            plot.crossovers(s["crossovers"])
            out.append(plot.finish(ctx.output_dir / f"{inp.stem}_bkgcomp.png", ylabel=s["ylabel"], dpi=dpi,
                                   notes=[inp.name, f"no background for: {', '.join(missing)}" if missing else ""],
                                   features=features, label_offset=label_offset, photospheric=photospheric))
        return out
