"""jwstflow's stock QA steps.

These double as documentation of the :class:`jwstflow.Step` API:

* ``run(inputs, ctx, **params)`` receives the selected files and a
  :class:`RunContext` (output directory, other stage directories, ...),
* it returns the paths it wrote (jwstflow records them for checkpointing),
* ``batch = "all"`` makes a step receive every input at once.

Reference them in YAML by entry-point name (``plot_spectrum``) or dotted path
(``jwstflow.contrib.qa:PlotSpectrum``).

Every figure follows the jwstflow QA figure standard (``docs/qa_figures.md``)
through :mod:`jwstflow.qafig`: no titles (legends annotate), fluxes in mJy,
mid-point step plots, height-matched colorbars, nan-aware cube collapses, the
cmasher ``torch`` palette with black main lines, and units in square brackets.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .. import qafig
from ..data.discovery import read_metadata
from ..steps.base import RunContext, Step

log = logging.getLogger(__name__)


class PlotSpectrum(Step):
    """Figure of every 1-D spectrum (``*_x1d.fits`` / ``*_c1d.fits``), flux in mJy.

    For each input file, every EXTRACT1D/COMBINE1D table extension is read,
    its flux column (``column``, default FLUX) converted to mJy from whatever
    unit the table declares, and drawn as a mid-point step plot over
    wavelength -- all extensions of one file share a figure, labelled per
    slit/source when a file carries several spectra (e.g. NIRSpec MOS).
    Inputs without such an extension are skipped with a warning. One figure
    per input file, named after it, following the jwstflow QA figure
    standard (``docs/qa_figures.md``); the wavelength axis is logarithmic
    unless ``xscale: linear``, the flux axis linear unless ``yscale: log``
    (mid-infrared spectra span decades). With two or more spectra a combined
    overview figure (``<stage>_all.png``) draws every spectrum on one
    log-log axis, coloured from the standard colormap. With ``features`` the
    selected spectral features (jwstflow ``features:`` grammar; ``all`` takes
    every bundled dataset in range) are marked on every figure -- gas lines
    as labelled ticks, emission bands and ice bands as shaded wavelength
    lanes (``qafig.annotate_features``).
    """

    level = "qa"
    batch = "all"   # all inputs in one task, so the combined overview sees every spectrum
    version = "5"   # 4 -> 5: optional spectral-feature annotation (features parameter)

    def run(
        self,
        inputs: list[Path],
        ctx: RunContext,
        *,
        ylim: tuple[float, float] | list[float] | None = None,
        column: str = "FLUX",
        xscale: str = "log",
        yscale: str = "linear",
        features: Any = None,
        dpi: int = 150,
        fmt: str = "png",
        **_: Any,
    ) -> Iterable[Path]:
        from astropy.io import fits

        out: list[Path] = []
        combined: list[tuple[np.ndarray, np.ndarray, str, str]] = []
        for inp in sorted(inputs):
            with fits.open(inp) as hdul:
                tables = [h for h in hdul if h.name in ("EXTRACT1D", "COMBINE1D")]
                if not tables:
                    log.warning("%s has no EXTRACT1D/COMBINE1D extension; skipped", inp.name)
                    continue
                fig, ax = qafig.subplots(figsize=(9, 4))
                colors = qafig.line_colors(len(tables))
                ylabel = ""
                waves: list[np.ndarray] = []
                for i, (h, color) in enumerate(zip(tables, colors)):
                    tab = h.data
                    if "WAVELENGTH" not in tab.names or column not in tab.names:
                        continue
                    values, ylabel = qafig.to_mjy(tab[column], _column_unit(h, column))
                    label = None
                    if len(tables) > 1 and len(tables) <= 12:
                        label = str(h.header.get("SLTNAME") or h.header.get("SRCNAME") or f"ext {i + 1}")
                    qafig.step(ax, tab["WAVELENGTH"], values, color=color, label=label)
                    waves.append(np.asarray(tab["WAVELENGTH"], float))
                    combined.append((waves[-1], values,
                                     inp.stem + (f":{label}" if label else ""), ylabel))
                ax.set_xlabel(qafig.WAVE_LABEL)
                ax.set_ylabel(ylabel or f"{column}")
                qafig.set_wave_scale(ax, xscale)
                ax.set_yscale(yscale)
                if ylim:
                    ax.set_ylim(*ylim)
                _annotate_features_in_range(ax, features, waves)
                qafig.annotate(ax, inp.name)
                qafig.figlegend(fig)
                out.append(qafig.save(fig, ctx.output_dir / f"{inp.stem}.{fmt}", dpi=dpi))
        overview = _all_spectra_figure(combined, ctx.output_dir / f"{ctx.stage}_all.{fmt}",
                                       what="spectra", dpi=dpi, features=features)
        if overview is not None:
            out.append(overview)
        return out


class QuicklookImage(Step):
    """PNG of the SCI extension (2-D images; 3-D cubes are nan-median collapsed).

    For each input file, the SCI array is rendered as one image panel: cubes
    are first collapsed along the spectral axis with a nan-aware statistic
    (``collapse``: median/mean/sum/max; spaxels with less than
    ``min_coverage`` finite planes are blanked -- at the footprint edges a
    nanmedian over a handful of planes is noise that would otherwise stretch
    the display limits), then shown in detector/sky pixels with robust
    percentile limits (``percentiles``), an optional non-linear ``stretch``,
    and a height-matched colorbar labelled with the data's BUNIT (surface
    brightness is converted to mJy/arcsec^2). Inputs without a SCI extension
    are skipped with a warning. One PNG per input file, named after it.
    """

    level = "qa"
    version = "3"   # 2 -> 3: robust display limits (min_coverage; hot pixels cannot set vmax)

    def run(
        self,
        inputs: list[Path],
        ctx: RunContext,
        *,
        percentiles: tuple[float, float] | list[float] = (1.0, 99.0),
        collapse: str = "median",
        min_coverage: float = 0.5,
        stretch: str = "linear",
        cmap: str | None = None,
        dpi: int = 150,
        **_: Any,
    ) -> Iterable[Path]:
        from astropy.io import fits

        out: list[Path] = []
        for inp in inputs:
            with fits.open(inp) as hdul:
                if "SCI" not in hdul:
                    log.warning("%s has no SCI extension; skipped", inp.name)
                    continue
                data = np.asarray(hdul["SCI"].data, dtype=float)
                unit = hdul["SCI"].header.get("BUNIT")
            image = qafig.collapse(data, collapse, min_coverage=min_coverage)
            fig, ax = qafig.subplots(figsize=(6.5, 6))
            qafig.imshow(ax, image, unit=unit, stretch=stretch,
                         percentiles=tuple(percentiles), cmap=cmap or qafig.CMAP)
            ax.set(xlabel="x [pix]", ylabel="y [pix]")
            qafig.annotate(ax, inp.name)
            qafig.figlegend(fig)
            out.append(qafig.save(fig, ctx.output_dir / f"{inp.stem}.png", dpi=dpi))
        return out


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
    axis linear unless ``yscale: log``. With two or more stitched spectra a
    combined overview figure (``<stage>_all.png``) draws every one on a
    single log-log axis, coloured from the standard colormap. With
    ``features`` the selected spectral features (jwstflow ``features:``
    grammar; ``all`` takes every bundled dataset in range) are marked on
    every figure -- gas lines as labelled ticks, emission bands and ice
    bands as shaded wavelength lanes (``qafig.annotate_features``).
    """

    level = "qa"
    batch = "all"   # all inputs in one task, so the combined overview sees every spectrum
    inputs = ("*_s1dcomb.ecsv",)
    version = "5"   # 4 -> 5: optional spectral-feature annotation (features parameter)

    def run(self, inputs: list[Path], ctx: RunContext, *, segments_stage: str | None = None,
            unscaled: bool = True, xscale: str = "log", yscale: str = "linear",
            features: Any = None, dpi: int = 150, **_: Any) -> Iterable[Path]:
        from astropy.table import Table

        out: list[Path] = []
        combined: list[tuple[np.ndarray, np.ndarray, str, str]] = []
        for inp in sorted(inputs):
            tab = Table.read(inp)
            wave = np.asarray(tab["WAVELENGTH"], dtype=float)
            flux, ylabel = qafig.to_mjy(np.asarray(tab["FLUX"], dtype=float),
                                        str(tab["FLUX"].unit or "Jy"))
            combined.append((wave, flux, inp.stem, ylabel))
            meta = tab.meta
            names = [str(n) for n in meta.get("inputs", [])]
            labels = [str(s) for s in meta.get("segments", names)]
            scales = [float(s) for s in meta.get("scales_applied", [1.0] * len(names))]
            colors = qafig.line_colors(len(names)) if len(names) > 1 else qafig.line_colors(2)
            crossovers = [float(b) for b in meta.get("crossovers_um", [])]
            segments = []   # (label, scale, color, wavelength, flux, unit) of every segment still on disk
            for name, label, scale, color in zip(names, labels, scales, colors):
                segment = _find_file(name, ctx, segments_stage, inp.parent)
                if segment is None:
                    log.warning("%s: segment %s not found in any stage directory; left out", inp.name, name)
                    continue
                w, f, _, unit = _read_segment(segment)
                segments.append((label, scale, color, w, f, unit))

            # the combination over its rescaled segments
            fig, ax = qafig.subplots(figsize=(11, 4))
            for label, scale, color, w, f, unit in segments:
                f, _ = qafig.to_mjy(f * scale, unit)
                qafig.step(ax, w, f, color=color, alpha=0.6, lw=0.7,
                           label=label + (f" x {scale:.3f}" if scale != 1.0 else ""))
            qafig.step(ax, wave, flux, color=qafig.MAIN_COLOR, label="stitched")
            _mark_crossovers(ax, crossovers)
            ax.set(xlabel=qafig.WAVE_LABEL, ylabel=ylabel)
            qafig.set_wave_scale(ax, xscale)
            ax.set_yscale(yscale)
            _annotate_features_in_range(ax, features, [wave])
            qafig.annotate(ax, inp.name)
            qafig.figlegend(fig)
            out.append(qafig.save(fig, ctx.output_dir / f"{inp.stem}.png", dpi=dpi))

            # the segments as extracted, before the multiplicative rescaling -- only
            # when a scale was actually applied (otherwise it repeats the figure above)
            if unscaled and any(scale != 1.0 for _, scale, *_ in segments):
                fig, ax = qafig.subplots(figsize=(11, 4))
                for label, _, color, w, f, unit in segments:
                    f, ylabel = qafig.to_mjy(f, unit)
                    qafig.step(ax, w, f, color=color, alpha=0.8, lw=0.9, label=label)
                _mark_crossovers(ax, crossovers)
                ax.set(xlabel=qafig.WAVE_LABEL, ylabel=ylabel)
                qafig.set_wave_scale(ax, xscale)
                ax.set_yscale(yscale)
                _annotate_features_in_range(ax, features, [wave])
                qafig.annotate(ax, f"{inp.name}: segments as extracted (before rescaling)")
                qafig.figlegend(fig)
                out.append(qafig.save(fig, ctx.output_dir / f"{inp.stem}_unscaled.png", dpi=dpi))
        overview = _all_spectra_figure(combined, ctx.output_dir / f"{ctx.stage}_all.png",
                                       what="stitched spectra", dpi=dpi, features=features)
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
    quality of the factor is visible exactly where it was measured, and each
    panel's legend entry states the factor with its 1-sigma uncertainty.
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
    version = "1"

    #: Shading of the ratio-measurement window (muted orange, alpha applied at draw time).
    WINDOW_COLOR = "#d95f02"

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
            colors = qafig.line_colors(len(names)) if len(names) > 1 else qafig.line_colors(2)
            segments: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
            for name in names:
                path = _find_file(name, ctx, segments_stage, inp.parent)
                if path is None:
                    log.warning("%s: segment %s not found in any stage directory; its panels "
                                "are annotated instead of drawn", inp.name, name)
                    segments.append(None)
                    continue
                w, f, e, unit = _read_segment(path)
                f, _ = qafig.to_mjy(f, unit)
                e, _ = qafig.to_mjy(e, unit)
                segments.append((w, f, e))

            n_pairs = len(names) - 1
            ncols = min(3, n_pairs)
            nrows = (n_pairs + ncols - 1) // ncols
            fig, axes = qafig.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)
            for i in range(n_pairs):
                ax = axes[i // ncols][i % ncols]
                self._overlap_panel(ax, i, labels, colors, segments, crossovers[i],
                                    None if windows[i] is None else (float(windows[i][0]), float(windows[i][1])),
                                    None if ratios[i] is None else float(ratios[i]),
                                    None if ratio_errors[i] is None else float(ratio_errors[i]),
                                    xscale=xscale, yscale=yscale)
                if i % ncols == 0:
                    ax.set_ylabel(qafig.FLUX_LABEL)
                if i // ncols == nrows - 1 or i + ncols >= n_pairs:
                    ax.set_xlabel(qafig.WAVE_LABEL)
            for i in range(n_pairs, nrows * ncols):
                axes[i // ncols][i % ncols].set_axis_off()
            first = axes[0][0]
            first.plot([], [], color="0.6", lw=0.6, ls=":", label="crossover")
            first.fill_between([], [], [], color=self.WINDOW_COLOR, alpha=0.18, label="ratio window")
            frac = meta.get("ratio_window_frac")
            qafig.annotate(first, inp.name,
                           f"window: {float(frac):g} x crossover wavelength, clipped to the overlap"
                           if frac is not None else "")
            qafig.figlegend(fig)
            out.append(qafig.save(fig, ctx.output_dir / f"{inp.stem}_overlaps.png", dpi=dpi))
        return out

    def _overlap_panel(self, ax: Any, i: int, labels: list[str], colors: list[Any],
                       segments: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None],
                       crossover: float, window: tuple[float, float] | None,
                       ratio: float | None, ratio_error: float | None, *,
                       xscale: str, yscale: str) -> None:
        """One neighbour pair: both segments around their overlap, crossover and ratio window."""
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
            qafig.step(ax, w[show], f[show], color=color, lw=0.9, label=label)
            finite_err = show & np.isfinite(e)
            if finite_err.any():
                ax.fill_between(w[finite_err], (f - e)[finite_err], (f + e)[finite_err],
                                step="mid", color=color, alpha=0.20, lw=0)
        if ratio is not None and seg_a is not None and seg_b is not None:
            w, f, _ = seg_a
            inside = (w >= float(seg_b[0].min())) & (w <= float(seg_a[0].max()))
            if inside.any():
                qafig.step(ax, w[inside], f[inside] * ratio, color=colors[i], lw=0.9, ls="--",
                           label=f"{labels[i]} x {ratio:.3f}")
        ax.axvline(crossover, color="0.6", lw=0.6, ls=":")
        if window is not None:
            ax.axvspan(window[0], window[1], color=self.WINDOW_COLOR, alpha=0.18, lw=0)
        measured = (f"x {ratio:.3f} ± {ratio_error:.2g}" if ratio is not None and ratio_error is not None
                    else "ratio not measured" if ratio is None else f"x {ratio:.3f}")
        missing = [labels[i + k] for k, seg in ((0, seg_a), (1, seg_b)) if seg is None]
        qafig.annotate(ax, f"{labels[i]} | {labels[i + 1]}: {measured}",
                       f"segment file missing: {', '.join(missing)}" if missing else "")
        ax.set_xlim(*view)
        if xscale != "linear":
            qafig.set_wave_scale(ax, xscale)
        ax.set_yscale(yscale)


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
    every bundled dataset in range) are annotated -- gas lines as labelled
    ticks, emission bands and ice bands as shaded wavelength lanes.

    Segments whose files are gone or carry no background record (SKYSUB not
    set -- extraction without background estimation, or products from before
    the columns travelled through defringing/cleaning) leave a gap in the
    background curves and are named in the legend annotation; a spectrum
    with no background record at all still gets its figure, showing only the
    combination. The wavelength axis is logarithmic unless ``xscale:
    linear``, the flux axis linear unless ``yscale: log`` (backgrounds and
    mid-infrared fluxes span decades; negative excursions of a noisy
    background disappear on a log axis).
    """

    level = "qa"
    batch = "all"
    inputs = ("*_s1dcomb.ecsv",)
    version = "1"

    def run(self, inputs: list[Path], ctx: RunContext, *, segments_stage: str | None = None,
            xscale: str = "log", yscale: str = "linear", features: Any = "all",
            line_width_um: float = 0.02, dpi: int = 150, **_: Any) -> Iterable[Path]:
        from astropy.table import Table

        out: list[Path] = []
        for inp in sorted(inputs):
            tab = Table.read(inp)
            wave = np.asarray(tab["WAVELENGTH"], dtype=float)
            flux, ylabel = qafig.to_mjy(np.asarray(tab["FLUX"], dtype=float),
                                        str(tab["FLUX"].unit or "Jy"))
            segment_of = np.asarray(tab["SEGMENT"], dtype=str) if "SEGMENT" in tab.colnames else \
                np.full(len(wave), "", dtype=object)
            meta = tab.meta
            names = [str(n) for n in meta.get("inputs", [])]
            labels = [str(s) for s in meta.get("segments", names)]
            scales = [float(s) for s in meta.get("scales_applied", [1.0] * len(names))]
            crossovers = [float(b) for b in meta.get("crossovers_um", [])]
            background = np.full(len(wave), np.nan)
            bkg_error = np.full(len(wave), np.nan)
            missing: list[str] = []
            for name, label, scale in zip(names, labels, scales):
                rows = segment_of == label
                if not rows.any():
                    continue
                segment = _find_file(name, ctx, segments_stage, inp.parent)
                if segment is None:
                    log.warning("%s: segment %s not found in any stage directory; "
                                "background left out", inp.name, name)
                    missing.append(label)
                    continue
                record = _read_background(segment)
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

            fig, ax = qafig.subplots(figsize=(11, 4))
            src_color, bkg_color = qafig.line_colors(2)
            if np.isfinite(background).any():
                qafig.step(ax, wave, flux + background, color=src_color, lw=0.7,
                           label="source (background not subtracted)")
                qafig.step(ax, wave, background, color=bkg_color, lw=0.7, label="background estimate")
                finite_err = np.isfinite(bkg_error)
                if finite_err.any():
                    ax.fill_between(wave, background - bkg_error, background + bkg_error,
                                    step="mid", color=bkg_color, alpha=0.25, lw=0)
            else:
                log.warning("%s: no segment carries a background record; drawing the "
                            "combination only", inp.name)
            qafig.step(ax, wave, flux, color=qafig.MAIN_COLOR, label="source - background (stitched)")
            _mark_crossovers(ax, crossovers)
            ax.set(xlabel=qafig.WAVE_LABEL, ylabel=ylabel)
            qafig.set_wave_scale(ax, xscale)
            ax.set_yscale(yscale)
            qafig.annotate_features(ax, features, wave_min=float(np.nanmin(wave)),
                                    wave_max=float(np.nanmax(wave)), line_width_um=line_width_um)
            qafig.annotate(ax, inp.name,
                           f"no background for: {', '.join(missing)}" if missing else "")
            qafig.figlegend(fig)
            out.append(qafig.save(fig, ctx.output_dir / f"{inp.stem}_bkgcomp.png", dpi=dpi))
        return out


def _read_background(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Finite ``(wavelength, background, bkgd_error)`` [um, mJy, mJy] of a segment's
    background record, or None when the file carries none (no SKYSUB header, no
    BACKGROUND column, or no finite samples)."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        if not hdul[0].header.get("SKYSUB"):
            return None
        exts = [h for h in hdul if h.name in ("EXTRACT1D", "COMBINE1D")]
        if not exts or "BACKGROUND" not in exts[0].data.names:
            return None
        tab = exts[0].data
        wave = np.asarray(tab["WAVELENGTH"], dtype=float)
        bkg, _ = qafig.to_mjy(np.asarray(tab["BACKGROUND"], dtype=float),
                              _column_unit(exts[0], "BACKGROUND"))
        err_raw = np.asarray(tab["BKGD_ERROR"], dtype=float) if "BKGD_ERROR" in tab.names \
            else np.zeros(len(wave))
        err, _ = qafig.to_mjy(err_raw, _column_unit(exts[0], "BKGD_ERROR"))
    ok = np.isfinite(wave) & np.isfinite(bkg)
    if not ok.any():
        return None
    order = np.argsort(wave[ok])
    return wave[ok][order], bkg[ok][order], np.where(np.isfinite(err[ok]), err[ok], 0.0)[order]


def _mark_crossovers(ax: Any, crossovers: list[float]) -> None:
    """Dotted vertical line (plus one legend handle) per crossover wavelength."""
    for b in crossovers:
        ax.axvline(b, color="0.6", lw=0.6, ls=":")
    if crossovers:
        ax.plot([], [], color="0.6", lw=0.6, ls=":", label="crossover")


def _annotate_features_in_range(ax: Any, features: Any, waves: list[np.ndarray]) -> None:
    """``qafig.annotate_features`` over the common wavelength range of ``waves`` (no-op without either)."""
    finite = [w[np.isfinite(w)] for w in waves if np.isfinite(w).any()]
    if features in (None, False) or not finite:
        return
    qafig.annotate_features(ax, features, wave_min=float(min(w.min() for w in finite)),
                            wave_max=float(max(w.max() for w in finite)))


def _all_spectra_figure(series: list[tuple[np.ndarray, np.ndarray, str, str]], path: Path, *,
                        what: str, dpi: int, features: Any = None) -> Path | None:
    """The combined overview: every spectrum of the stage on one log-log axis.

    ``series`` is ``(wavelength, values_mjy, name, ylabel)`` per spectrum;
    fewer than two spectra draw nothing (the per-file figure already shows
    everything). Colors are sampled from the standard colormap in order of
    increasing starting wavelength; labels are the file names with their
    common prefix/suffix stripped.
    """
    if len(series) < 2:
        return None
    series = sorted(series, key=lambda s: float(np.nanmin(s[0])) if len(s[0]) else np.inf)
    labels = _short_labels([name for _, _, name, _ in series])
    fig, ax = qafig.subplots(figsize=(11, 4.5))
    colors = qafig.line_colors(len(series))
    ylabel = ""
    for (wave, values, _, ylab), label, color in zip(series, labels, colors):
        ylabel = ylab or ylabel
        qafig.step(ax, wave, values, color=color, lw=0.7,
                   label=label if len(series) <= 14 else None)
    ax.set(xlabel=qafig.WAVE_LABEL, ylabel=ylabel)
    qafig.set_wave_scale(ax, "log")
    ax.set_yscale("log")
    _annotate_features_in_range(ax, features, [w for w, *_ in series])
    qafig.annotate(ax, f"all {what} of this stage ({len(series)})")
    qafig.figlegend(fig)
    return qafig.save(fig, path, dpi=dpi)


def _short_labels(names: list[str]) -> list[str]:
    """Distinct cores of a family of product names: the common prefix and suffix stripped."""
    import os

    if len(names) < 2:
        return list(names)
    prefix = os.path.commonprefix(names)
    suffix = os.path.commonprefix([n[::-1] for n in names])[::-1]
    out = []
    for n in names:
        core = n[len(prefix):len(n) - len(suffix) if suffix else None]
        out.append(core.strip("_-.") or n)
    return out


def _find_file(name: str, ctx: RunContext, stage: str | None, *extra: Path) -> Path | None:
    """Locate ``name`` in a stage's directory, in any stage directory, in ``extra``,
    or -- last -- in the sibling runs of the target (``<target>/<run>/<level>/<stage>/``)."""
    dirs: list[Path] = []
    if stage:
        try:
            dirs.append(ctx.dir_of(stage))
        except KeyError:
            log.warning("stage %r is not part of this workflow; searching all stages", stage)
    dirs += [d for _, d in sorted(ctx.stage_dirs.items())] + list(extra)
    for d in dirs:
        candidate = Path(d) / name
        if candidate.exists():
            return candidate
    if ctx.target_dir is not None and Path(ctx.target_dir).is_dir():
        hits = sorted(Path(ctx.target_dir).glob(f"*/*/*/{name}"))
        if hits:
            return hits[0]
    return None


def _column_unit(hdu: Any, column: str) -> str | None:
    """A table column's TUNIT, falling back to the x1d contract's unit."""
    from ..spectra import X1D_UNITS

    return hdu.columns[column].unit or X1D_UNITS.get(column.upper())


def _read_segment(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    """(wavelength, flux, flux error, flux unit) of a segment file's EXTRACT1D/COMBINE1D
    table (the error all-NaN when the table carries no FLUX_ERROR column)."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        for name in ("EXTRACT1D", "COMBINE1D"):
            if name in hdul:
                tab = hdul[name].data
                wave = np.asarray(tab["WAVELENGTH"], dtype=float)
                error = np.asarray(tab["FLUX_ERROR"], dtype=float) if "FLUX_ERROR" in tab.names \
                    else np.full(len(wave), np.nan)
                return (wave, np.asarray(tab["FLUX"], dtype=float), error,
                        _column_unit(hdul[name], "FLUX"))
    raise ValueError(f"{path}: no EXTRACT1D/COMBINE1D extension")


class HeaderSummary(Step):
    """One JSON/CSV table with the key header values of all inputs (batch step).

    Runs once over the whole input set (``batch=all``): for every file the
    primary-header keywords in ``keys`` (default: exposure type, detector,
    optical elements, target and exposure time -- EXP_TYPE, DETECTOR,
    GRATING, FILTER, CHANNEL, BAND, TARGPROP, BKGDTARG, IS_IMPRT, EFFEXPTM)
    are collected into one row, and the resulting table is written twice:
    ``<name>.json`` and ``<name>.csv``. Handy as a run overview -- which
    exposure is which grating/band, which are backgrounds or imprints --
    without opening any FITS file.
    """

    level = "qa"

    batch: ClassVar[str] = "all"

    def run(
        self,
        inputs: list[Path],
        ctx: RunContext,
        *,
        name: str = "summary",
        keys: list[str] | None = None,
        **_: Any,
    ) -> Iterable[Path]:
        keys = keys or ["EXP_TYPE", "DETECTOR", "GRATING", "FILTER", "CHANNEL", "BAND", "TARGPROP", "BKGDTARG", "IS_IMPRT", "EFFEXPTM"]
        rows = []
        for inp in inputs:
            meta = read_metadata(inp)
            rows.append({"file": inp.name, **{k: meta.get(k) for k in keys}})
        js = ctx.output_dir / f"{name}.json"
        js.write_text(json.dumps(rows, indent=2, default=str))
        csv = ctx.output_dir / f"{name}.csv"
        with csv.open("w") as fh:
            fh.write(",".join(["file", *keys]) + "\n")
            for r in rows:
                fh.write(",".join(str(r.get(k, "")) for k in ["file", *keys]) + "\n")
        return [js, csv]


def copy_inputs(inputs: list[Path], ctx: RunContext, *, suffix: str = "_copy") -> list[Path]:
    """Trivial function-style step (mostly for tests and as a template)."""
    import shutil

    out = []
    for inp in inputs:
        dst = ctx.output_dir / f"{inp.stem}{suffix}{inp.suffix}"
        shutil.copy2(inp, dst)
        out.append(dst)
    return out
