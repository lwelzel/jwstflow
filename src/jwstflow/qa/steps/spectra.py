"""Spectrum-like figures of 1-D products: the spectrum, its signal-to-noise ratio, and
what the 1-D processing did to it."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from ... import qa
from ...steps import RunContext, Step

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
    standard (``docs/qa_figures.md``); both axes are logarithmic unless
    ``xscale: linear`` / ``yscale: linear`` (a log flux axis has no negative
    range to waste). With two or more spectra a combined overview figure
    (``<stage>_all.png``) draws every spectrum on one log-log axis, coloured
    from the standard colormap. With ``features`` the selected spectral
    features (jwstflow ``features:`` grammar; ``all`` takes every bundled
    dataset in range) are annotated on every figure
    (``qa.annotate_features``: every gas line labelled at a constant factor
    ``label_offset`` over the spectrum at its own wavelength, colored by
    species; the absorption lines of the stellar photosphere dashed, labelled
    below, unless ``photospheric: false``; ices, silicates and PAH as
    rectangles in rows at the foot of the plot). The 1-sigma uncertainty (the
    ``<column>_ERROR`` column, where the table carries one) is shaded around
    each spectrum of the per-file figures (``errors: false`` leaves it out).
    """

    level = "qa"
    batch = "all"   # all inputs in one task, so the combined overview sees every spectrum
    version = "7"   # 6 -> 7: built on qa.Spectrum

    def run(
        self,
        inputs: list[Path],
        ctx: RunContext,
        *,
        ylim: tuple[float, float] | list[float] | None = None,
        column: str = "FLUX",
        xscale: str = "log",
        yscale: str = "log",
        features: Any = None,
        label_offset: float = qa.FEATURE_LABEL_OFFSET,
        photospheric: bool = True,
        errors: bool = True,
        dpi: int = 150,
        fmt: str = "png",
        **_: Any,
    ) -> Iterable[Path]:
        from astropy.io import fits

        out: list[Path] = []
        combined: list[tuple[np.ndarray, np.ndarray, str, str]] = []
        for inp in sorted(inputs):
            with fits.open(inp) as hdul:
                tables = [h for h in hdul if h.name in qa.SPECTRUM_EXTENSIONS]
                if not tables:
                    log.warning("%s has no EXTRACT1D/COMBINE1D extension; skipped", inp.name)
                    continue
                plot = qa.Spectrum(xscale=xscale, yscale=yscale)
                ylabel = ""
                for i, (h, color) in enumerate(zip(tables, qa.line_colors(len(tables)))):
                    tab = h.data
                    if "WAVELENGTH" not in tab.names or column not in tab.names:
                        continue
                    wave = np.asarray(tab["WAVELENGTH"], dtype=float)
                    values, ylabel = qa.to_mjy(tab[column], qa.column_unit(h, column))
                    label = qa.spectrum_name(h, i) if 1 < len(tables) <= 12 else None
                    error_column = f"{column}_ERROR"
                    sigma = qa.to_mjy(tab[error_column], qa.column_unit(h, error_column))[0] \
                        if errors and error_column in tab.names else None
                    plot.line(wave, values, error=sigma, color=color, label=label,
                              error_label=qa.SIGMA_BAND_LABEL if i == 0 else None)
                    combined.append((wave, values, inp.stem + (f":{label}" if label else ""), ylabel))
            out.append(plot.finish(ctx.output_dir / f"{inp.stem}.{fmt}", ylabel=ylabel or column, notes=[inp.name],
                                   features=features, label_offset=label_offset, photospheric=photospheric,
                                   ylim=ylim, dpi=dpi))
        overview = all_spectra_figure(combined, ctx.output_dir / f"{ctx.stage}_all.{fmt}", what="spectra", dpi=dpi,
                                      features=features, label_offset=label_offset, photospheric=photospheric)
        if overview is not None:
            out.append(overview)
        return out


class PlotSnr(Step):
    """Signal-to-noise ratio of every 1-D spectrum against wavelength (``FLUX / FLUX_ERROR``).

    One spectrum-like figure per input file (``*_x1d`` / ``*_c1d`` /
    ``*_s1d(comb)``): for every EXTRACT1D/COMBINE1D table the flux column
    (``column``, default FLUX) divided by its ``<column>_ERROR``, per
    wavelength sample, on logarithmic axes -- where the spectrum is
    trustworthy and where it is not, which the flux figure of
    ``plot_spectrum`` does not show. The spectral ``features`` (default: every
    bundled dataset in range; ``features: null`` for none) are annotated as on
    every spectrum-like figure, so a dip or a spike in the signal-to-noise
    ratio can be read against the line or band that causes it. Samples
    without a positive finite error are left out; each spectrum's median S/N
    is stated in the legend. Tables without an error column (or with an
    all-zero one) are skipped with a warning.
    """

    level = "qa"
    version = "2"   # 1 -> 2: a spectrum-like figure, features annotated

    def run(self, inputs: list[Path], ctx: RunContext, *, column: str = "FLUX", xscale: str = "log",
            yscale: str = "log", features: Any = "all", label_offset: float = qa.FEATURE_LABEL_OFFSET,
            photospheric: bool = True, dpi: int = 150, **_: Any) -> Iterable[Path]:
        from astropy.io import fits

        out: list[Path] = []
        for inp in sorted(inputs):
            with fits.open(inp) as hdul:
                series = []
                for i, h in enumerate(h for h in hdul if h.name in qa.SPECTRUM_EXTENSIONS):
                    tab = h.data
                    error_column = f"{column}_ERROR"
                    if not {"WAVELENGTH", column, error_column} <= set(tab.names):
                        continue
                    flux, error = np.asarray(tab[column], float), np.asarray(tab[error_column], float)
                    with np.errstate(all="ignore"):
                        snr = np.where(np.isfinite(error) & (error > 0), flux / error, np.nan)
                    if np.isfinite(snr).any():
                        series.append((np.asarray(tab["WAVELENGTH"], float), snr, qa.spectrum_name(h, i)))
            if not series:
                log.warning("%s: no spectrum with a usable %s_ERROR column; skipped", inp.name, column)
                continue
            plot = qa.Spectrum(xscale=xscale, yscale=yscale)
            for (wave, snr, name), color in zip(series, qa.line_colors(len(series))):
                median = f"median S/N {float(np.nanmedian(snr)):.3g}"
                plot.line(wave, snr, color=color,
                          label=(f"{name}: {median}" if len(series) > 1 else median) if len(series) <= 12 else None)
            out.append(plot.finish(ctx.output_dir / f"{inp.stem}_snr.png", ylabel=qa.SNR_LABEL, notes=[inp.name],
                                   features=features, label_offset=label_offset, photospheric=photospheric, dpi=dpi))
        return out


class PlotSpectrumStages(Step):
    """What 1-D processing did to a spectrum: the same spectrum at several stages, and their ratios.

    The inputs are the spectra of the *last* stage of a 1-D processing chain
    (e.g. the cleaned spectra); ``stages`` names the earlier stages to
    compare them with, in processing order (e.g. ``[mrs_extract,
    mrs_defringe_1d]``). For every input its ancestor in each of those
    stages is located by name -- the file whose name without the product
    suffix is the longest leading part of the input's
    (``..._circle1_s1d.fits`` -> ``..._circle1_defringed_s1d.fits`` ->
    ``..._circle1_defringed_clean_s1d.fits``). One figure per input
    (``*_stages.png``): on top every version of the spectrum in mJy, the
    earlier ones coloured in processing order under the final one in black;
    below, the ratio of each later version to the one before it
    (interpolated onto the later version's wavelengths), i.e. exactly what
    each processing step multiplied into the spectrum -- the fringe
    correction, the cleaned samples. ``ratio_range`` sets the half-width of
    the ratio axis around unity; ``features`` annotates the spectral features
    on the spectrum panel (off by default: the figure is about the ratios).
    Inputs without any ancestor are skipped with a warning.
    """

    level = "qa"
    version = "2"   # 1 -> 2: built on qa.Spectrum
    inputs = ("*_s1d.fits", "*_x1d.fits")

    def run(self, inputs: list[Path], ctx: RunContext, *, stages: list[str] | tuple[str, ...] = (),
            xscale: str = "log", yscale: str = "linear", ratio_range: float = 0.1, features: Any = None,
            dpi: int = 150, **_: Any) -> Iterable[Path]:
        out: list[Path] = []
        for inp in sorted(inputs):
            versions: list[tuple[str, Path]] = []
            for stage in stages:
                try:
                    directory = ctx.dir_of(stage)
                except KeyError:
                    log.warning("stage %r is not part of this workflow; left out of the comparison", stage)
                    continue
                earlier = qa.ancestor(inp, Path(directory))
                if earlier is not None:
                    versions.append((stage, earlier))
            if not versions:
                log.warning("%s: no earlier version found in %s; skipped", inp.name, list(stages))
                continue
            versions.append((qa.stage_of(inp), inp))
            spectra = []
            for stage, path in versions:
                try:
                    w, f, _, unit = qa.read_spectrum(path)
                except (ValueError, OSError) as exc:
                    log.warning("%s: %s; left out", path.name, exc)
                    continue
                f, ylabel = qa.to_mjy(f, unit)
                order = np.argsort(w)
                spectra.append((stage, w[order], f[order], ylabel))
            if len(spectra) < 2:
                continue
            plot = qa.Spectrum(xscale=xscale, yscale=yscale, panels=2, height_ratios=[2, 1])
            ratios = plot.axes[1]
            colors = qa.line_colors(len(spectra))[:-1] + [qa.MAIN_COLOR]
            for k, ((stage, w, f, _), color) in enumerate(zip(spectra, colors)):
                plot.line(w, f, color=color, label=stage, context=k < len(spectra) - 1)
            for (before, later), color in zip(pairwise(spectra), colors[1:]):
                good = np.isfinite(before[1]) & np.isfinite(before[2])
                if good.sum() < 2:
                    continue
                with np.errstate(all="ignore"):
                    ratio = later[2] / np.interp(later[1], before[1][good], before[2][good], left=np.nan, right=np.nan)
                plot.line(later[1], ratio, ax=ratios, color=color, lw=qa.LW["thin"], label=f"{later[0]} / {before[0]}")
            qa.refline(ratios, y=1.0)
            ratios.set(ylabel=r"$F_\nu$ / $F_\nu$ of the stage before", ylim=(1.0 - ratio_range, 1.0 + ratio_range))
            out.append(plot.finish(ctx.output_dir / f"{inp.stem}_stages.png", ylabel=spectra[-1][3], notes=[inp.name],
                                   features=features, dpi=dpi))
        return out


def all_spectra_figure(series: list[tuple[np.ndarray, np.ndarray, str, str]], path: Path, *,
                       what: str, dpi: int, features: Any = None,
                       label_offset: float = qa.FEATURE_LABEL_OFFSET, photospheric: bool = True) -> Path | None:
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
    labels = qa.short_labels([name for _, _, name, _ in series])
    plot = qa.Spectrum(figsize=(11.0, 4.5))
    ylabel = ""
    for (wave, values, _, ylab), label, color in zip(series, labels, qa.line_colors(len(series))):
        ylabel = ylab or ylabel
        plot.line(wave, values, color=color, lw=qa.LW["thin"], label=label if len(series) <= 14 else None)
    return plot.finish(path, ylabel=ylabel, notes=[f"all {what} of this stage ({len(series)})"], features=features,
                       label_offset=label_offset, photospheric=photospheric, dpi=dpi)
