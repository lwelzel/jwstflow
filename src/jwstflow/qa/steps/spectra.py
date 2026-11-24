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

    For each input file the EXTRACT1D/COMBINE1D table of the spectrum is read,
    its flux column (``column``, default FLUX) converted to mJy from whatever
    unit the table declares, and drawn as a mid-point step plot over
    wavelength. A file of named apertures of one extraction (``APERNAME``,
    e.g. ``extract_extended``: the disk, tighter contours, outflows) is drawn
    as its target aperture only -- ``aperture`` (default ``disk``; the first
    table when no aperture has that name); the apertures side by side are
    ``plot_spectrum_apertures``. A file of unnamed spectra (several slits or
    sources, e.g. NIRSpec MOS) draws them all in one figure, labelled per
    slit/source.
    Inputs without such an extension are skipped with a warning. One figure
    per input file, named after it, following the jwstflow QA figure
    standard (``docs/qa_figures.md``); both axes are logarithmic unless
    ``xscale: linear`` / ``yscale: linear`` (a log flux axis has no negative
    range to waste). With two or more spectra a combined overview figure
    (``<stage>_all.png``) draws every spectrum on one log-log axis, coloured
    from the standard colormap. With ``features`` the selected spectral
    features (jwstflow ``features:`` grammar; ``all``, the default, takes
    every bundled dataset in range; ``features: null`` for none) are annotated
    on every figure
    (``qa.annotate_features``: every gas line labelled at a constant factor
    ``label_offset`` over the spectrum at its own wavelength, colored by
    species; the absorption lines of the stellar photosphere dashed, labelled
    below, unless ``photospheric: false``; ices, silicates and PAH as
    rectangles in rows at the foot of the plot; strong peaks no catalogue
    line explains as ``? [wavelength]``), and every input gets a second
    figure (``*_detected.png``) that labels only the lines the spectrum
    shows -- and the gas bands only where it shows their forest
    (``detected: false`` leaves it out). The 1-sigma uncertainty (the
    ``<column>_ERROR`` column, where the table carries one) is shaded around
    each spectrum of the per-file figures (``errors: false`` leaves it out).
    """

    level = "qa"
    batch = "all"   # all inputs in one task, so the combined overview sees every spectrum
    version = "9"   # 8 -> 9: the target aperture of a file of apertures; features annotated by default; 7 -> 8: detected lines
    figure_suffix = ""   # appended to each per-file figure's name
    overview = True      # the combined <stage>_all figure

    def run(
        self,
        inputs: list[Path],
        ctx: RunContext,
        *,
        ylim: tuple[float, float] | list[float] | None = None,
        column: str = "FLUX",
        xscale: str = "log",
        yscale: str = "log",
        features: Any = "all",
        label_offset: float = qa.FEATURE_LABEL_OFFSET,
        photospheric: bool = True,
        errors: bool = True,
        detected: bool = True,
        aperture: str | None = "disk",
        dpi: int = 150,
        fmt: str = "png",
        **_: Any,
    ) -> Iterable[Path]:
        return self._figures(inputs, ctx, ylim=ylim, column=column, xscale=xscale, yscale=yscale, features=features,
                             label_offset=label_offset, photospheric=photospheric, errors=errors, detected=detected,
                             aperture=aperture, select=True, dpi=dpi, fmt=fmt)

    def _figures(self, inputs: list[Path], ctx: RunContext, *, ylim: Any, column: str, xscale: str, yscale: str,
                 features: Any, label_offset: float, photospheric: bool, errors: bool, detected: bool,
                 aperture: str | None, select: bool, dpi: int, fmt: str, max_labels: int = 12) -> list[Path]:
        """The per-file figures (and the overview): the target ``aperture`` of each file when ``select``,
        else every table of it, the target aperture among them in black."""
        from astropy.io import fits

        out: list[Path] = []
        combined: list[tuple[np.ndarray, np.ndarray, str, str]] = []
        # every line in range labelled; with features also only the lines the spectrum shows
        variants = (("", False), ("_detected", True))[:2 if detected and features not in (None, False) else 1]
        for inp in sorted(inputs):
            series = []     # (wavelength, mJy, 1-sigma, label, color) of every spectrum in the file
            ylabel = ""
            with fits.open(inp) as hdul:
                tables = [h for h in hdul if h.name in qa.SPECTRUM_EXTENSIONS]
                tables = qa.select_spectra(tables, aperture) if select else tables
                if not tables:
                    log.warning("%s has no EXTRACT1D/COMBINE1D extension; skipped", inp.name)
                    continue
                for i, (h, color) in enumerate(zip(tables, self._colors(tables, aperture))):
                    tab = h.data
                    if "WAVELENGTH" not in tab.names or column not in tab.names:
                        continue
                    wave = np.asarray(tab["WAVELENGTH"], dtype=float)
                    values, ylabel = qa.to_mjy(tab[column], qa.column_unit(h, column))
                    label = qa.spectrum_name(h, i) if 1 < len(tables) <= max_labels else None
                    error_column = f"{column}_ERROR"
                    sigma = qa.to_mjy(tab[error_column], qa.column_unit(h, error_column))[0] \
                        if errors and error_column in tab.names else None
                    series.append((wave, values, sigma, label, color))
                    combined.append((wave, values, inp.stem + (f":{label}" if label else ""), ylabel))
            for suffix, detected_only in variants:
                plot = qa.Spectrum(xscale=xscale, yscale=yscale)
                for i, (wave, values, sigma, label, color) in enumerate(series):
                    on_top = {"zorder": 3} if len(series) > 1 and color == qa.MAIN_COLOR else {}   # the target aperture
                    plot.line(wave, values, error=sigma, color=color, label=label,
                              error_label=qa.SIGMA_BAND_LABEL if i == 0 else None, **on_top)
                out.append(plot.finish(ctx.output_dir / f"{inp.stem}{self.figure_suffix}{suffix}.{fmt}",
                                       ylabel=ylabel or column,
                                       notes=[inp.name], features=features, label_offset=label_offset,
                                       photospheric=photospheric, ylim=ylim, dpi=dpi, detected_only=detected_only))
        overview = all_spectra_figure(combined, ctx.output_dir / f"{ctx.stage}_all.{fmt}", what="spectra", dpi=dpi,
                                      features=features, label_offset=label_offset,
                                      photospheric=photospheric) if self.overview else None
        if overview is not None:
            out.append(overview)
        return out

    @staticmethod
    def _colors(tables: list[Any], aperture: str | None) -> list[Any]:
        """The standard colormap's colors -- the target aperture among several in black (the main color)."""
        colors = list(qa.line_colors(len(tables)))
        if aperture is not None and len(tables) > 1:
            for k, h in enumerate(tables):
                if str(h.header.get("APERNAME", "")).strip() == aperture:
                    colors[k] = qa.MAIN_COLOR
        return colors


class PlotSpectrumApertures(PlotSpectrum):
    """The spectra of all apertures of a 1-D product side by side, flux in mJy.

    One figure per input file (``*_apertures.png``): every EXTRACT1D/COMBINE1D
    table of the file -- for an extraction with several apertures (the disk,
    tighter contours, outflows, objects, the nebulae of a disk) each one,
    labelled by its ``APERNAME`` (slit or source for unnamed tables), the
    target ``aperture`` (default ``disk``) in black, the others in the colors
    of the standard colormap. A comparison of the apertures, not the
    annotated spectrum of ``plot_spectrum``: no features (``features: all``
    annotates them), no 1-sigma bands (``errors: true`` draws them), no
    detected-lines figure and no combined overview. Logarithmic axes unless
    ``xscale`` / ``yscale: linear``.
    """

    level = "qa"
    batch = "all"
    version = "1"
    figure_suffix = "_apertures"
    overview = False

    def run(self, inputs: list[Path], ctx: RunContext, *, ylim: tuple[float, float] | list[float] | None = None,
            column: str = "FLUX", xscale: str = "log", yscale: str = "log", features: Any = None,
            label_offset: float = qa.FEATURE_LABEL_OFFSET, photospheric: bool = True, errors: bool = False,
            aperture: str = "disk", dpi: int = 150, fmt: str = "png", **_: Any) -> Iterable[Path]:
        return self._figures(inputs, ctx, ylim=ylim, column=column, xscale=xscale, yscale=yscale, features=features,
                             label_offset=label_offset, photospheric=photospheric, errors=errors, detected=False,
                             aperture=aperture, select=False, dpi=dpi, fmt=fmt, max_labels=16)


class PlotSnr(Step):
    """Signal-to-noise ratio of every 1-D spectrum against wavelength (``FLUX / FLUX_ERROR``).

    One spectrum-like figure per input file (``*_x1d`` / ``*_c1d`` /
    ``*_s1d(comb)``): for every EXTRACT1D/COMBINE1D table the flux column
    (``column``, default FLUX) divided by its ``<column>_ERROR``, per
    wavelength sample, on logarithmic axes -- where the spectrum is
    trustworthy and where it is not, which the flux figure of
    ``plot_spectrum`` does not show. Like ``plot_spectrum`` it draws the target
    ``aperture`` (default ``disk``) of a file of named apertures. The spectral ``features`` (default: every
    bundled dataset in range; ``features: null`` for none) are annotated as on
    every spectrum-like figure, so a dip or a spike in the signal-to-noise
    ratio can be read against the line or band that causes it. Samples
    without a positive finite error are left out; each spectrum's median S/N
    is stated in the legend. Tables without an error column (or with an
    all-zero one) are skipped with a warning.
    """

    level = "qa"
    version = "4"   # 3 -> 4: the target aperture of a file of apertures; 2 -> 3: "? [wavelength]" for unidentified peaks

    def run(self, inputs: list[Path], ctx: RunContext, *, column: str = "FLUX", xscale: str = "log",
            yscale: str = "log", features: Any = "all", label_offset: float = qa.FEATURE_LABEL_OFFSET,
            photospheric: bool = True, aperture: str | None = "disk", dpi: int = 150, **_: Any) -> Iterable[Path]:
        from astropy.io import fits

        out: list[Path] = []
        for inp in sorted(inputs):
            with fits.open(inp) as hdul:
                series = []
                tables = qa.select_spectra([h for h in hdul if h.name in qa.SPECTRUM_EXTENSIONS], aperture)
                for i, h in enumerate(tables):
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
    version = "3"   # 2 -> 3: "? [wavelength]" for unidentified peaks, lines identified within one sample / 50 km/s
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
