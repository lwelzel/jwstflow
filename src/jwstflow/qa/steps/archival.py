"""Archival photometry and spectra over a reduction's spectra (``plot_archival``)."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ... import qa
from ...steps import RunContext, Step

log = logging.getLogger(__name__)

#: one marker per source, in order. No circles: they read like the hexagons that mark JWST photometry.
MARKERS = ("s", "^", "D", "v", "P", "X", "<", ">", "*", "d")
#: the length of an upper limit's arrow, as a fraction of the limit (on a log axis about a quarter decade)
ARROW = 0.45


class PlotArchival(Step):
    """Archival photometry and spectra over the spectra of a reduction (optional, data kept by hand).

    For every input spectrum (``*_x1d`` / ``*_s1d`` / a stitched ``*.ecsv``;
    of a file of named apertures the target ``aperture``, default ``disk``)
    the archival data of the target -- ECSV files in ``<target>/archival/``
    (``archival``: patterns relative to the target directory, or absolute;
    layout in :mod:`jwstflow.spectra.archival` and ``docs/archival_data.md``)
    -- are drawn over it in two figures: ``*_archival.png`` over the
    spectrum's wavelength range (widened by ``margin``) and, with
    ``full_range`` (default), ``*_archival_sed.png`` over everything the
    archival data cover. The spectrum is the black line; archival spectra
    are thin colored lines; photometry is one marker and color per source
    (``CITATION``), with the 1-sigma error as an I-shaped bar and the band
    width as a thin horizontal bar; upper limits are hollow markers with a
    downward arrow at the limit, their significance in the legend.
    ``features`` annotates the spectral features on the first figure (off by
    default). Without archival files the step draws nothing and says so.

    The archival files are parameters, not inputs: after editing them, re-run
    the step with ``--force``. When inputs share a file name (the variants of
    a combination), the figure names carry the stage.
    """

    level = "qa"
    batch = "all"
    version = "1"

    def run(self, inputs: list[Path], ctx: RunContext, *, archival: list[str] | str = ("archival/*.ecsv",),
            full_range: bool = True, margin: float = 1.25, features: Any = None,
            label_offset: float = qa.FEATURE_LABEL_OFFSET, photospheric: bool = True, aperture: str = "disk",
            dpi: int = 150, fmt: str = "png", **_: Any) -> Iterable[Path]:
        from ...spectra.archival import find_archival, read_archival

        target_dir = ctx.target_dir if ctx.target_dir is not None else Path(ctx.root).parent
        files = find_archival(target_dir, archival)
        if not files:
            log.warning("no archival files (%s under %s): nothing drawn", archival, target_dir)
            return []
        data = [read_archival(p) for p in files]
        groups = _groups(data)
        stems = Counter(Path(p).stem for p in inputs)
        out: list[Path] = []
        for inp in sorted(inputs):
            try:
                wave, flux, _, unit = qa.read_target_spectrum(inp, aperture)
            except (ValueError, OSError) as exc:
                log.warning("%s: %s; skipped", inp.name, exc)
                continue
            values, ylabel = qa.to_mjy(flux, unit)
            name = inp.stem + (f"_{qa.stage_of(inp)}" if stems[inp.stem] > 1 else "")
            notes = [inp.name if stems[inp.stem] == 1 else f"{qa.stage_of(inp)}/{inp.name}",
                     "archival: " + ", ".join(p.name for p in files)]
            ok = np.isfinite(wave) & np.isfinite(values) & (values > 0)
            if not ok.any():
                log.warning("%s: no positive flux to draw; skipped", inp.name)
                continue
            spectrum_range = (float(wave[ok].min()) / margin, float(wave[ok].max()) * margin)
            everything = [float(wave[ok].min()), float(wave[ok].max())] + \
                [float(x) for d in data for x in d.wavelength[np.isfinite(d.wavelength)]]
            views = [("_archival", spectrum_range, features)]
            if full_range:
                views.append(("_archival_sed", (min(everything) / 1.4, max(everything) * 1.4), None))
            for suffix, wave_range, annotate in views:
                plot = qa.Spectrum()
                plot.line(wave, values, label="JWST")
                ylim = _draw_archival(plot, data, groups, wave_range, (wave[ok], values[ok]))
                out.append(plot.finish(ctx.output_dir / f"{name}{suffix}.{fmt}", ylabel=ylabel or qa.FLUX_LABEL,
                                       notes=notes, features=annotate, label_offset=label_offset,
                                       photospheric=photospheric, wave_range=wave_range, ylim=ylim, dpi=dpi))
        return out


def _groups(data: list[Any]) -> dict[str, tuple[str, Any]]:
    """source -> (marker, color) for every source of photometry, in order of its shortest wavelength."""
    first: dict[str, float] = {}
    for d in data:
        if d.kind != "photometry":
            continue
        for source, wavelength in zip(_sources(d), d.wavelength):
            first[source] = min(first.get(source, np.inf), float(wavelength))
    order = sorted(first, key=first.get)
    colors = qa.line_colors(max(len(order), 2))
    # neighbours in wavelength take colors from opposite halves of the colormap, so they stand apart
    half = (len(colors) + 1) // 2
    interleaved = [colors[k // 2 + (k % 2) * half] for k in range(len(colors))]
    return {s: (MARKERS[k % len(MARKERS)], interleaved[k]) for k, s in enumerate(order)}


def _sources(d: Any) -> list[str]:
    return [str(c or r or d.path.stem) for c, r in zip(d.citation, d.reference)]


def _draw_archival(plot: Any, data: list[Any], groups: dict[str, tuple[str, Any]],
                   wave_range: tuple[float, float], spectrum: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
    """Photometry and spectra on ``plot.ax``; returns y limits that hold everything in ``wave_range``."""
    ax = plot.ax
    lo_w, hi_w = wave_range
    w_s, f_s = spectrum
    inside = (w_s >= lo_w) & (w_s <= hi_w)
    lows = [float(np.percentile(f_s[inside], 0.5))] if inside.any() else []
    highs = [float(f_s[inside].max())] if inside.any() else []
    sigma = {d.upper_limit_sigma for d in data if d.kind == "photometry" and d.upper_limit.any()}
    spectra = [d for d in data if d.kind == "spectrum"]
    for d, color in zip(spectra, qa.line_colors(len(spectra) + 2)[1:-1]):
        f, _ = qa.to_mjy(d.flux, "Jy")
        plot.line(d.wavelength, f, color=color, label=d.label, context=True)
        sel = (d.wavelength >= lo_w) & (d.wavelength <= hi_w) & np.isfinite(f) & (f > 0)
        if sel.any():
            lows.append(float(np.percentile(f[sel], 0.5)))
            highs.append(float(f[sel].max()))
    labelled: set[str] = set()
    for d in (d for d in data if d.kind == "photometry"):
        flux, _ = qa.to_mjy(d.flux, "Jy")
        error, _ = qa.to_mjy(d.error, "Jy")
        sources = np.asarray(_sources(d), dtype=object)
        for source in dict.fromkeys(sources):
            marker, color = groups[source]
            for limit in (False, True):
                sel = (sources == source) & (d.upper_limit == limit) & np.isfinite(flux) & (flux > 0)
                if not sel.any():
                    continue
                x, y = d.wavelength[sel], flux[sel]
                half = np.where(np.isfinite(d.bandwidth[sel]), d.bandwidth[sel] / 2.0, 0.0)
                ax.errorbar(x, y, xerr=half, fmt="none", ecolor=color, elinewidth=0.7, alpha=0.55, capsize=0,
                            zorder=3)
                label = source if source not in labelled else None
                if limit:   # hollow marker at the limit, arrow down
                    down, up = ARROW * y, np.zeros_like(y)
                    ax.errorbar(x, y, yerr=down, uplims=True, fmt=marker, color=color, mfc="white", mec=color,
                                ms=5.5, mew=1.1, elinewidth=1.0, capsize=2.5, zorder=4, label=label)
                else:       # filled marker, I-shaped 1-sigma bar (kept above zero on the log axis)
                    e = np.where(np.isfinite(error[sel]), error[sel], 0.0)
                    down, up = np.minimum(e, 0.95 * y), e
                    ax.errorbar(x, y, yerr=np.vstack([down, up]), fmt=marker, color=color, ms=5.5, elinewidth=1.0,
                                capsize=2.5, zorder=4, label=label)
                shown = (x >= lo_w) & (x <= hi_w)
                if shown.any():
                    lows.append(float(np.min((y - down)[shown])))
                    highs.append(float(np.max((y + up)[shown])))
                if label:
                    labelled.add(source)
    if sigma:
        text = ", ".join(f"{s:g}" for s in sorted(s for s in sigma if s is not None))
        ax.plot([], [], ls="none", marker=r"$\downarrow$", ms=10, color=qa.REF_COLOR,
                label=rf"upper limit ({text}$\sigma$), hollow")
    lows = [v for v in lows if v > 0]
    if not lows or not highs:
        return None
    return min(lows) / 1.6, max(highs) * 1.6
