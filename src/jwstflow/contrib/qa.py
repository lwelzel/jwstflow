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
    unless ``xscale: linear``.
    """

    level = "qa"
    version = "3"   # 2 -> 3: logarithmic wavelength axis by default (xscale parameter)

    def run(
        self,
        inputs: list[Path],
        ctx: RunContext,
        *,
        ylim: tuple[float, float] | list[float] | None = None,
        column: str = "FLUX",
        xscale: str = "log",
        dpi: int = 150,
        fmt: str = "png",
        **_: Any,
    ) -> Iterable[Path]:
        from astropy.io import fits

        out: list[Path] = []
        for inp in inputs:
            with fits.open(inp) as hdul:
                tables = [h for h in hdul if h.name in ("EXTRACT1D", "COMBINE1D")]
                if not tables:
                    log.warning("%s has no EXTRACT1D/COMBINE1D extension; skipped", inp.name)
                    continue
                fig, ax = qafig.subplots(figsize=(9, 4))
                colors = qafig.line_colors(len(tables))
                ylabel = ""
                for i, (h, color) in enumerate(zip(tables, colors)):
                    tab = h.data
                    if "WAVELENGTH" not in tab.names or column not in tab.names:
                        continue
                    values, ylabel = qafig.to_mjy(tab[column], _column_unit(h, column))
                    label = None
                    if len(tables) > 1 and len(tables) <= 12:
                        label = str(h.header.get("SLTNAME") or h.header.get("SRCNAME") or f"ext {i + 1}")
                    qafig.step(ax, tab["WAVELENGTH"], values, color=color, label=label)
                ax.set_xlabel(qafig.WAVE_LABEL)
                ax.set_ylabel(ylabel or f"{column}")
                qafig.set_wave_scale(ax, xscale)
                if ylim:
                    ax.set_ylim(*ylim)
                qafig.annotate(ax, inp.name)
                qafig.figlegend(fig)
                out.append(qafig.save(fig, ctx.output_dir / f"{inp.stem}.{fmt}", dpi=dpi))
        return out


class QuicklookImage(Step):
    """PNG of the SCI extension (2-D images; 3-D cubes are nan-median collapsed).

    For each input file, the SCI array is rendered as one image panel: cubes
    are first collapsed along the spectral axis with a nan-aware statistic
    (``collapse``: median/mean/sum/max), then shown in detector/sky pixels
    with robust percentile limits (``percentiles``), an optional non-linear
    ``stretch``, and a height-matched colorbar labelled with the data's
    BUNIT (surface brightness is converted to mJy/arcsec^2). Inputs without
    a SCI extension are skipped with a warning. One PNG per input file,
    named after it.
    """

    level = "qa"
    version = "2"   # 1 -> 2: qafig standard; cubes nan-collapsed instead of the middle slice

    def run(
        self,
        inputs: list[Path],
        ctx: RunContext,
        *,
        percentiles: tuple[float, float] | list[float] = (1.0, 99.0),
        collapse: str = "median",
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
            image = qafig.collapse(data, collapse)
            fig, ax = qafig.subplots(figsize=(6.5, 6))
            qafig.imshow(ax, image, unit=unit, stretch=stretch,
                         percentiles=tuple(percentiles), cmap=cmap or qafig.CMAP)
            ax.set(xlabel="x [pix]", ylabel="y [pix]")
            qafig.annotate(ax, inp.name)
            qafig.figlegend(fig)
            out.append(qafig.save(fig, ctx.output_dir / f"{inp.stem}.png", dpi=dpi))
        return out


class PlotStitch(Step):
    """Comparison figure of a stitched spectrum (``*_s1dcomb.ecsv``): the combination
    in black over its rescaled input segments.

    The segment files named in the ECSV metadata are searched in
    ``segments_stage`` (when given), in every stage directory of the run,
    next to the ECSV itself, and in the sibling runs of the target (so a
    combination run stitching across runs still finds its segments);
    segments that are not found any more are simply left out of the figure.
    The wavelength axis is logarithmic unless ``xscale: linear``.
    """

    level = "qa"
    inputs = ("*_s1dcomb.ecsv",)
    version = "2"   # 1 -> 2: log wavelength axis by default; segments found in sibling runs

    def run(self, inputs: list[Path], ctx: RunContext, *, segments_stage: str | None = None,
            xscale: str = "log", dpi: int = 150, **_: Any) -> Iterable[Path]:
        from astropy.table import Table

        out: list[Path] = []
        for inp in inputs:
            tab = Table.read(inp)
            wave = np.asarray(tab["WAVELENGTH"], dtype=float)
            flux, ylabel = qafig.to_mjy(np.asarray(tab["FLUX"], dtype=float),
                                        str(tab["FLUX"].unit or "Jy"))
            meta = tab.meta
            names = [str(n) for n in meta.get("inputs", [])]
            labels = [str(s) for s in meta.get("segments", names)]
            scales = [float(s) for s in meta.get("scales_applied", [1.0] * len(names))]
            fig, ax = qafig.subplots(figsize=(11, 4))
            colors = qafig.line_colors(len(names)) if len(names) > 1 else qafig.line_colors(2)
            for name, label, scale, color in zip(names, labels, scales, colors):
                segment = _find_file(name, ctx, segments_stage, inp.parent)
                if segment is None:
                    log.warning("%s: segment %s not found in any stage directory; left out", inp.name, name)
                    continue
                w, f, unit = _read_segment(segment)
                f, _ = qafig.to_mjy(f * scale, unit)
                qafig.step(ax, w, f, color=color, alpha=0.6, lw=0.7,
                           label=label + (f" x {scale:.3f}" if scale != 1.0 else ""))
            qafig.step(ax, wave, flux, color=qafig.MAIN_COLOR, label="stitched")
            crossovers = [float(b) for b in meta.get("crossovers_um", [])]
            for b in crossovers:
                ax.axvline(b, color="0.6", lw=0.6, ls=":")
            if crossovers:
                ax.plot([], [], color="0.6", lw=0.6, ls=":", label="crossover")
            ax.set(xlabel=qafig.WAVE_LABEL, ylabel=ylabel)
            qafig.set_wave_scale(ax, xscale)
            qafig.annotate(ax, inp.name)
            qafig.figlegend(fig)
            out.append(qafig.save(fig, ctx.output_dir / f"{inp.stem}.png", dpi=dpi))
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


def _read_segment(path: Path) -> tuple[np.ndarray, np.ndarray, str | None]:
    """(wavelength, flux, flux unit) of a segment file's EXTRACT1D/COMBINE1D table."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        for name in ("EXTRACT1D", "COMBINE1D"):
            if name in hdul:
                tab = hdul[name].data
                return (np.asarray(tab["WAVELENGTH"], dtype=float),
                        np.asarray(tab["FLUX"], dtype=float), _column_unit(hdul[name], "FLUX"))
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
