"""Figures of a joint cube (``*_s3dcomb.fits``): the join over its segments, the overlaps where
the ratios were measured, and whether neighbouring segments put the source in the same place."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ... import qa
from ...cubes.stitching import JOINT_SUFFIX, read_joint_records, region_flux
from ...steps import RunContext, Step
from .stitch import overlap_legend, overlap_panel

log = logging.getLogger(__name__)

#: The frames of the position figure show this many times the extent of the largest bright core
#: around the cores, and at least :data:`MIN_ZOOM` pixels either side.
ZOOM, MIN_ZOOM = 3.0, 8.0

def _spectrum_rows(table: Any, index: int, label: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(wavelength, flux, error)`` [um, Jy, Jy] of one entry of a SEGSPEC / OVLSPEC table."""
    rows = np.asarray(table["index"]) == index
    if label is not None:
        rows &= np.char.strip(np.asarray(table["segment"]).astype(str)) == label
    return (np.asarray(table["wavelength"], dtype=float)[rows], np.asarray(table["flux"], dtype=float)[rows],
            np.asarray(table["error"], dtype=float)[rows])


class QaCubeStitch(Step):
    """The figures of a joint cube: the join, the overlaps, and the positions across every crossover.
    In development with ``stitch_cubes``: joint cubes are not science-ready products.

    All of them are drawn from the records the ``stitch_cubes`` step wrote
    into its ``*_s3dcomb.fits`` product; only the first also reads the cube.

    ``*_spectrum.png`` -- the flux of the source across the join: the joint
    cube itself (black, read plane by plane from its SCI extension) over
    every segment as it went in, times the scale the stitch applied (thin,
    colored), with the crossovers. The flux is taken as the stitch takes it
    for its ratios -- the excess over the level of the field all segments
    cover, summed where the source stands out in it (or the plain sum of
    that field, for ``ratio_region: footprint``). Where the black line lies
    on its segment the cube holds what the segment held; the segments running
    on past their crossovers show how well neighbours agree once scaled. With
    ``features`` the spectral features are annotated.

    ``*_overlaps.png`` -- one panel per neighbouring pair, zoomed into the
    overlap, like ``plot_stitch_overlaps`` for spectra: both segments summed
    over the pair's measurement region as measured (with error bands), the
    crossover, the shaded window of the ratio, the bluer segment times the
    measured factor (dashed), and the factor with its uncertainty.

    ``*_positions.png`` -- one frame per pair, zoomed onto the source: the
    bluer segment collapsed over the ratio window (its excess over the
    field level in units of its peak, so all frames share one display
    scale), with the outline of its bright core (solid) and the outline of
    the bright core of the redder segment at the same wavelengths (dashed),
    both inside the window the stitch registered the pair in (``OVLCORE``).
    Where both cubes put the source in the same place the outlines
    coincide. The region the ratio of the pair was measured over
    (``OVLREG``) is outlined too (dotted): it must hold the source and
    nothing the edge of the field cuts. Each frame is tagged with its pair
    and wavelength and with the measured shift of the redder against the
    bluer segment.

    ``*_offsets.png`` (two or more pairs) -- those shifts against
    wavelength, east and north, with their uncertainties and a tenth of a
    spaxel for scale: the disagreement between neighbouring segments about
    where the source is, which the joint cube carries because every segment
    is placed by its own WCS.
    """

    name = "qa_cube_stitch"
    level = "qa"
    inputs = (f"*_{JOINT_SUFFIX}.fits",)
    version = "1"

    def run(self, inputs: list[Path], ctx: RunContext, *, xscale: str = "log", yscale: str = "log",
            features: Any = None, stretch: str = "asinh", ncols: int = 4, panel: float = 3.0, dpi: int = 150,
            **_: Any) -> list[Path]:
        out: list[Path] = []
        for path in inputs:
            try:
                rec = read_joint_records(path)
            except KeyError as exc:
                log.warning("%s carries no stitch records (%s); skipped", path.name, exc)
                continue
            out += [p for p in (self._spectrum(path, rec, ctx, xscale, yscale, features, dpi),
                                self._overlaps(path, rec, ctx, dpi),
                                self._positions(path, rec, ctx, stretch, ncols, panel, dpi),
                                self._offsets(path, rec, ctx, dpi)) if p is not None]
        return out

    # ------------------------------------------------------------------ the join over its segments
    def _spectrum(self, path: Path, rec: dict[str, Any], ctx: RunContext, xscale: str, yscale: str, features: Any,
                  dpi: int) -> Path | None:
        segments, common, source = rec["segments"], rec["common"], rec["source"]
        excess = str(rec["header"].get("JWFJCREG", "source")).strip() == "source"
        if not source.any() or not len(rec["segment_spectra"]):
            log.warning("%s: no field common to all segments; no spectrum figure", path.name)
            return None
        labels = [str(v).strip() for v in segments["label"]]
        colors = qa.line_colors(max(len(labels), 2))
        many = len(labels) > 4
        plot = qa.Spectrum(xscale=xscale, yscale=yscale)
        for k, (label, color) in enumerate(zip(labels, colors)):
            wave, flux, _ = _spectrum_rows(rec["segment_spectra"], k)
            scale = float(segments["scale"][k])
            plot.line(wave, qa.to_mjy(flux * scale, "Jy")[0], color=color, alpha=0.7, context=True,
                      label=(f"{len(labels)} segments, as joined" if k == len(labels) // 2 else None) if many
                      else label + (f" {qa.times(scale)}" if scale != 1.0 else ""))
        wave, flux = region_flux(path, source, common if excess else None)
        plot.line(wave, qa.to_mjy(flux, "Jy")[0], label="joint cube")
        plot.crossovers([float(c) for c in rec["overlaps"]["crossover"]])
        resampled = [lab for lab, mode in zip(labels, segments["placement"]) if str(mode).strip() != "identity"]
        scaling = str(rec["header"].get("JWFJCSCL", "none")).strip()
        scaled = {"measured": f"rescaled onto {rec['header'].get('JWFJCREF')}",
                  "adopted": f"scales of the stitched spectrum of {rec['header'].get('JWFJCSFM')}"}.get(scaling, "segments not rescaled")
        return plot.finish(
            ctx.output_dir / f"{path.stem}_spectrum.png", ylabel=qa.FLUX_LABEL, dpi=dpi, features=features,
            notes=[path.name,
                   (f"excess over the field level, summed over the {int(source.sum())} spaxels where the source stands "
                    f"out in the field all {len(labels)} segments cover ({int(common.sum())} spaxels)" if excess
                    else f"summed over the {int(common.sum())} spaxels all {len(labels)} segments cover") + f"; {scaled}",
                   (f"resampled onto the joint grid: {', '.join(resampled)}" if resampled
                    else "every segment on the joint grid as built (no resampling)")])

    # ------------------------------------------------------------------ the overlaps
    def _overlaps(self, path: Path, rec: dict[str, Any], ctx: RunContext, dpi: int) -> Path | None:
        overlaps = rec["overlaps"]
        n_pairs = len(overlaps)
        if n_pairs == 0:
            return None
        labels = [str(v).strip() for v in rec["segments"]["label"]]
        colors = qa.line_colors(max(len(labels), 2))
        ncols = min(3, n_pairs)
        nrows = (n_pairs + ncols - 1) // ncols
        fig, axes = qa.subplots(nrows, ncols, figsize=(max(4.6 * ncols, 7.0), 3.4 * nrows), squeeze=False)
        qa.annotate(axes[0][0], path.name)
        for i in range(n_pairs):
            ax = axes[i // ncols][i % ncols]
            row = overlaps[i]
            pair: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
            for label in (labels[i], labels[i + 1]):
                wave, flux, error = _spectrum_rows(rec["overlap_spectra"], i, label)
                pair.append((wave, qa.to_mjy(flux, "Jy")[0], qa.to_mjy(error, "Jy")[0]) if len(wave) else None)
            measured = bool(np.isfinite(row["ratio"]))
            window = (float(row["window_min"]), float(row["window_max"])) if np.isfinite(row["window_min"]) else None
            overlap_panel(ax, 0, [labels[i], labels[i + 1]], [colors[i], colors[i + 1]], pair, float(row["crossover"]),
                          window, float(row["ratio"]) if measured else None,
                          float(row["ratio_error"]) if measured else None)
            if i % ncols == 0:
                ax.set_ylabel(qa.FLUX_LABEL)
            if i // ncols == nrows - 1 or i + ncols >= n_pairs:
                ax.set_xlabel(qa.WAVE_LABEL)
        for i in range(n_pairs, nrows * ncols):
            axes[i // ncols][i % ncols].set_axis_off()
        overlap_legend(axes[0][0], rec["header"].get("JWFJCWFR"), segments="segments summed over the pair's region")
        counts = sorted({int(n) for n in overlaps["n_spaxels"]})
        qa.annotate(axes[0][0], f"region: {rec['header'].get('JWFJCREG')}, "
                    + (f"{counts[0]}" if len(counts) == 1 else f"{counts[0]}\N{EN DASH}{counts[-1]}") + " spaxels per pair")
        qa.figlegend(fig)
        return qa.save(fig, ctx.output_dir / f"{path.stem}_overlaps.png", dpi=dpi)

    # ------------------------------------------------------------------ the positions
    @staticmethod
    def _measured_pairs(rec: dict[str, Any]) -> list[int]:
        images = rec["overlap_images"]
        return [i for i in range(len(rec["overlaps"])) if np.isfinite(images[2 * i]).any() and np.isfinite(images[2 * i + 1]).any()]

    def _positions(self, path: Path, rec: dict[str, Any], ctx: RunContext, stretch: str, ncols: int, panel: float,
                   dpi: int) -> Path | None:
        overlaps, images, cores, regions = rec["overlaps"], rec["overlap_images"], rec["overlap_cores"], rec["overlap_regions"]
        pairs = self._measured_pairs(rec)
        if not pairs:
            log.warning("%s: no overlapping pair; no position figure", path.name)
            return None
        ny, nx = images.shape[1:]
        sky = qa.SkyFrame.from_header(rec["map_header"], (ny, nx), primary=rec["header"])
        ncol = max(1, min(int(ncols), len(pairs)))
        nrow = -(-len(pairs) // ncol)
        if len(pairs) < 3:
            panel *= 1.3                      # a figure of one or two frames can afford larger ones
        # one square window for all frames: around the cores, ZOOM times the extent of the largest
        in_cores = np.zeros((ny, nx), dtype=bool)
        for i in pairs:
            in_cores |= cores[2 * i] | cores[2 * i + 1]
        if in_cores.any():
            ys, xs = np.nonzero(in_cores)
            x_mid, y_mid = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
            half = max(ZOOM * 0.5 * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1), MIN_ZOOM)
        else:
            x_mid, y_mid, half = (nx - 1) / 2, (ny - 1) / 2, max(nx, ny) / 2
        half = min(half, max(nx, ny) / 2)
        # every image as its excess over the field level, in units of the peak of its bright core:
        # one display scale for all frames, whatever the brightness at their wavelengths
        shown: dict[int, np.ndarray] = {}
        for i in pairs:
            a, core = images[2 * i], cores[2 * i]
            both = np.isfinite(a) & np.isfinite(images[2 * i + 1])
            level = float(overlaps[i]["level_a"])
            level = level if np.isfinite(level) else float(np.nanmedian(a[both]))
            peak = float(np.nanmax(a[core])) if core.any() else float(np.nanmax(a[both]))
            shown[i] = np.where(both, (a - level) / (peak - level if peak > level else 1.0), np.nan)
        norm = qa.stretch_norm(np.concatenate([img.ravel() for img in shown.values()]), stretch, vmin=-0.05, vmax=1.0)
        fig, axes = qa.subplots(nrow, ncol, squeeze=False, sharex=True, sharey=True,
                                figsize=(max(panel * ncol + 1.6, 7.5), panel * nrow + 1.5))
        for n, ax in enumerate(axes.ravel()):
            if n >= len(pairs):
                ax.set_axis_off()
                continue
            i = pairs[n]
            row = overlaps[i]
            # the right-hand axis and the colorbar (one norm for all frames) in the last column only:
            # on the last frame of a shorter last row they would push the columns apart
            last_in_row = n % ncol == ncol - 1
            sides = ("l" if n % ncol == 0 else "") + ("b" if n + ncol >= len(pairs) else "") \
                + ("t" if n < ncol else "") + ("r" if last_in_row else "")
            frame = qa.Frame(ax)
            frame.image(shown[i], sky=sky, norm=norm, sides=sides, cbar=last_in_row,
                        cbar_label=r"$(I_\nu - I_\mathrm{field})\,/\,\mathrm{peak}$")
            frame.mask(cores[2 * i], label="bright core, bluer segment" if n == 0 else None)
            frame.mask(cores[2 * i + 1], label="bright core, redder segment" if n == 0 else None)
            frame.mask(regions[i], ls="dotted", color="white")      # named in the notes: white has no place in a legend
            qa.tag(ax, f"{str(row['segment_a']).strip()} | {str(row['segment_b']).strip()}\n{qa.um(float(row['crossover']))}")
            if np.isfinite(row["dra"]) and np.isfinite(row["ddec"]):
                east, north = round(float(row["dra"]) * 1e3) + 0, round(float(row["ddec"]) * 1e3) + 0   # + 0: no "-0"
                qa.tag(ax, f"({east:+d}, {north:+d}) mas".replace("-", "\N{MINUS SIGN}"), loc="lower left")
        axes[0, 0].set(xlim=(x_mid - half, x_mid + half), ylim=(y_mid - half, y_mid + half))
        qa.annotate(axes[0, 0], path.name,
                    "image: the bluer segment of each pair over its ratio window (both cores from those wavelengths)",
                    "dotted: the region the ratio is measured over",
                    "lower left of each frame: shift of the redder against the bluer segment (east, north)",
                    "; ".join(t for t in ((f"1 pix = {abs(sky.sy) * 1e3:.0f} mas" if sky is not None else ""),
                                          sky.note() if sky is not None else "") if t))
        qa.figlegend(fig)
        return qa.save(fig, ctx.output_dir / f"{path.stem}_positions.png", dpi=dpi)

    def _offsets(self, path: Path, rec: dict[str, Any], ctx: RunContext, dpi: int) -> Path | None:
        overlaps = rec["overlaps"]
        pairs = [i for i in self._measured_pairs(rec) if np.isfinite(overlaps[i]["dra"]) and np.isfinite(overlaps[i]["ddec"])]
        if len(pairs) < 2:
            return None                       # a single difference is a number, not a figure: it is tagged in its frame
        sky = qa.SkyFrame.from_header(rec["map_header"], rec["overlap_images"].shape[1:], primary=rec["header"])
        fig, ax = qa.subplots(figsize=(qa.FIGSIZE["spectrum"][0], 3.4))
        wave = np.array([float(overlaps[i]["crossover"]) for i in pairs])
        for key, err_key, color, marker, label in (("dra", "dra_error", qa.ACCENTS[0], "o", "east"),
                                                    ("ddec", "ddec_error", qa.ACCENTS[1], "s", "north")):
            value = np.array([float(overlaps[i][key]) for i in pairs]) * 1e3
            error = np.array([float(overlaps[i][err_key]) for i in pairs]) * 1e3
            ax.errorbar(wave, value, yerr=np.where(np.isfinite(error), error, 0.0), color=color, marker=marker, ms=4,
                        lw=qa.LW["thin"], capsize=2, label=f"shift of the redder against the bluer segment, {label}")
        qa.refline(ax, y=0.0)
        if sky is not None:
            tenth = 0.1 * abs(sky.sy) * 1e3
            for sign in (-1.0, 1.0):
                qa.refline(ax, y=sign * tenth, ls=":", label=f"a tenth of a spaxel ({tenth:.0f} mas)" if sign > 0 else None)
        ax.set(xlabel=qa.WAVE_LABEL, ylabel=qa.label("offset", "mas"))
        qa.set_wave_scale(ax, "log")
        labels = [f"{str(overlaps[i]['segment_a']).strip()} | {str(overlaps[i]['segment_b']).strip()}" for i in pairs]
        worst = max(pairs, key=lambda i: float(np.hypot(overlaps[i]["dra"], overlaps[i]["ddec"])))
        qa.annotate(ax, path.name,
                    f"{len(pairs)} crossovers, {labels[0]} to {labels[-1]}; each segment is placed by its own WCS",
                    f"largest: {str(overlaps[worst]['segment_a']).strip()} | {str(overlaps[worst]['segment_b']).strip()}, "
                    f"{float(np.hypot(overlaps[worst]['dra'], overlaps[worst]['ddec'])) * 1e3:.0f} mas")
        qa.figlegend(fig)
        return qa.save(fig, ctx.output_dir / f"{path.stem}_offsets.png", dpi=dpi)
