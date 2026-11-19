"""QA figures of IFU cubes (``*_s3d.fits``): what is in a cube beyond its collapsed image.

``quicklook_image`` shows the nan-median of a cube and nothing else. The two
steps here answer the questions that image cannot:

* ``qa_cube_summary`` -- is the cube complete and how good is it? The median
  image next to the coverage of every spaxel and its signal-to-noise ratio,
  and, against wavelength, the flux in the field and the fraction of the
  footprint that carries data in each plane.
* ``qa_cube_slices`` -- how does the field change with wavelength? Channel
  maps: the cube in a handful of wavelength bins on one common display scale.

Both read only the cube (SCI, ERR and the WCS of the SCI header); every map
is a :class:`jwstflow.qa.Frame`.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ... import qa
from ...steps.base import RunContext, Step

log = logging.getLogger(__name__)

#: A plane counts as complete when it carries data in this fraction of the typical plane's spaxels;
#: the field-integrated flux of an incomplete plane is not comparable and is not drawn.
COMPLETE_PLANE = 0.9
#: [inch] What the labels around one map take: at its sides (axis labels, sky axis, colorbar) and
#: above/below it.
MAP_MARGIN = (1.45, 1.0)


def _scale_note(sky: qa.SkyFrame | None) -> str:
    return f"1 pix = {abs(sky.sy):.3f} arcsec" if sky is not None else ""


class QaCubeSummary(Step):
    """Completeness and quality of an IFU cube: median image, coverage, signal-to-noise, and both against wavelength.

    One figure per ``*_s3d.fits``. Top row, three frames on the cube's pixel
    grid, each with pixel and sky offsets from the target position: the
    nan-median image (``stretch``, default asinh), the *coverage* (the
    fraction of wavelength planes in which a spaxel carries data -- gaps,
    dither edges and masked regions show up here) and the median
    signal-to-noise ratio per spaxel (SCI / ERR, log stretch; left out when
    the cube has no ERR extension). Below them, over the full width and on
    one wavelength axis: the flux density integrated over the field (in mJy,
    from the surface brightness and the spaxel solid angle; planes with
    incomplete coverage are left out, their sum is not comparable) and the
    fraction of the footprint -- the spaxels that carry data in *any* plane
    -- that carries data in each plane, which exposes missing or thinned
    planes. Spaxels with less than ``min_coverage`` finite planes are blanked
    in the median and S/N maps (their medians are noise). The figure is sized
    from the shape of the cube, so the maps fill their row.
    """

    name = "qa_cube_summary"
    level = "qa"
    inputs = ("*_s3d.fits",)
    version = "2"   # 1 -> 2: sky offsets on every map, the spectra over the full width

    def run(self, inputs: list[Path], ctx: RunContext, *, stretch: str = "asinh", min_coverage: float = 0.5,
            percentiles: tuple[float, float] | list[float] = (1.0, 99.5), xscale: str = "log",
            yscale: str = "log", width: float = 13.0, dpi: int = 150, **_: Any) -> list[Path]:
        out: list[Path] = []
        for path in inputs:
            cube = qa.read_cube(path)
            if cube is None:
                continue
            data, error, wave, sky = cube["data"], cube["error"], cube["wave"], cube["sky"]
            finite = np.isfinite(data)
            coverage = finite.mean(axis=0)
            footprint = coverage > 0
            image = qa.collapse(data, "median", min_coverage=min_coverage)

            # three maps side by side set the height of their row; the two spectra take fixed heights
            map_height = (width / 3.0 - MAP_MARGIN[0]) * data.shape[1] / data.shape[2] + MAP_MARGIN[1]
            heights = [map_height, 1.9, 0.9]
            fig = qa.figure(figsize=(width, sum(heights) + 0.5))
            grid = fig.add_gridspec(3, 3, height_ratios=heights)
            ax_img, ax_cov, ax_snr = (fig.add_subplot(grid[0, k]) for k in range(3))
            ax_flux = fig.add_subplot(grid[1, :])
            ax_fill = fig.add_subplot(grid[2, :], sharex=ax_flux)

            qa.Frame(ax_img).image(image, unit=cube["unit"], sky=sky, stretch=stretch, percentiles=tuple(percentiles))
            qa.Frame(ax_cov).image(np.where(footprint, coverage, np.nan), sky=sky, stretch="linear", vmin=0.0, vmax=1.0,
                                   cbar_label="fraction of planes with data", sides="btr")
            if error is not None:
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    snr = np.nanmedian(np.where(error > 0, data / error, np.nan), axis=0)
                qa.Frame(ax_snr).image(np.where(coverage >= min_coverage, snr, np.nan), sky=sky, stretch="log",
                                       percentiles=(5.0, 99.5), cbar_label="median S/N per plane", sides="btr")
            else:
                ax_snr.set_axis_off()
                log.info("%s: no ERR extension; no signal-to-noise map", path.name)

            with np.errstate(all="ignore"):
                total = np.where(finite.any(axis=(1, 2)), np.nansum(data, axis=(1, 2)), np.nan)
            if cube["pixar_sr"] is not None and str(cube["unit"] or "").replace(" ", "") == "MJy/sr":
                total, flux_label = total * cube["pixar_sr"] * 1e9, qa.FLUX_LABEL
            else:
                total, flux_label = qa.to_mjy(total, cube["unit"])
                flux_label = f"sum over the field {flux_label}"
            n_foot = max(int(footprint.sum()), 1)
            filled = finite.sum(axis=(1, 2)) / n_foot
            complete = filled >= COMPLETE_PLANE * float(np.nanmedian(filled[filled > 0])) if (filled > 0).any() else filled > 0
            qa.step(ax_flux, wave, np.where(complete, total, np.nan), color=qa.MAIN_COLOR,
                    label="flux in the field (complete planes)")
            ax_flux.set_ylabel(flux_label)
            ax_flux.tick_params(labelbottom=False)
            qa.set_flux_scale(ax_flux, yscale)
            qa.step(ax_fill, wave, filled, color=qa.MAIN_COLOR, gap_factor=None)
            ax_fill.set(xlabel=qa.WAVE_LABEL, ylabel="footprint with data", ylim=(-0.05, 1.05))
            qa.set_wave_scale(ax_fill, xscale)

            empty = int((~finite.any(axis=(1, 2))).sum())
            qa.annotate(ax_img, path.name,
                        f"{data.shape[0]} planes, {qa.um_range(float(np.nanmin(wave)), float(np.nanmax(wave)))}; "
                        f"{int(footprint.sum())} spaxels in the footprint; {empty} empty, "
                        f"{int((~complete).sum()) - empty} incomplete plane(s) (left out of the flux)",
                        "; ".join(t for t in (_scale_note(sky), sky.note() if sky is not None else "") if t))
            qa.figlegend(fig)
            out.append(qa.save(fig, ctx.output_dir / f"{path.stem}_summary.png", dpi=dpi))
        return out


class QaCubeSlices(Step):
    """Channel maps of an IFU cube: the field in ``n_slices`` wavelength bins on one display scale.

    One figure per ``*_s3d.fits``: the wavelength range is cut into
    ``n_slices`` bins at equal spacing (in log wavelength), and each panel
    shows the nan-median of the ``width`` fraction of the cube's planes
    around its bin centre -- narrow enough to follow the morphology with
    wavelength, wide enough to beat single-plane noise. All panels share one
    stretch and one set of display limits (``stretch``, default asinh, over
    robust limits of all panels together), so brightness can be compared
    between them; the colorbar at the end of each row serves the whole row.
    Every panel is a frame with both sets of axes -- offsets in pixels
    (bottom/left) and on the sky in arcsec (top/right) from one zero at the
    target position; the tick labels are written once, on the outer panels.
    Each panel is tagged with its wavelength range; bins without data are
    left empty.
    """

    name = "qa_cube_slices"
    level = "qa"
    inputs = ("*_s3d.fits",)
    version = "2"   # 1 -> 2: pixel and sky offsets on every panel

    def run(self, inputs: list[Path], ctx: RunContext, *, n_slices: int = 8, width: float = 0.02,
            stretch: str = "asinh", percentiles: tuple[float, float] | list[float] = (1.0, 99.8),
            ncols: int = 4, panel: float = 2.7, dpi: int = 150, **_: Any) -> list[Path]:
        out: list[Path] = []
        for path in inputs:
            cube = qa.read_cube(path)
            if cube is None:
                continue
            data, wave, sky = cube["data"], cube["wave"], cube["sky"]
            nplanes = data.shape[0]
            usable = np.nonzero(np.isfinite(data).any(axis=(1, 2)) & np.isfinite(wave) & (wave > 0))[0]
            if usable.size < 2:
                log.warning("%s: fewer than two planes with data; no channel maps", path.name)
                continue
            n = max(2, min(int(n_slices), usable.size))
            half = max(1, round(0.5 * width * nplanes))
            edges = np.geomspace(wave[usable[0]], wave[usable[-1]], n + 1)
            centres = np.sqrt(edges[:-1] * edges[1:])
            slices: list[tuple[np.ndarray, float, float] | None] = []
            for centre in centres:
                k = int(usable[np.argmin(np.abs(wave[usable] - centre))])
                lo, hi = max(0, k - half), min(nplanes, k + half + 1)
                img = qa.collapse(data[lo:hi], "median")
                slices.append((img, float(wave[lo]), float(wave[hi - 1])) if np.isfinite(img).any() else None)
            shown = [s[0] for s in slices if s is not None]
            if not shown:
                continue
            stacked, _ = qa.to_mjy(np.concatenate([img.ravel() for img in shown]), cube["unit"])
            lo_hi = [qa.robust_limits(qa.to_mjy(img, cube["unit"])[0], tuple(percentiles)) for img in shown]
            norm = qa.stretch_norm(stacked, stretch, vmin=min(v[0] for v in lo_hi), vmax=max(v[1] for v in lo_hi))
            ncol = max(1, min(int(ncols), n))
            nrow = -(-n // ncol)
            fig, axes = qa.subplots(nrow, ncol, squeeze=False, sharex=True, sharey=True,
                                    figsize=(panel * ncol + 1.6, panel * data.shape[1] / data.shape[2] * nrow + 1.5))
            for i, ax in enumerate(axes.ravel()):
                if i >= n or slices[i] is None:
                    ax.set_axis_off()
                    continue
                img, w_lo, w_hi = slices[i]
                last_in_row = (i % ncol == ncol - 1) or i == n - 1
                sides = ("l" if i % ncol == 0 else "") + ("b" if i + ncol >= n else "") \
                    + ("t" if i < ncol else "") + ("r" if last_in_row else "")
                qa.Frame(ax).image(img, unit=cube["unit"], sky=sky, norm=norm, cbar=last_in_row, sides=sides)
                qa.tag(ax, qa.um_range(w_lo, w_hi, ".3f"))
            qa.annotate(axes[0, 0], path.name,
                        f"nan-median of {2 * half + 1} planes per panel; one display scale for all panels",
                        "; ".join(t for t in (_scale_note(sky), sky.note() if sky is not None else "") if t))
            qa.figlegend(fig)
            out.append(qa.save(fig, ctx.output_dir / f"{path.stem}_slices.png", dpi=dpi))
        return out
