"""QA figures of image-like products: a quick look at anything with a SCI extension, and the
summary of a resampled image.

* ``quicklook_image`` -- the SCI array as one frame (cubes collapsed): what is in a file.
* ``qa_image_summary`` -- is a resampled image (``*_i2d.fits``) complete and how deep is it? The
  image next to the number of exposures behind every pixel and its signal-to-noise ratio, with the
  level and scatter of its sky in the notes.

Both read only the product; every map is a :class:`jwstflow.qa.Frame`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ... import qa
from ...images import read_image
from ...steps import RunContext, Step

log = logging.getLogger(__name__)


class QuicklookImage(Step):
    """PNG of the SCI extension (2-D images; 3-D cubes are nan-median collapsed).

    For each input file, the SCI array is rendered as one frame: cubes are
    first collapsed along the spectral axis with a nan-aware statistic
    (``collapse``: median/mean/sum/max; spaxels with less than
    ``min_coverage`` finite planes are blanked -- at the footprint edges a
    nanmedian over a handful of planes is noise that would otherwise stretch
    the display limits), then shown with robust percentile limits
    (``percentiles``) under the ``stretch`` (``asinh`` by default: linear
    around the background, logarithmic above, so a bright source does not
    blank the rest of the field; ``linear``/``sqrt``/``log`` are the
    alternatives), and a height-matched colorbar labelled with the data's
    BUNIT (surface brightness is converted to mJy/arcsec^2). Pixels without
    data are grey. When the SCI header carries a celestial WCS (cubes,
    resampled images) the frame has two sets of axes from one zero at the
    target position: offsets in pixels and offsets on the sky in arcsec;
    detector images keep plain pixel coordinates. Inputs without a SCI
    extension are skipped with a warning. One PNG per input file, named
    after it.
    """

    level = "qa"
    version = "5"   # 4 -> 5: pixel and sky offsets from one zero

    def run(
        self,
        inputs: list[Path],
        ctx: RunContext,
        *,
        percentiles: tuple[float, float] | list[float] = (1.0, 99.0),
        collapse: str = "median",
        min_coverage: float = 0.5,
        stretch: str = "asinh",
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
                sky = qa.SkyFrame.from_header(hdul["SCI"].header, data.shape, primary=hdul[0].header)
            frame = qa.Frame()
            frame.image(qa.collapse(data, collapse, min_coverage=min_coverage), unit=unit, sky=sky, stretch=stretch,
                        percentiles=tuple(percentiles), cmap=cmap)
            out.append(frame.finish(ctx.output_dir / f"{inp.stem}.png", dpi=dpi, notes=[
                inp.name, f"{collapse} over the spectral axis" if data.ndim > 2 else ""]))
        return out


#: [inch] What the labels around one map take: at its sides (axis labels, sky axis, colorbar) and
#: above/below it (as for the cube summary).
MAP_MARGIN = (1.45, 1.0)
#: Sigma-clipping of the sky statistics in the notes: rounds and threshold.
SKY_CLIP = (5, 3.0)


def sky_statistics(data: np.ndarray) -> tuple[float, float, int]:
    """Level (median) and scatter (1.4826 x MAD) of the pixels with data, clipped iteratively at
    ``SKY_CLIP`` sigma so the sources do not set them; and the number of pixels that remain."""
    values = data[np.isfinite(data)]
    level, scatter = float("nan"), float("nan")
    for _ in range(SKY_CLIP[0]):
        if values.size < 10:
            break
        level = float(np.median(values))
        scatter = 1.4826 * float(np.median(np.abs(values - level)))
        if scatter <= 0:
            break
        keep = np.abs(values - level) < SKY_CLIP[1] * scatter
        if keep.all():
            break
        values = values[keep]
    return level, scatter, int(values.size)


class QaImageSummary(Step):
    """Completeness and depth of a resampled image: the image, its exposures and its signal-to-noise.

    One figure per ``*_i2d.fits`` (any image with SCI and, ideally, ERR and CON): three frames on the
    image's own pixel grid, each with pixel and sky offsets from the target position --

    * the image (``stretch``, default asinh: linear around the sky, logarithmic above it);
    * the number of input exposures behind every pixel, counted from the context image (CON: one
      bit per exposure) -- dither gaps, the gaps between detectors and the edges of the mosaic show
      up here; where the product has no CON, its weight map (WHT) instead;
    * the signal-to-noise ratio per pixel above the sky level of the notes ((SCI - sky) / ERR, asinh
      stretch): where the sources are, also on an image that still carries its background.

    The notes name the band (pivot wavelength), the pixel scale and size of the field, and the sky:
    the median and the scatter (1.4826 x MAD) of the pixels with data, sigma-clipped so that the
    sources do not set them -- for a background-subtracted image the level should be near zero, and
    the scatter is the noise a faint source is measured against. Pixels without data are grey.
    """

    name = "qa_image_summary"
    level = "qa"
    inputs = ("*_i2d.fits",)
    version = "2"   # 1 -> 2: S/N above the sky level

    def run(self, inputs: list[Path], ctx: RunContext, *, stretch: str = "asinh",
            percentiles: tuple[float, float] | list[float] = (1.0, 99.5), width: float = 13.0, dpi: int = 150,
            **_: Any) -> list[Path]:
        out: list[Path] = []
        for path in inputs:
            image = read_image(path)
            if image is None:
                continue
            data = image.data
            sky = qa.SkyFrame.from_header(image.header, data.shape, primary=image.primary)
            finite = np.isfinite(data)
            map_height = (width / 3.0 - MAP_MARGIN[0]) * data.shape[0] / data.shape[1] + MAP_MARGIN[1]
            fig = qa.figure(figsize=(width, map_height + 0.5))
            grid = fig.add_gridspec(1, 3)
            ax_img, ax_exp, ax_snr = (fig.add_subplot(grid[0, k]) for k in range(3))
            qa.Frame(ax_img).image(data, unit=image.unit, sky=sky, stretch=stretch, percentiles=tuple(percentiles))

            n_exp = image.n_exposures
            if n_exp is not None:
                shown = np.where(finite & (n_exp > 0), n_exp, np.nan).astype(float)
                qa.Frame(ax_exp).image(shown, sky=sky, stretch="linear", vmin=0.0, vmax=max(float(np.nanmax(shown)), 1.0)
                                       if np.isfinite(shown).any() else 1.0, cbar_label="exposures per pixel", sides="btr")
            elif image.weight is not None:
                weight = np.where(finite & (image.weight > 0), image.weight, np.nan)
                qa.Frame(ax_exp).image(weight / np.nanmax(weight) if np.isfinite(weight).any() else weight, sky=sky,
                                       stretch="linear", vmin=0.0, vmax=1.0, cbar_label="relative weight", sides="btr")
            else:
                ax_exp.set_axis_off()
                log.info("%s: neither CON nor WHT; no exposure map", path.name)

            level, scatter, n_sky = sky_statistics(data)
            if image.error is not None:
                # above the sky: an image that still carries its background (MIRI's thermal sky) is
                # otherwise at high S/N everywhere and the sources do not show
                with np.errstate(all="ignore"):
                    snr = np.where(image.error > 0, (data - (level if np.isfinite(level) else 0.0)) / image.error, np.nan)
                qa.Frame(ax_snr).image(snr, sky=sky, stretch="asinh", percentiles=(1.0, 99.5),
                                       cbar_label="S/N per pixel above the sky", sides="btr")
            else:
                ax_snr.set_axis_off()
                log.info("%s: no ERR extension; no signal-to-noise map", path.name)

            scale = image.pixscale_arcsec
            band = image.band
            band_note = f"{band.name} ({qa.um(band.pivot_um)} pivot)" if band is not None else image.label
            field = (f"{data.shape[1]} $\\times$ {data.shape[0]} pix ({data.shape[1] * scale:.1f} $\\times$ "
                     f"{data.shape[0] * scale:.1f} arcsec), 1 pix = {scale:.4f} arcsec") if scale else ""
            qa.annotate(ax_img, path.name, "; ".join(t for t in (band_note, field) if t),
                        f"{int(finite.sum())} pixels with data"
                        + (f", median {np.nanmedian(np.where(n_exp > 0, n_exp, np.nan)):.0f} exposures" if n_exp is not None
                           and (n_exp > 0).any() else "")
                        + f"; sky {level:.3g} {image.unit or ''}, scatter {scatter:.3g} ({n_sky} pixels after clipping)",
                        sky.note() if sky is not None else "")
            qa.figlegend(fig)
            out.append(qa.save(fig, ctx.output_dir / f"{path.stem}_summary.png", dpi=dpi))
        return out
