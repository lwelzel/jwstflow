"""QA figures of image-like products: a quick look at anything with a SCI extension, the
summary of a resampled image, and the registration of a run's images.

* ``quicklook_image`` -- the SCI array as one frame (cubes collapsed): what is in a file.
* ``qa_image_summary`` -- is a resampled image (``*_i2d.fits``) complete and how deep is it? The
  image next to the number of exposures behind every pixel and its signal-to-noise ratio, with the
  level and scatter of its sky in the notes.
* ``qa_image_astrometry`` -- how well do the images of a run agree on the sky, with each other and
  with Gaia DR3? The point-source offsets of every image and their medians against wavelength,
  from the ``*_astrom.fits`` of ``image_astrometry``.

All read only the product; every map is a :class:`jwstflow.qa.Frame`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ... import qa
from ...images import read_image
from ...steps import RunContext, Step

log = logging.getLogger(__name__)


def target_reference(ctx: RunContext) -> dict[str, Any]:
    """The zero of a figure's axes: the workflow's ``target_coords`` when given (the position every
    analysis step uses), as keywords of :meth:`qa.SkyFrame.from_header`; else nothing, and the frame
    takes the header's TARG_RA / TARG_DEC (the pointing, which can lie far from the source: HH 30's
    NIRCam mosaics are pointed 51 arcsec from it)."""
    from ...data.targets import parse_coords

    coords = getattr(ctx, "target_coords", None) or {}
    if coords.get("ra") is None or coords.get("dec") is None:
        return {}
    return {"ref_world": parse_coords(coords["ra"], coords["dec"]), "reference": "target"}


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
    version = "6"   # 5 -> 6: zero at the workflow's target; 4 -> 5: pixel and sky offsets from one zero

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
                sky = qa.SkyFrame.from_header(hdul["SCI"].header, data.shape, primary=hdul[0].header,
                                              **target_reference(ctx))
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

    * the image less the sky level of the notes (``stretch``, default asinh: linear around the sky,
      logarithmic above it);
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
    version = "4"   # 3 -> 4: the image less its sky; 2 -> 3: zero at the workflow's target; 1 -> 2: S/N above the sky

    def run(self, inputs: list[Path], ctx: RunContext, *, stretch: str = "asinh",
            percentiles: tuple[float, float] | list[float] = (1.0, 99.5), width: float = 13.0, dpi: int = 150,
            **_: Any) -> list[Path]:
        out: list[Path] = []
        for path in inputs:
            image = read_image(path)
            if image is None:
                continue
            data = image.data
            sky = qa.SkyFrame.from_header(image.header, data.shape, primary=image.primary, **target_reference(ctx))
            finite = np.isfinite(data)
            map_height = (width / 3.0 - MAP_MARGIN[0]) * data.shape[0] / data.shape[1] + MAP_MARGIN[1]
            fig = qa.figure(figsize=(width, map_height + 0.5))
            grid = fig.add_gridspec(1, 3)
            ax_img, ax_exp, ax_snr = (fig.add_subplot(grid[0, k]) for k in range(3))
            level, scatter, n_sky = sky_statistics(data)
            # less the sky level of the notes: a background of ~6 MJy/sr (MIRI) under sources of a few
            # hundredths would leave the colour bar two six-digit numbers apart
            base = level if np.isfinite(level) else 0.0
            qa.Frame(ax_img).image(data - base, unit=image.unit, sky=sky, stretch=stretch, percentiles=tuple(percentiles))

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
                        "left: the image less the sky level", sky.note() if sky is not None else "")
            qa.figlegend(fig)
            out.append(qa.save(fig, ctx.output_dir / f"{path.stem}_summary.png", dpi=dpi))
        return out


#: The comparisons of an ``*_astrom.fits`` product (its SUMMARY/MATCHES ``against``): color (categorical
#: marks on white, in the fixed order of the accents) and marker -- they differ in both.
ASTROMETRY_STYLE: dict[str, tuple[str, str]] = {"reference": (qa.ACCENTS[0], "o"), "gaia": (qa.ACCENTS[1], "^")}
DRA_MAS_LABEL = qa.label(r"$\Delta\alpha\,\cos\delta$", "mas")
DDEC_MAS_LABEL = qa.label(r"$\Delta\delta$", "mas")
#: [inch] Room around the data square of an offsets panel (tick labels, axis labels).
ASTROMETRY_PANEL_MARGIN = 0.95
#: Lines a panel note can have (band, reference image, two comparisons), for the strip above its data.
ASTROMETRY_NOTE_LINES = 3
#: [inch] Height of each of the two panels of medians against wavelength.
ASTROMETRY_WAVE_PANEL = 1.7
#: Factor between the wavelengths at which the two comparisons of one image are drawn, so that
#: their error bars stand side by side.
ASTROMETRY_DODGE = 1.012
#: A median of fewer stars than this is drawn hollow: one or two stars are no measure of an offset.
ASTROMETRY_MIN_STARS = 3


def _astrometry_tables(path: Path) -> dict[str, Any] | None:
    """The tables and primary header of an ``*_astrom.fits`` (None, with a warning, when one is missing)."""
    from astropy.io import fits
    from astropy.table import Table

    with fits.open(path) as hdul:
        names = {h.name for h in hdul}
        missing = {"IMAGES", "SUMMARY", "MATCHES"} - names
        if missing:
            log.warning("%s has no %s extension(s); skipped", path.name, ", ".join(sorted(missing)))
            return None
        out: dict[str, Any] = {"header": hdul[0].header.copy()}
        for name in ("IMAGES", "SUMMARY", "MATCHES"):
            out[name.lower()] = Table(hdul[name].data)
    for name in ("images", "summary", "matches"):
        for col in out[name].colnames:
            if out[name][col].dtype.kind in "SU":
                out[name][col] = np.char.strip(np.asarray(out[name][col]).astype(str))
    return out


class QaImageAstrometry(Step):
    """How well the images of a run are registered: point-source offsets of every image against the
    reference image and Gaia DR3, and their medians against wavelength.

    One figure per ``*_astrom.fits`` (``image_astrometry``), read from the product alone. On top,
    one panel per image, blue to red: the offset of every matched point source, image minus
    reference, in right ascension times cos(dec) (east to the left, as on the sky) against
    declination [mas] -- against the reference image (circles) and against Gaia DR3 at the
    image's epoch (triangles); matches left out of the median (clipped, or Gaia sources without a
    proper motion) hollow; the median with its error (as ``image_astrometry`` records it) as the
    large black-edged mark; a dotted circle of a tenth of the image's pixel for scale. Each panel
    notes its band and, per comparison, the median offsets [mas] and the matches in the median (of
    those matched). Below, over the full width, the medians of both comparisons against the pivot
    wavelength (log axis; the bands named on top), one panel per coordinate, the reference image's
    band dashed: what the science that compares positions across filters has to allow for. The
    legend notes the product, the reference image and the Gaia query; ``details`` adds the cuts
    the product records. Each panel keeps a strip above its data for its note, and matches left out
    of the median that lie beyond the panel's range (clipped outliers) are not drawn.

    The figure is neither a spectrum nor a frame: scatter plots of offsets on equal axes and
    measurements against wavelength (``qa.set_wave_scale``).
    """

    name = "qa_image_astrometry"
    level = "qa"
    inputs = ("*_astrom.fits",)
    version = "3"   # 2 -> 3: those medians do not set the scale; 1 -> 2: medians of fewer than 3 stars hollow; the observation where a band repeats

    def run(self, inputs: list[Path], ctx: RunContext, *, ncols: int = 4, panel: float = 3.0, details: bool = False,
            dpi: int = 150, **_: Any) -> list[Path]:
        out: list[Path] = []
        for path in inputs:
            prod = _astrometry_tables(path)
            if prod is None:
                continue
            if len(prod["summary"]) == 0:
                log.warning("%s: no comparison recorded (one image and no Gaia?); nothing to draw", path.name)
                continue
            out.append(self._figure(path, prod, ctx, max(1, int(ncols)), float(panel), bool(details), dpi))
        return out

    def _figure(self, path: Path, prod: dict[str, Any], ctx: RunContext, ncols: int, panel: float, details: bool,
                dpi: int) -> Path:
        images, summary, matches, hdr = prod["images"], prod["summary"], prod["matches"], prod["header"]
        order = np.argsort(np.where(np.isfinite(images["pivot"]), images["pivot"], np.inf), kind="stable")
        images = images[order]
        n = len(images)
        cols = min(ncols, n)
        rows = -(-n // cols)
        width = max(qa.FIGSIZE["spectrum"][0], cols * panel + 1.0)
        box = min(width / cols, panel + 0.8) - ASTROMETRY_PANEL_MARGIN    # [inch] the data square of a panel
        strip = ASTROMETRY_NOTE_LINES * qa.FONT["note"] * 1.25 / 72.0 + 0.12   # [inch] its note, above the data
        row = box + strip + ASTROMETRY_PANEL_MARGIN
        fig = qa.figure(figsize=(width, rows * row + 2 * ASTROMETRY_WAVE_PANEL + 1.6))
        grid = fig.add_gridspec(rows + 2, cols, height_ratios=[row] * rows + [ASTROMETRY_WAVE_PANEL] * 2)
        ref_band = str(hdr.get("JWFASRBD", "")).strip()
        names = {"reference": ref_band, "gaia": "Gaia DR3"}
        bands = [str(b) for b in images["band"]]
        for k, image in enumerate(images):
            ax = fig.add_subplot(grid[k // cols, k % cols])
            self._offsets_panel(ax, image, summary, matches, names, strip / box, first_column=k % cols == 0,
                                last_row=k // cols == rows - 1 or k + cols >= n,
                                epoch=bands.count(str(image["band"])) > 1)
        ax_ra = fig.add_subplot(grid[rows, :])
        ax_dec = fig.add_subplot(grid[rows + 1, :], sharex=ax_ra)
        self._wave_panels(ax_ra, ax_dec, images, summary)
        # the marks every panel shares, once: what hollow and large marks mean
        ax_ra.plot([], [], ls="none", marker="o", ms=3.2, mfc="none", mec=qa.CONTEXT_COLOR, mew=0.6,
                   label="matched, not in the median")
        ax_ra.errorbar([], [], xerr=[], yerr=[], ls="none", marker="o", ms=7, mfc=qa.CONTEXT_COLOR, mec=qa.MAIN_COLOR,
                       mew=0.9, ecolor=qa.MAIN_COLOR, elinewidth=qa.LW["main"], capsize=2, label=r"median $\pm$ its error")
        ax_ra.errorbar([], [], xerr=[], yerr=[], ls="none", marker="o", ms=7, mfc="white", mec=qa.MAIN_COLOR,
                       mew=0.9, ecolor=qa.MAIN_COLOR, elinewidth=qa.LW["main"], capsize=2,
                       label=f"median of fewer than {ASTROMETRY_MIN_STARS} stars")
        qa.annotate(ax_ra, *self._notes(path, hdr, images, details))
        qa.figlegend(fig)
        return qa.save(fig, ctx.output_dir / f"{path.stem}.png", dpi=dpi)

    @staticmethod
    def _offsets_panel(ax: Any, image: Any, summary: Any, matches: Any, names: dict[str, str], strip: float, *,
                       first_column: bool, last_row: bool, epoch: bool = False) -> None:
        """The offsets of one image on equal axes within +-lim, and above them a strip (``strip`` of the
        data height) that holds the panel's note and no data."""
        from matplotlib.patches import Circle
        from matplotlib.ticker import MaxNLocator

        file = str(image["file"])
        scale = float(image["pixscale"]) * 1e3 if np.isfinite(image["pixscale"]) else np.nan
        extent = [0.15 * scale] if np.isfinite(scale) else [1.0]
        note = [f"{image['band']}  {qa.um(float(image['pivot']))}" if np.isfinite(image["pivot"]) else str(image["band"])]
        obs = re.search(r"-(o\d{3})_", file)
        if epoch and obs:                       # the band more than once: which observation this is
            note[0] += f", {obs.group(1)}"
        if bool(image["reference"]):
            note.append("reference image")
        drawn: list[tuple[Any, Any, str, str]] = []
        for against, (color, symbol) in ASTROMETRY_STYLE.items():
            m = matches[(matches["file"] == file) & (matches["against"] == against)]
            s = summary[(summary["file"] == file) & (summary["against"] == against)]
            if len(s) == 0:
                continue
            row = s[0]
            drawn.append((m, row, color, symbol))
            used = np.asarray(m["used"], bool)
            if used.any():
                extent += [float(np.max(np.abs(m["dra"][used]))), float(np.max(np.abs(m["ddec"][used])))]
            # a median of one or two stars (a wrong match after a large first shift) does not set the scale:
            # its value is in the panel's note
            if np.isfinite(row["dra"]) and np.isfinite(row["ddec"]) and int(row["n_used"]) >= ASTROMETRY_MIN_STARS:
                extent += [abs(float(row["dra"])) + _finite(row["dra_error"]), abs(float(row["ddec"])) + _finite(row["ddec_error"])]
        lim = 1.15 * max(extent)
        for m, row, color, symbol in drawn:
            against = str(row["against"])
            used = np.asarray(m["used"], bool)
            inside = (np.abs(m["dra"]) <= lim) & (np.abs(m["ddec"]) <= lim)     # clipped outliers beyond stay out
            ax.plot(m["dra"][used & inside], m["ddec"][used & inside], ls="none", marker=symbol, ms=3.2, mfc=color,
                    mec=color, mew=0.5, alpha=0.75, zorder=3, label=f"image $-$ {names[against]}"
                    + (" (the reference image)" if against == "reference" else " at the image's epoch"))
            ax.plot(m["dra"][~used & inside], m["ddec"][~used & inside], ls="none", marker=symbol, ms=3.2, mfc="none",
                    mec=color, mew=0.6, alpha=0.75, zorder=3)
            n_used, n_matched = int(row["n_used"]), int(row["n_matched"])
            count = f"N = {n_used}" + (f"/{n_matched}" if n_matched > n_used else "")
            if np.isfinite(row["dra"]) and np.isfinite(row["ddec"]):
                few = n_used < ASTROMETRY_MIN_STARS
                ax.errorbar([row["dra"]], [row["ddec"]], xerr=[_finite(row["dra_error"])], yerr=[_finite(row["ddec_error"])],
                            ls="none", marker=symbol, ms=7, mfc="white" if few else color, mec=qa.MAIN_COLOR, mew=0.9,
                            ecolor=qa.MAIN_COLOR,
                            elinewidth=qa.LW["main"], capsize=2, zorder=5)
                note.append(f"{names[against]}: {_offset_text(row)}; {count}")
            else:
                note.append(f"{names[against]}: no match")
        if np.isfinite(scale):
            ax.add_patch(Circle((0.0, 0.0), 0.1 * scale, fill=False, ls=":", ec=qa.REF_COLOR, lw=qa.LW["reference"], zorder=2))
            ax.plot([], [], ls=":", color=qa.REF_COLOR, lw=qa.LW["reference"], label="a tenth of the image's pixel")
        qa.refline(ax, x=0.0)
        qa.refline(ax, y=0.0)
        ax.set_xlim(lim, -lim)                 # east to the left, as on the sky
        ax.set_ylim(-lim, lim * (1.0 + 2.0 * strip))
        ax.set_aspect("equal", adjustable="box")
        ticks = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(-lim, lim)
        ax.set_yticks(ticks[np.abs(ticks) <= lim])
        ax.set_ylim(-lim, lim * (1.0 + 2.0 * strip))
        if last_row:
            ax.set_xlabel(DRA_MAS_LABEL)
        if first_column:
            ax.set_ylabel(DDEC_MAS_LABEL)
        qa.panel_note(ax, "\n".join(note), loc="upper left")

    @staticmethod
    def _wave_panels(ax_ra: Any, ax_dec: Any, images: Any, summary: Any) -> None:
        from matplotlib.ticker import NullLocator

        present = [a for a in ASTROMETRY_STYLE if (summary["against"] == a).any()]
        for k, against in enumerate(present):
            color, symbol = ASTROMETRY_STYLE[against]
            rows = summary[(summary["against"] == against) & np.isfinite(summary["pivot"]) & np.isfinite(summary["dra"])]
            if len(rows) == 0:
                continue
            dodge = ASTROMETRY_DODGE ** (k - 0.5 * (len(present) - 1))
            wave = np.asarray(rows["pivot"], float) * dodge
            few = np.asarray(rows["n_used"], int) < ASTROMETRY_MIN_STARS
            for ax, key in ((ax_ra, "dra"), (ax_dec, "ddec")):
                err = np.asarray(rows[f"{key}_error"], float)
                for sel, face in ((~few, color), (few, "white")):     # hollow: a median of one or two stars
                    if sel.any():
                        ax.errorbar(wave[sel], np.asarray(rows[key], float)[sel],
                                    yerr=np.where(np.isfinite(err), err, 0.0)[sel], ls="none", marker=symbol, ms=6,
                                    mfc=face, mec=qa.MAIN_COLOR, mew=0.9, ecolor=qa.MAIN_COLOR,
                                    elinewidth=qa.LW["main"], capsize=2, zorder=4)
        # the coordinate panels' range from the medians of 3 or more stars (others may fall outside: notes)
        solid = summary[np.isfinite(summary["pivot"]) & np.isfinite(summary["dra"])
                        & (np.asarray(summary["n_used"], int) >= ASTROMETRY_MIN_STARS)]
        for ax, key in ((ax_ra, "dra"), (ax_dec, "ddec")):
            if len(solid):
                v = np.asarray(solid[key], float)
                e = np.nan_to_num(np.asarray(solid[f"{key}_error"], float), nan=0.0)
                lo, hi = min(float(np.min(v - e)), 0.0), max(float(np.max(v + e)), 0.0)
                pad = 0.15 * max(hi - lo, 1.0)
                ax.set_ylim(lo - pad, hi + pad)
        ref = images[np.asarray(images["reference"], bool)]
        pivots = np.asarray(images["pivot"], float)
        finite = np.isfinite(pivots)
        for ax in (ax_ra, ax_dec):
            qa.refline(ax, y=0.0)
            if len(ref) and np.isfinite(ref["pivot"][0]):
                qa.refline(ax, x=float(ref["pivot"][0]), ls="--", label=f"band of the reference image ({ref['band'][0]})"
                           if ax is ax_ra else None)
        if finite.any():
            lo, hi = float(pivots[finite].min()), float(pivots[finite].max())
            pad = max(1.15, (2.0 * lo / hi) ** 0.5 if hi > 0 else 1.15)
            ax_ra.set_xlim(lo / pad, hi * pad)
        for ax in (ax_ra, ax_dec):
            qa.set_wave_scale(ax, "log")
        ax_ra.tick_params(labelbottom=False)
        ax_ra.set_ylabel(DRA_MAS_LABEL)
        ax_dec.set_ylabel(DDEC_MAS_LABEL)
        ax_dec.set_xlabel(qa.WAVE_LABEL)
        if finite.any():
            top = ax_ra.secondary_xaxis("top")
            top.set_xticks(pivots[finite], labels=[str(b) for b in images["band"][finite]])
            top.xaxis.set_minor_locator(NullLocator())
            top.tick_params(axis="x", which="major", length=0, rotation=90, pad=2)

    @staticmethod
    def _notes(path: Path, hdr: Any, images: Any, details: bool) -> list[str]:
        """The product, the reference image and the Gaia query; with ``details`` also the cuts."""
        ref = images[np.asarray(images["reference"], bool)]
        ref_text = (f"reference image {ref['band'][0]} ({int(ref['n_point'][0])} point sources)" if len(ref)
                    else "no reference image")
        status = str(hdr.get("JWFASGAI", "")).strip()
        if status in ("ok", "cached"):
            queried = str(hdr.get("JWFASGQT", "")).split("T")[0]
            gaia = f"Gaia DR3 ({hdr.get('JWFASGSV', '?')}, {queried}): {int(hdr.get('JWFASGN', 0))} sources"
            if details and hdr.get("JWFASGPM", True):
                gaia += ", in the medians those with a proper motion"
        elif status == "failed":
            gaia = "Gaia DR3: the query failed, the images are compared with each other only"
        elif status == "empty":
            gaia = "Gaia DR3: no source in the field"
        else:
            gaia = "Gaia DR3: not compared"
        notes = [path.name, f"offsets image $-$ reference; {ref_text}", gaia]
        if details:
            notes.append(rf"point sources: S/N $\geq$ {hdr.get('JWFASSNR', np.nan):g}, isolated by "
                         f"{hdr.get('JWFASISO', np.nan):g} FWHM, data within {hdr.get('JWFASCOR', np.nan):g} FWHM; matched "
                         f"within {hdr.get('JWFASMRD', np.nan):g} FWHM; clipped at {hdr.get('JWFASCLP', np.nan):g} scatters")
        return notes


def _signed(value: float, fmt: str = "+.1f") -> str:
    """A number with a real minus sign (rule 17)."""
    return f"{value:{fmt}}".replace("-", "\N{MINUS SIGN}")


def _offset_text(row: Any) -> str:
    """``+20.5 ± 1.6, −10.0 ± 1.3`` (the median offsets and their errors [mas]; without errors when there are none)."""
    parts = []
    for key in ("dra", "ddec"):
        err = float(row[f"{key}_error"])
        parts.append(f"{_signed(float(row[key]))} $\\pm$ {err:.1f}" if np.isfinite(err) else _signed(float(row[key])))
    return ", ".join(parts)


def _finite(value: Any) -> float:
    """A value, or 0 where it is not finite (an error bar that cannot be drawn)."""
    value = float(value)
    return value if np.isfinite(value) else 0.0
