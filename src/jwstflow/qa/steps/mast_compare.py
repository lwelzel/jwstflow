"""Compare jwstflow products with the archive's own reductions of the same observations.

Opt-in through ``download.reference_products: true`` (which fetches MAST's
products into ``<target>/mast_reference/<run>/``, mirroring the run's
``stage/step`` layout) plus a ``mast_compare`` stage fed with the jwstflow
products. A product is matched with the reference file at the same relative
path (``stage3/calwebb_spec3/<same DMS name>``); jwstflow reproduces the DMS
names, so ``jw01751-o006_t010_nirspec_g235h-f170lp_s3d.fits`` finds its twin.

Outputs (``qa/mast_compare/``):

* ``<name>_s3ddiff.fits``: DIFF = jwstflow - MAST and RATIO cubes (when the two
  cubes share the same grid), plus a SUMMARY table of per-plane statistics;
* ``<name>_x1ddiff.fits``: table WAVELENGTH, FLUX_JWSTFLOW, FLUX_MAST (interpolated
  onto jwstflow's wavelengths), DIFF, RATIO, and a PNG with both spectra and the ratio;
* ``<name>_i2ddiff.fits``: around the target, jwstflow's mosaic, MAST's interpolated onto its
  grid, DIFF = jwstflow - MAST, with the flux ratio, the sky offset and the shift between the two
  mosaics in the header, and a PNG with the three frames.

Every output records both provenances: CAL_VER/CRDS_CTX of the jwstflow product
(``JWFCAL``, ``JWFCTX``) and of the archive product (``MASTCAL``, ``MASTCTX``,
``MASTFILE``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ...steps import RunContext, Step

log = logging.getLogger(__name__)


class MastCompare(Step):
    """Difference a jwstflow s3d/x1d product against the MAST archive product of the same name.

    The reference is MAST's own reduction of the same observations, fetched by
    ``download.reference_products: true`` into ``<target>/mast_reference/<run>/``
    (mirroring the run's layout). Each input is matched with the reference file
    at the same relative path -- jwstflow reproduces the DMS names, so
    ``..._s3d.fits`` finds its archive twin -- falling back to a search by
    name. Then, per product type:

    * ``_s3d``: DIFF = jwstflow - MAST and RATIO cubes (when both share a
      grid) plus a SUMMARY table of per-plane statistics -> ``*_s3ddiff.fits``;
    * ``_x1d``: a table WAVELENGTH / FLUX_JWSTFLOW / FLUX_MAST (interpolated
      onto jwstflow's wavelengths) / DIFF / RATIO -> ``*_x1ddiff.fits``, and a
      figure with both spectra and their ratio;
    * ``_i2d``: a box of ``size`` arcsec around the target on jwstflow's
      mosaic, MAST's mosaic interpolated onto it (bilinear: the two are never
      on one grid -- their alignments differ), DIFF = jwstflow - MAST ->
      ``*_i2ddiff.fits``; the header and the figure (both images under one
      stretch, the difference on a signed scale) give the median ratio over
      pixels jwstflow holds at S/N > 10 and the ratio of the sums over the
      pixels at S/N > 3 (both of the light above each image's own sky), the
      difference of the skies (medians over the pixels within jwstflow's sky
      scatter of its clipped sky level) and the shift between
      the mosaics (cross-correlation peak, pixels and arcsec).

    Every output records both provenances (CAL_VER/CRDS context of each side).
    Inputs without a reference are skipped with a warning.
    """

    level = "qa"
    version = "4"   # 3 -> 4: resampled images (_i2d); 2 -> 3: logarithmic wavelength axis on the spectrum figure
    needs_reference_products = True   # the run waits for the background download of the MAST products

    def run(self, inputs: list[Path], ctx: RunContext, *, reference_dir: str | None = None,
            plot: bool = True, size: float = 12.0, **params: Any) -> list[Path]:
        (ours,) = inputs
        reference = Path(reference_dir).expanduser() if reference_dir else ctx.reference_dir
        if reference is None:
            reference = (ctx.target_dir or ctx.root.parent) / "mast_reference" / ctx.run_name.rpartition("/")[2]
        theirs = find_reference(ours, ctx.root, reference)
        if theirs is None:
            log.warning("%s: no MAST reference product under %s (download.reference_products on? names DMS-compliant?)",
                        ours.name, reference)
            return []
        if ours.name.endswith("_s3d.fits"):
            return compare_cubes(ours, theirs, ctx.output_dir)
        if ours.name.endswith("_x1d.fits"):
            return compare_spectra(ours, theirs, ctx.output_dir, plot=plot)
        if ours.name.endswith("_i2d.fits"):
            return compare_images(ours, theirs, ctx.output_dir, size=size, plot=plot, target=ctx.target_coords)
        log.warning("%s: only _s3d, _x1d and _i2d products are compared", ours.name)
        return []


def find_reference(ours: Path, run_dir: Path, reference: Path) -> Path | None:
    """The MAST twin of a jwstflow product: same relative path under the reference tree
    (variant suffixes such as ``calwebb_spec3-pass1`` map to ``calwebb_spec3``), else a search by name."""
    try:
        rel = ours.parent.resolve().relative_to(Path(run_dir).resolve())
    except ValueError:
        rel = None
    candidates = []
    if rel is not None:
        parts = list(rel.parts)
        if parts:
            parts[-1] = parts[-1].split("-", 1)[0]  # drop a stage variant
        candidates.append(reference / Path(*parts) / ours.name)
    for c in candidates:
        if c.exists():
            return c
    hits = sorted(reference.rglob(ours.name)) if reference.exists() else []
    return hits[0] if hits else None


def _provenance(hdr_ours: Any, hdr_theirs: Any, theirs: Path) -> dict[str, Any]:
    return {
        "JWFCAL": (str(hdr_ours.get("CAL_VER", "")), "jwst version of the jwstflow product"),
        "JWFCTX": (str(hdr_ours.get("CRDS_CTX", "")), "CRDS context of the jwstflow product"),
        "MASTFILE": (theirs.name, "MAST archive product compared against"),
        "MASTCAL": (str(hdr_theirs.get("CAL_VER", "")), "jwst version of the MAST product"),
        "MASTCTX": (str(hdr_theirs.get("CRDS_CTX", "")), "CRDS context of the MAST product"),
        "JWFCOMP": ("jwstflow - MAST", "sign convention of DIFF; RATIO = jwstflow / MAST"),
    }


def compare_cubes(ours: Path, theirs: Path, out_dir: Path) -> list[Path]:
    from astropy.io import fits

    with fits.open(ours) as a, fits.open(theirs) as b:
        da, db = np.asarray(a["SCI"].data, float), np.asarray(b["SCI"].data, float)
        ha, hb = a["SCI"].header, b["SCI"].header
        prov = _provenance(a[0].header, b[0].header, theirs)
        same_grid = da.shape == db.shape and all(
            np.isclose(float(ha.get(k, 0)), float(hb.get(k, 0)), rtol=0, atol=1e-6)
            for k in ("CRVAL1", "CRVAL2", "CRVAL3", "CDELT3", "CRPIX1", "CRPIX2", "CRPIX3")
        )
        hdus: list[Any] = [fits.PrimaryHDU()]
        for k, v in prov.items():
            hdus[0].header[k] = v
        hdus[0].header["SAMEGRID"] = (same_grid, "cubes share WCS and shape")
        if same_grid:
            with np.errstate(all="ignore"):
                diff = da - db
                ratio = da / db
            hdus.append(fits.ImageHDU(diff.astype("f4"), header=ha, name="DIFF"))
            hdus.append(fits.ImageHDU(ratio.astype("f4"), header=ha, name="RATIO"))
        # per-plane summary (works whether or not the grids match)
        with np.errstate(all="ignore"):
            sa, sb = np.nansum(da, axis=(1, 2)), np.nansum(db, axis=(1, 2))
            med_a, med_b = np.nanmedian(da, axis=(1, 2)), np.nanmedian(db, axis=(1, 2))
        n = min(len(sa), len(sb))
        wave = _wavelengths(ha)[:n]
        cols = [fits.Column("WAVELENGTH", "D", "um", array=wave),
                fits.Column("SUM_JWSTFLOW", "D", "MJy/sr", array=sa[:n]),
                fits.Column("SUM_MAST", "D", "MJy/sr", array=sb[:n]),
                fits.Column("MEDIAN_JWSTFLOW", "D", "MJy/sr", array=med_a[:n]),
                fits.Column("MEDIAN_MAST", "D", "MJy/sr", array=med_b[:n])]
        hdus.append(fits.BinTableHDU.from_columns(cols, name="SUMMARY"))
        out = out_dir / ours.name.replace("_s3d.fits", "_s3ddiff.fits")
        fits.HDUList(hdus).writeto(out, overwrite=True)
    with np.errstate(all="ignore"):
        rel = np.nanmedian(sa[:n] / sb[:n])
    log.info("%s vs MAST: same grid=%s, median(sum ratio)=%.4f", ours.name, same_grid, rel)
    return [out]


def compare_spectra(ours: Path, theirs: Path, out_dir: Path, *, plot: bool = True) -> list[Path]:
    from astropy.io import fits

    with fits.open(ours) as a, fits.open(theirs) as b:
        ta, tb = a["EXTRACT1D"].data, b["EXTRACT1D"].data
        wa, fa = np.asarray(ta["WAVELENGTH"], float), np.asarray(ta["FLUX"], float)
        wb, fb = np.asarray(tb["WAVELENGTH"], float), np.asarray(tb["FLUX"], float)
        prov = _provenance(a[0].header, b[0].header, theirs)
    good = np.isfinite(wb) & np.isfinite(fb)
    fb_on_a = np.interp(wa, wb[good], fb[good], left=np.nan, right=np.nan)
    with np.errstate(all="ignore"):
        diff, ratio = fa - fb_on_a, fa / fb_on_a
    cols = [fits.Column("WAVELENGTH", "D", "um", array=wa), fits.Column("FLUX_JWSTFLOW", "D", "Jy", array=fa),
            fits.Column("FLUX_MAST", "D", "Jy", array=fb_on_a), fits.Column("DIFF", "D", "Jy", array=diff),
            fits.Column("RATIO", "D", "", array=ratio)]
    primary = fits.PrimaryHDU()
    for k, v in prov.items():
        primary.header[k] = v
    primary.header["MEDRATIO"] = (float(np.nanmedian(ratio)), "median jwstflow/MAST flux ratio")
    out = out_dir / ours.name.replace("_x1d.fits", "_x1ddiff.fits")
    fits.HDUList([primary, fits.BinTableHDU.from_columns(cols, name="COMPARISON")]).writeto(out, overwrite=True)
    log.info("%s vs MAST: median flux ratio %.4f", ours.name, float(np.nanmedian(ratio)))
    outputs = [out]
    if plot:
        outputs.append(_plot(out.with_suffix(".png"), wa, fa, fb_on_a, ratio, out.name, prov))
    return outputs


def compare_images(ours: Path, theirs: Path, out_dir: Path, *, size: float = 12.0, plot: bool = True,
                   target: dict[str, Any] | None = None) -> list[Path]:
    """A resampled image against MAST's: around the target, MAST's mosaic on jwstflow's grid."""
    from astropy.io import fits

    from ...cubes import resample_image
    from ...images import read_image

    a, b = read_image(ours), read_image(theirs)
    if a is None or b is None or a.wcs is None or b.wcs is None:
        log.warning("%s: no comparable images", ours.name)
        return []
    ra, dec = (target or {}).get("ra"), (target or {}).get("dec")
    if ra is None or dec is None:
        ra, dec = float(a.primary.get("TARG_RA")), float(a.primary.get("TARG_DEC"))
    cut = a.cutout(float(ra), float(dec), size)
    mast = resample_image(b.data, b.wcs, cut.wcs, cut.shape)
    from .images import sky_statistics

    with np.errstate(all="ignore"):
        diff = cut.data - mast
        both = np.isfinite(cut.data) & np.isfinite(mast)
        # the sky of each, over the pixels where jwstflow's image holds nothing (within its sky scatter
        # of its sky level): the ratio is of the light above each sky, the skies' difference on its own
        level, scatter, _ = sky_statistics(np.where(both, cut.data, np.nan))
        faint = both & (np.abs(cut.data - level) < scatter)
        enough = faint.sum() >= 50
        sky_a = float(np.median(cut.data[faint])) if enough else 0.0
        sky_b = float(np.median(mast[faint])) if enough else 0.0
        sky = sky_a - sky_b if enough else float("nan")
        snr = (cut.data - sky_a) / cut.error if cut.error is not None else np.full(cut.shape, np.nan)
        strong = both & (snr > 10.0)
        ratio = float(np.median((cut.data[strong] - sky_a) / (mast[strong] - sky_b))) if strong.sum() >= 10 \
            else float("nan")
        bright = both & (snr > 3.0)
        sum_ratio = float((cut.data[bright] - sky_a).sum() / (mast[bright] - sky_b).sum()) if bright.sum() >= 10 \
            else float("nan")
    dy, dx = _shift(cut.data, mast, both)
    scale = cut.pixscale_arcsec or float("nan")
    prov = _provenance(a.primary, b.primary, theirs)
    prim = fits.PrimaryHDU()
    for k, v in {**prov,
                 "JWFRATIO": (_or_none(ratio), "jwstflow/MAST above sky, median, S/N>10"),
                 "JWFSUMRT": (_or_none(sum_ratio), "jwstflow/MAST above sky, sums, S/N>3"),
                 "JWFSKYDF": (_or_none(sky), "sky jwstflow - MAST [MJy/sr]"),
                 "JWFSHFTX": (_or_none(dx), "[pix] shift of jwstflow against MAST, x"),
                 "JWFSHFTY": (_or_none(dy), "[pix] shift of jwstflow against MAST, y"),
                 "JWFCMPSZ": (float(size), "[arcsec] side of the compared box"),
                 "JWFINTRP": ("bilinear", "MAST interpolated onto our grid")}.items():
        if v[0] is not None:
            prim.header[k] = v
    header = cut.header.copy()
    hdus = [prim, fits.ImageHDU(cut.data.astype("f4"), header=header, name="JWSTFLOW"),
            fits.ImageHDU(mast.astype("f4"), header=header, name="MAST"),
            fits.ImageHDU(diff.astype("f4"), header=header, name="DIFF")]
    out = out_dir / ours.name.replace("_i2d.fits", "_i2ddiff.fits")
    fits.HDUList(hdus).writeto(out, overwrite=True)
    log.info("%s vs MAST: median ratio %.4f (S/N > 10), sum ratio %.4f (S/N > 3), sky difference %.3g, shift "
             "(%.2f, %.2f) pix", ours.name, ratio, sum_ratio, sky, dx, dy)
    outputs = [out]
    if plot:
        outputs.append(_plot_images(out.with_suffix(".png"), cut, mast, diff, ours.name, theirs.name, prov,
                                    ratio=ratio, sum_ratio=sum_ratio, sky=sky, shift=(dx, dy), scale=scale))
    return outputs


def _or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _shift(a: np.ndarray, b: np.ndarray, both: np.ndarray) -> tuple[float, float]:
    """(dy, dx) [pix] by which ``a`` is displaced against ``b``: the peak of their cross-correlation
    (over the pixels both hold, each less its median), refined by a parabola through its neighbours."""
    if both.sum() < 50:
        return float("nan"), float("nan")
    fa = np.where(both, a - np.median(a[both]), 0.0)
    fb = np.where(both, b - np.median(b[both]), 0.0)
    corr = np.fft.ifft2(np.fft.fft2(fa) * np.conj(np.fft.fft2(fb))).real
    corr = np.fft.fftshift(corr)
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    def vertex(c_m: float, c_0: float, c_p: float) -> float:
        den = c_m - 2 * c_0 + c_p
        return 0.0 if den == 0 else 0.5 * (c_m - c_p) / den
    ny, nx = corr.shape
    sub_y = vertex(corr[py - 1, px], corr[py, px], corr[(py + 1) % ny, px]) if 0 < py else 0.0
    sub_x = vertex(corr[py, px - 1], corr[py, px], corr[py, (px + 1) % nx]) if 0 < px else 0.0
    return float(py - ny // 2 + sub_y), float(px - nx // 2 + sub_x)


def _plot_images(path: Path, cut: Any, mast: np.ndarray, diff: np.ndarray, name: str, theirs: str,
                 prov: dict[str, Any], *, ratio: float, sum_ratio: float, sky: float, shift: tuple[float, float],
                 scale: float) -> Path:
    from ... import qa

    sky_frame = qa.SkyFrame.from_header(cut.header, cut.shape, primary=cut.primary)
    fig, axes = qa.subplots(1, 3, figsize=(15.0, 4.6))
    shown = [qa.to_mjy(v, cut.unit)[0] for v in (cut.data, mast)]        # the norm acts on what is drawn
    norm = qa.stretch_norm(np.concatenate([v[np.isfinite(v)] for v in shown]), "asinh", (1.0, 99.5))
    qa.Frame(axes[0]).image(cut.data, unit=cut.unit, sky=sky_frame, norm=norm, sides="lbt", cbar=False)
    qa.tag(axes[0], "jwstflow")
    qa.Frame(axes[1]).image(mast, unit=cut.unit, sky=sky_frame, norm=norm, sides="bt")
    qa.tag(axes[1], "MAST, on jwstflow's grid")
    qa.Frame(axes[2]).image(diff, unit=cut.unit, sky=sky_frame, center=0.0, sides="btr")
    qa.tag(axes[2], "jwstflow $-$ MAST")
    qa.annotate(axes[0], name, f"MAST: {theirs} ({prov['MASTCAL'][0]}, {prov['MASTCTX'][0]}); jwstflow: "
                f"{prov['JWFCAL'][0]}, {prov['JWFCTX'][0]}",
                f"jwstflow / MAST above each sky: median {ratio:.4f} over S/N > 10, {sum_ratio:.4f} summed over "
                f"S/N > 3; sky difference {sky:.3g} {cut.unit or ''}",
                f"shift of jwstflow against MAST ({shift[1]:+.2f}, {shift[0]:+.2f}) pix = "
                f"({shift[1] * scale * 1e3:+.0f}, {shift[0] * scale * 1e3:+.0f}) mas in x, y; MAST interpolated "
                "(bilinear) onto jwstflow's grid", sky_frame.note() if sky_frame is not None else "")
    qa.figlegend(fig)
    return qa.save(fig, path)


def _wavelengths(hdr: Any) -> np.ndarray:
    n = int(hdr.get("NAXIS3", 0))
    return float(hdr.get("CRVAL3", 0)) + (np.arange(n) + 1 - float(hdr.get("CRPIX3", 1))) * float(hdr.get("CDELT3", 1))


def _plot(path: Path, w, fa, fb, ratio, name: str, prov: dict[str, Any]) -> Path:
    from ... import qa

    fig, (ax1, ax2) = qa.subplots(2, 1, figsize=qa.FIGSIZE["spectrum2"], sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    # both spectra are in Jy; jwstflow's is the main product -> black, MAST the reference -> the model accent
    qa.step(ax1, w, fb * 1e3, color=qa.ROLE_COLORS["model"], lw=qa.LW["thin"],
               label=f"MAST (jwst {prov['MASTCAL'][0]}, {prov['MASTCTX'][0]})")
    qa.step(ax1, w, fa * 1e3, color=qa.MAIN_COLOR, lw=qa.LW["thin"],
               label=f"jwstflow (jwst {prov['JWFCAL'][0]}, {prov['JWFCTX'][0]})")
    ax1.set_ylabel(qa.FLUX_LABEL)
    qa.step(ax2, w, ratio, color=qa.MAIN_COLOR, lw=qa.LW["thin"])
    qa.refline(ax2, y=1.0)
    ax2.set(xlabel=qa.WAVE_LABEL, ylabel=r"$F_\nu$ / $F_{\nu,\,\mathrm{MAST}}$", ylim=(0.5, 1.5))
    qa.set_wave_scale(ax2)   # shared x: sets both panels
    qa.annotate(ax1, name)
    qa.figlegend(fig)
    return qa.save(fig, path, dpi=150)
