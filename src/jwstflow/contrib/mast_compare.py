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
  onto jwstflow's wavelengths), DIFF, RATIO, and a PNG with both spectra and the ratio.

Every output records both provenances: CAL_VER/CRDS_CTX of the jwstflow product
(``JWFCAL``, ``JWFCTX``) and of the archive product (``MASTCAL``, ``MASTCTX``,
``MASTFILE``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..steps.base import RunContext, Step

log = logging.getLogger(__name__)


class MastCompare(Step):
    """Difference a jwstflow s3d/x1d product against the MAST archive product of the same name."""

    level = "qa"

    def run(self, inputs: list[Path], ctx: RunContext, *, reference_dir: str | None = None,
            plot: bool = True, **params: Any) -> list[Path]:
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
        log.warning("%s: only _s3d and _x1d products are compared", ours.name)
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
        png = _plot(out.with_suffix(".png"), wa, fa, fb_on_a, ratio, ours.name, prov)
        if png:
            outputs.append(png)
    return outputs


def _wavelengths(hdr: Any) -> np.ndarray:
    n = int(hdr.get("NAXIS3", 0))
    return float(hdr.get("CRVAL3", 0)) + (np.arange(n) + 1 - float(hdr.get("CRPIX3", 1))) * float(hdr.get("CDELT3", 1))


def _plot(path: Path, w, fa, fb, ratio, title: str, prov: dict[str, Any]) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(w, fb, lw=0.7, alpha=0.7, label=f"MAST (jwst {prov['MASTCAL'][0]}, {prov['MASTCTX'][0]})")
    ax1.plot(w, fa, lw=0.7, alpha=0.7, label=f"jwstflow (jwst {prov['JWFCAL'][0]}, {prov['JWFCTX'][0]})")
    ax1.set_ylabel("flux [Jy]"); ax1.legend(fontsize=8); ax1.set_title(title)
    ax2.plot(w, ratio, lw=0.7, color="k"); ax2.axhline(1, color="grey", lw=0.5)
    ax2.set(xlabel="wavelength [um]", ylabel="jwstflow / MAST", ylim=(0.5, 1.5))
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return path
