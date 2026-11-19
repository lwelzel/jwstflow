"""The cluster steps: ``flag_spaxel_clusters`` (the cubes) and ``propagate_cluster_flags`` (the cal files).

The chain they belong to is described in :mod:`jwstflow.cubes.clusters`; the figure of it,
``qa_spaxel_clusters``, is in :mod:`jwstflow.qa.steps.clusters`.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from ...steps import RunContext, Step, StepParams
from ...steps.naming import derived_name
from ..detector import dq_value
from ..morphology import dilate
from ..wcs import celestial_wcs
from .compare import deviation_from_others, robust_sigma
from .detector import detector_pixels_in_masks
from .product import CLUSTERMASK_SUFFIX, find_cluster_products, write_cluster_product
from .regions import (
    MODE_ALL,
    MODE_AUTO,
    MODE_EXPLICIT,
    ClusterRegion,
    band_key,
    dither_of,
    matches_where,
    planes_in_window,
    region_aperture,
    region_centre,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- step 1: the cubes
class FlagSpaxelClusters(Step):
    """Find bad spaxel clusters in per-dither cubes: which dithers, which spaxels.

    One task receives every per-dither cube (``batch = "all"``). For each
    region of ``regions`` (sky position + search aperture + wavelength
    window, see the ``regions`` grammar below) and each cube covering it, the
    aperture's spaxels on the affected planes are compared with the *other*
    dithers at the same wavelengths, resampled onto this cube's sky grid:
    the deviation kept per spaxel is the one with the smallest magnitude over
    the others, so a real feature -- present in every dither -- never
    deviates, while a cluster confined to one dither deviates from all of
    them. Spaxels beyond ``sigma`` times the robust (MAD) noise of that
    deviation image are *deviant*; with ``dithers: auto`` a dither is
    included when it holds at least ``min_spaxels`` deviant spaxel-planes
    (with two dithers the comparison is symmetric -- name the dither
    explicitly). Included dithers get the deviant spaxels flagged (``refine``,
    the cluster's actual, non-circular shape), grown by ``grow`` spaxels;
    ``refine: false``, or an explicitly named dither without any deviant
    spaxel, flags the whole aperture. Every cube gets one
    ``*_clustermask.fits`` product (the jwstflow mask contract plus the
    per-region REGIONS table and QA slices of every dither, included or not).

    ``regions`` entries::

        - id: g395h-4.570                 # optional label
          ra: "11:11:10.93"               # degrees or sexagesimal (ICRS)
          dec: "-76:41:57.7"
          radius: 0.38                    # arcsec; shape: circle (default) | ellipse (radius_minor,
                                          # angle) | polygon (vertices: [[ra, dec], ...])
          wave_min: 4.5694                # um: planes whose bin overlaps the window
          wave_max: 4.5708
          where: {GRATING: G395H}         # optional header match
          dithers: auto                   # auto | all | [3]
          refine: true
    """

    name = "flag_spaxel_clusters"
    level = 4
    batch = "all"
    inputs = ("*_s3d.fits",)
    outputs = (CLUSTERMASK_SUFFIX,)
    version = "1"

    class Params(StepParams):
        regions: list[ClusterRegion] = Field(description="the clusters to look for (see the step description)")
        sigma: float = Field(5.0, gt=0, description="deviation threshold in robust-noise units")
        min_spaxels: int = Field(3, ge=1, description="deviant spaxel-planes needed to include a dither (auto)")
        grow: int = Field(1, ge=0, description="dilate the flagged spaxels by this many spaxels")
        plane_pad: int = Field(0, ge=0, description="extra planes flagged on each side of the window")
        qa_pad: int = Field(2, ge=0, description="context planes stored on each side of the window for QA")

    def run(self, inputs: list[Path], ctx: RunContext, *, regions: list[dict[str, Any]], sigma: float = 5.0,
            min_spaxels: int = 3, grow: int = 1, plane_pad: int = 0, qa_pad: int = 2, **params: Any) -> list[Path]:
        import warnings

        from astropy.io import fits
        from stdatamodels.jwst import datamodels

        from ..wcs import cube_wavelengths

        if not inputs:
            raise ValueError("flag_spaxel_clusters received no cubes")
        # validated parameters arrive without their defaults (exclude_unset): fill them in
        regions = [ClusterRegion.model_validate(r if isinstance(r, dict) else r.model_dump()).model_dump()
                   for r in regions]
        for i, r in enumerate(regions, start=1):
            r["id"] = r["id"] or f"region{i}"
        cubes: list[dict[str, Any]] = []
        for path in sorted(inputs):
            with datamodels.open(path) as cube:
                cubes.append({"path": Path(path), "data": np.asarray(cube.data, dtype=float),
                              "wave": cube_wavelengths(cube), "wcs": celestial_wcs(cube),
                              "header": fits.getheader(path)})
            c = cubes[-1]
            c["dither"] = dither_of(c["header"], c["path"])
            c["band"] = band_key(c["header"])
            c["mask"] = np.zeros(c["data"].shape, dtype=bool)
            c["rows"], c["slices"] = [], []
        by_band: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for c in cubes:
            by_band.setdefault(c["band"], []).append(c)
        log.info("flag_spaxel_clusters: %d cube(s) in %d band(s), %d region(s)", len(cubes), len(by_band), len(regions))

        for region in regions:
            rid = region["id"]
            n_hit = 0
            for band, members in by_band.items():
                covering = [c for c in members if matches_where(c["header"], region.get("where"))
                            and c["wave"].min() <= region["wave_max"] and c["wave"].max() >= region["wave_min"]]
                if not covering:
                    continue
                n_hit += len(covering)
                dithers = sorted({c["dither"] for c in covering})
                if len(dithers) < len(covering):
                    log.warning("region %s: several cubes share a dither number in band %s; "
                                "each is compared with every other cube", rid, band)
                if len(covering) == 2 and region["dithers"] == "auto":
                    log.warning("region %s: only two dithers cover it -- the comparison is symmetric, "
                                "both may be included; give `dithers:` explicitly", rid)
                for c in covering:
                    others = [o for o in covering if o is not c]
                    ny, nx = c["data"].shape[1:]
                    aperture = region_aperture(region, c["wcs"], (ny, nx))
                    planes = planes_in_window(c["wave"], region["wave_min"], region["wave_max"], plane_pad)
                    qa_planes = planes_in_window(c["wave"], region["wave_min"], region["wave_max"], plane_pad + qa_pad)
                    if not aperture.any():
                        log.warning("region %s: aperture falls outside %s", rid, c["path"].name)
                    n_others_max, score, noises = 0, 0.0, []
                    deviant = np.zeros((len(planes), ny, nx), dtype=bool)
                    qa = {"data": [], "ref": [], "sigma": [], "wave": []}
                    for k in qa_planes:
                        plane = c["data"][k]
                        dev, ref, n_others = deviation_from_others(plane, float(c["wave"][k]), c["wcs"], others)
                        n_others_max = max(n_others_max, n_others)
                        with warnings.catch_warnings(), np.errstate(all="ignore"):
                            warnings.filterwarnings("ignore", message=".*All-NaN.*")
                            floor = 1e-6 * float(np.nanmax(np.abs(ref))) if np.isfinite(ref).any() else 0.0
                            noise = max(robust_sigma(dev[~aperture]), floor, 1e-12)
                            sig = np.abs(dev) / noise
                        if k in planes:
                            j = int(np.flatnonzero(planes == k)[0])
                            deviant[j] = aperture & np.isfinite(sig) & (sig > sigma)
                            if aperture.any() and np.isfinite(sig[aperture]).any():
                                score = max(score, float(np.nanmax(sig[aperture])))
                            noises.append(noise)
                        qa["data"].append(plane)
                        qa["ref"].append(ref)
                        qa["sigma"].append(sig)
                        qa["wave"].append(float(c["wave"][k]))
                    n_deviant = int(deviant.sum())
                    mode = region["dithers"] if isinstance(region["dithers"], str) else MODE_EXPLICIT
                    if mode == MODE_AUTO:
                        included = n_deviant >= min_spaxels
                        reason = f"{n_deviant} deviant spaxel-plane(s) {'>=' if included else '<'} {min_spaxels}"
                    elif mode == MODE_ALL:
                        included, reason = True, "dithers: all"
                    else:
                        included = c["dither"] in [int(d) for d in region["dithers"]]
                        reason = f"dither {c['dither']} {'listed' if included else 'not listed'}"
                    if n_others_max == 0 and included and region["refine"]:
                        log.warning("region %s: %s has no other dither to compare with; flagging the whole aperture",
                                    rid, c["path"].name)
                    refined = bool(region["refine"] and n_deviant > 0)
                    n_flagged = 0
                    if included and aperture.any():
                        for j, k in enumerate(planes):
                            flagged = deviant[j] if refined else aperture
                            if grow > 0:
                                flagged = dilate(flagged, grow)
                            c["mask"][k] |= flagged
                            n_flagged += int(flagged.sum())
                        if not refined and region["refine"]:
                            log.warning("region %s: no deviant spaxel in %s; the whole aperture is flagged",
                                        rid, c["path"].name)
                    log.info("region %s on %s (dither %d): %s -- score %.1f sigma, %d deviant, %d flagged",
                             rid, c["path"].name, c["dither"], "included" if included else "excluded",
                             score, n_deviant, n_flagged)
                    c["rows"].append({
                        "id": rid, "ra": region_centre(region)[0], "dec": region_centre(region)[1],
                        "shape": region.get("shape", "circle"), "radius": float(region.get("radius") or 0.0),
                        "radius_minor": float(region.get("radius_minor") or region.get("radius") or 0.0),
                        "angle": float(region.get("angle", 0.0)), "wave_min": float(region["wave_min"]),
                        "wave_max": float(region["wave_max"]), "plane_min": int(planes.min()),
                        "plane_max": int(planes.max()), "n_planes": len(planes), "dither": int(c["dither"]),
                        "mode": mode, "included": bool(included), "refined": refined,
                        "n_aperture": int(aperture.sum()), "n_deviant": n_deviant, "n_flagged": n_flagged,
                        "score": float(score), "noise": float(np.median(noises)) if noises else 0.0,
                        "n_others": int(n_others_max), "reason": reason,
                    })
                    c["slices"].append({"data": np.stack(qa["data"]), "ref": np.stack(qa["ref"]),
                                        "sigma": np.stack(qa["sigma"]), "wave": np.array(qa["wave"]),
                                        "aperture": aperture})
            if n_hit == 0:
                log.warning("region %s: no cube covers %.4f-%.4f um%s", rid, region["wave_min"], region["wave_max"],
                            f" with {region['where']}" if region.get("where") else "")

        out: list[Path] = []
        for c in cubes:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.filterwarnings("ignore", message=".*All-NaN.*")
                contimg = np.nanmedian(c["data"], axis=0)
            included = [r["id"] for r in c["rows"] if r["included"]]
            keys = {"JWFCLNRG": (len(c["rows"]), "flag_spaxel_clusters: regions evaluated"),
                    "JWFCLNPX": (int(c["mask"].sum()), "flag_spaxel_clusters: flagged spaxel-planes"),
                    "JWFCLDIT": (int(c["dither"]), "flag_spaxel_clusters: dither number of the cube"),
                    "JWFCLINC": (",".join(included) or "none", "flag_spaxel_clusters: regions flagged in this cube"),
                    "JWFCLSIG": (float(sigma), "flag_spaxel_clusters: deviation threshold [sigma]"),
                    "JWFCLGRW": (int(grow), "flag_spaxel_clusters: mask growth [spaxels]")}
            path = ctx.output_dir / derived_name(c["path"], CLUSTERMASK_SUFFIX)
            out.append(write_cluster_product(path, c["path"], c["mask"], c["wave"], contimg, c["rows"], c["slices"], keys))
        log.info("flag_spaxel_clusters: %d product(s), %d with flagged spaxels",
                 len(out), sum(1 for c in cubes if c["mask"].any()))
        return out


# --------------------------------------------------------------------------- step 2: the cal files
class PropagateClusterFlags(Step):
    """Back-propagate cluster masks to the detector: DQ-flagged copies of the cal files.

    For every cal file, the ``*_clustermask.fits`` products of the same band
    and dither (``masks_stage``, matched by header) are looked up. When one of
    them holds flagged spaxels, the file's WCS is evaluated (NIRSpec IFU:
    slice by slice) over the columns whose wavelengths reach the flagged
    planes, and every detector pixel whose (RA, Dec, wavelength) lands on a
    flagged spaxel -- the cube mask widened by ``plane_pad`` planes and
    ``grow`` spaxels, so drizzle's spreading is covered -- gets
    ``DO_NOT_USE | OUTLIER`` in DQ; ``cube_build`` then ignores it. Files
    without a matching product, or whose product flags nothing, are copied
    unchanged, so the stage directory always holds the complete set of
    frames. The copies keep the official ``_cal`` name; JWFCLNPX (pixels
    flagged), JWFCLMSK (products applied) and JWFCLRGS (regions) record the
    edit.
    """

    name = "propagate_cluster_flags"
    level = 2
    inputs = ("*_cal.fits",)
    writes_official_products = True   # DQ-edited copies keep the official _cal name; JWFCL* records the edit
    version = "1"

    class Params(StepParams):
        masks_stage: str = Field("flag_spaxel_clusters", description="stage holding the *_clustermask.fits products")
        grow: int = Field(0, ge=0, description="extra spatial widening of the cube masks [spaxels] at lookup")
        plane_pad: int = Field(1, ge=0, description="extra planes on each side of the flagged planes at lookup")

    def run(self, inputs: list[Path], ctx: RunContext, *, masks_stage: str = "flag_spaxel_clusters", grow: int = 0,
            plane_pad: int = 1, **params: Any) -> list[Path]:
        from astropy.io import fits

        (src,) = inputs
        dst = ctx.output_dir / src.name
        try:
            mask_dir = ctx.dir_of(masks_stage)
        except KeyError as exc:
            raise KeyError(f"masks_stage {masks_stage!r} is not a stage of this workflow") from exc
        hdr = fits.getheader(src)
        products = find_cluster_products(hdr, src, mask_dir)
        active = [p for p in products if p["mask"].any()]
        n_flagged, used, regions = 0, [], []
        if not active:
            shutil.copy2(src, dst)
            log.info("%s: no cluster flags for this band/dither (%d product(s) matched); copied", src.name, len(products))
        else:
            from stdatamodels.jwst import datamodels

            with datamodels.open(src) as model:
                pixels, used = detector_pixels_in_masks(model, active, grow=grow, plane_pad=plane_pad)
                n_flagged = int(pixels.sum())
                model.dq[pixels] |= dq_value(("OUTLIER",))
                model.save(str(dst))
            regions = sorted({str(r["id"]) for p in active for r in p["regions"] if bool(r["included"])})
            log.info("%s: %d detector pixel(s) flagged from %d product(s)", src.name, n_flagged, len(used))
        with fits.open(dst, mode="update") as hdul:
            hdul[0].header["JWFCLNPX"] = (n_flagged, "propagate_cluster_flags: detector pixels flagged")
            hdul[0].header["JWFCLMSK"] = (",".join(used) or "none", "propagate_cluster_flags: products applied")
            hdul[0].header["JWFCLRGS"] = (",".join(regions) or "none", "propagate_cluster_flags: regions applied")
        return [dst]
