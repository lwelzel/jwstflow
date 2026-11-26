"""The registration step: ``image_astrometry``.

What it measures, how and in which product is described in :mod:`jwstflow.images.astrometry`;
the figure of its product, ``qa_image_astrometry``, is in :mod:`jwstflow.qa.steps.images`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from ...steps import RunContext, Step, StepParams
from ...steps.naming import derived_name, slug, split_suffix
from .. import read_image
from .catalogs import catalog_path, observation_mjd, read_point_sources
from .gaia import GAIA_NAME, GAIA_SUFFIX, SERVICES, GaiaQueryError, angular_distance, julian_year, load_or_query_gaia, propagate
from .match import Comparison, compare, mean_position
from .product import ASTROM_SUFFIX, write_astrometry_product

log = logging.getLogger(__name__)

#: [deg] Added to the radius of the circle around the images' footprints that Gaia is asked for.
GAIA_MARGIN = 5.0 / 3600.0


class ImageAstrometry(Step):
    """Registration of a run's images: point-source offsets of every image against a reference image
    and against Gaia DR3, per filter, in one ``*_astrom.fits``.

    One task receives every resampled image (``batch = "all"``, ``*_i2d.fits``) and reads the
    source catalogue ``source_catalog`` wrote next to each (``<base>_cat.ecsv``; an image without
    one is listed but not compared). Its point sources are the catalogue's compact sources
    (``is_extended`` false) with ``S/N >= min_snr`` (aperture-corrected total flux over its error),
    no catalogued neighbour within ``isolation`` FWHM and data in every pixel within ``core`` FWHM
    of the centroid (no saturated, blanked core, not cut by the mosaic's edge); the FWHM is the
    band's diffraction limit, so the cuts follow the resolution from 1 to 25 um.

    Every image is matched to the ``reference`` image (a band such as ``F200W`` or a file name;
    by default the image with the most point sources, the bluer on a tie) and, with ``gaia``, to
    Gaia DR3: queried once for the union of the footprints (the ESA archive, else the STScI
    service ``tweakreg`` reads; ``gaia_services``), cached in the stage as ``<base>_gaiadr3.ecsv``
    and reused while it covers the footprint, its positions moved with their proper motions to
    each image's mid-exposure epoch. Matching: the bulk shift from all pairs within ``search``
    arcsec, then mutual nearest neighbours within ``match_radius`` FWHM (of the coarser band) of
    it. Offsets are image minus reference, in right ascension times cos(dec) and in declination
    [mas]; per comparison the median of the matches after clipping at ``clip`` robust scatters
    (with five or more), the robust scatter (1.4826 x MAD), the standard deviation and the error of
    the median (1.2533 x the larger of the two / sqrt(N)), NaN with fewer than ``min_matches``.
    Gaia sources without a proper motion are matched and recorded but, with ``gaia_pm_only``, not
    in the median. A Gaia query that fails is logged and recorded in
    the product (JWFASGAI = failed); the relative comparison is written regardless.

    The product (``<base>_astrom.fits``; ``<base>``: the leading name fields all inputs share,
    else the target) holds IMAGES (band, pivot, FWHM, epoch, the sources left after each cut),
    SUMMARY (one row per comparison), MATCHES (one row per matched source) and, in its primary
    header, the reference, the cuts, the Gaia query and the conventions. ``qa_image_astrometry``
    draws it.
    """

    name = "image_astrometry"
    level = 4
    batch = "all"
    inputs = ("*_i2d.fits",)
    outputs = (ASTROM_SUFFIX, GAIA_SUFFIX)
    version = "1"

    class Params(StepParams):
        reference: str | None = Field(None, description="the reference image: a band (F200W) or a file name; "
                                                        "default the image with the most point sources")
        gaia: bool = Field(True, description="compare every image with Gaia DR3 too")
        gaia_services: list[str] = Field(default_factory=lambda: list(SERVICES),
                                         description="Gaia DR3 services to ask, in order (esa, stsci)")
        gaia_pm_only: bool = Field(True, description="only Gaia sources with a proper motion enter the median")
        min_snr: float = Field(10.0, gt=0, description="S/N of a point source (total flux over its error)")
        isolation: float = Field(3.0, ge=0, description="[FWHM] no catalogued neighbour closer than this")
        core: float = Field(1.0, ge=0, description="[FWHM] radius within which every pixel must have data")
        match_radius: float = Field(1.0, gt=0, description="[FWHM of the coarser band] match radius around the bulk shift")
        search: float = Field(1.0, gt=0, description="[arcsec] radius within which pairs vote for the bulk shift")
        clip: float = Field(3.0, gt=0, description="[robust scatters] clipping of the offsets before the median")
        min_matches: int = Field(3, ge=2, description="matches needed for a scatter and an error of the median")

    def run(self, inputs: list[Path], ctx: RunContext, *, reference: str | None = None, gaia: bool = True,
            gaia_services: list[str] | None = None, gaia_pm_only: bool = True, min_snr: float = 10.0,
            isolation: float = 3.0, core: float = 1.0, match_radius: float = 1.0, search: float = 1.0,
            clip: float = 3.0, min_matches: int = 3, **params: Any) -> list[Path]:
        if not inputs:
            raise ValueError("image_astrometry received no images")
        cuts = {"min_snr": min_snr, "isolation": isolation, "core": core}
        images: list[dict[str, Any]] = []
        for path in sorted(Path(p) for p in inputs):
            entry = self._image(path, cuts)
            if entry is not None:
                images.append(entry)
        if not images:
            raise ValueError("image_astrometry: none of the inputs is an image with a celestial WCS")
        images.sort(key=lambda e: (e["row"]["pivot"], e["row"]["file"]))
        ref = self._reference(images, reference)
        for e in images:
            e["row"]["reference"] = e is ref
        log.info("reference image: %s (%s, %d point sources)", ref["row"]["file"], ref["row"]["band"], len(ref["sources"]))

        summary: list[dict[str, Any]] = []
        matches: list[dict[str, Any]] = []
        test = {"radius": match_radius, "search": search, "clip": clip, "min_n": min_matches}
        for e in images:
            if e is ref:
                continue
            fwhm = max(e["row"]["fwhm"], ref["row"]["fwhm"])
            s, r = ref["sources"], e["sources"]
            comp = self._compare(r.ra, r.dec, s.ra, s.dec, fwhm, test)
            self._record(summary, matches, e, comp, against="reference", against_name=ref["row"]["band"],
                         ref_ids=s.ids, ref_has_pm=np.ones(len(s), bool))

        base = self._base(images, ctx)
        keys = self._keys(ref, cuts, test, len(images))
        cache = ctx.output_dir / derived_name(f"{base}.ecsv", GAIA_SUFFIX, ext=".ecsv")
        if gaia:
            keys.update(self._gaia(images, cache, gaia_services or list(SERVICES), gaia_pm_only, test, summary, matches))
        else:
            keys["JWFASGAI"] = ("off", "Gaia DR3: ok | cached | empty | failed | off")

        product = ctx.output_dir / derived_name(f"{base}.fits", ASTROM_SUFFIX)
        write_astrometry_product(product, images=[e["row"] for e in images], summary=summary, matches=matches,
                                 keys=keys, primary=images[0]["primary"])
        for row in summary:
            log.info("%-6s - %-8s N=%3d/%3d  dra %+8.2f +- %6.2f  ddec %+8.2f +- %6.2f mas (scatter %.1f, %.1f)",
                     row["band"], row["against_name"], row["n_used"], row["n_matched"], row["dra"], row["dra_error"],
                     row["ddec"], row["ddec_error"], row["dra_scatter"], row["ddec_scatter"])
        log.info("-> %s", product.name)
        return [product] + ([cache] if cache.is_file() else [])

    # -- one image ----------------------------------------------------------------------------
    @staticmethod
    def _image(path: Path, cuts: dict[str, float]) -> dict[str, Any] | None:
        image = read_image(path)
        if image is None:
            return None
        if image.wcs is None:
            log.warning("%s has no celestial WCS; skipped", path.name)
            return None
        band = image.band
        mjd = observation_mjd(image.primary)
        cat = catalog_path(path)
        row: dict[str, Any] = {"file": path.name, "catalog": cat.name if cat.is_file() else "",
                               "instrument": str(image.primary.get("INSTRUME", "")).strip(), "band": image.label,
                               "pivot": band.pivot_um if band else np.nan, "fwhm": band.fwhm_arcsec if band else np.nan,
                               "pixscale": image.pixscale_arcsec or np.nan, "mjd": mjd,
                               "epoch": julian_year(mjd) if np.isfinite(mjd) else np.nan, "snr_column": ""}
        if cat.is_file():
            sources = read_point_sources(image, cat, **cuts)
        else:
            log.warning("%s: no source catalogue (%s); listed, not compared", path.name, cat.name)
            from .catalogs import CUTS, PointSources

            z = np.zeros(0)
            sources = PointSources(np.zeros(0, np.int64), z, z, z, dict.fromkeys(CUTS, 0))
        row.update(sources.counts)
        row["snr_column"] = sources.snr_column
        log.info("%s (%s): %s", path.name, row["band"], ", ".join(f"{k[2:]} {v}" for k, v in sources.counts.items()))
        ny, nx = image.shape
        corners = image.wcs.all_pix2world(np.array([0, nx - 1, nx - 1, 0], float), np.array([0, 0, ny - 1, ny - 1], float), 0)
        return {"row": row, "sources": sources, "primary": image.primary, "corners": corners}

    @staticmethod
    def _reference(images: list[dict[str, Any]], reference: str | None) -> dict[str, Any]:
        """The image named by ``reference`` (band or file name), else the one with the most point sources."""
        if reference:
            key = str(reference).strip()
            named = [e for e in images if e["row"]["file"] == key or Path(e["row"]["file"]).stem == key
                     or e["row"]["band"].upper() == key.upper()]
            if not named:
                raise ValueError(f"reference {reference!r} is neither the band nor the file of an input "
                                 f"({', '.join(e['row']['band'] for e in images)})")
            candidates = named
        else:
            candidates = images
        return max(candidates, key=lambda e: (len(e["sources"]), -e["row"]["pivot"] if np.isfinite(e["row"]["pivot"]) else -np.inf))

    @staticmethod
    def _compare(ra: np.ndarray, dec: np.ndarray, ra_ref: np.ndarray, dec_ref: np.ndarray, fwhm: float,
                 test: dict[str, Any], eligible_ref: np.ndarray | None = None) -> Comparison:
        return compare(ra, dec, ra_ref, dec_ref, radius=test["radius"] * fwhm, search=test["search"],
                       eligible_ref=eligible_ref, clip=test["clip"], min_n=test["min_n"])

    @staticmethod
    def _record(summary: list[dict[str, Any]], matches: list[dict[str, Any]], e: dict[str, Any], comp: Comparison, *,
                against: str, against_name: str, ref_ids: np.ndarray, ref_has_pm: np.ndarray) -> None:
        row, src, st = e["row"], e["sources"], comp.stats
        summary.append({"file": row["file"], "band": row["band"], "pivot": row["pivot"], "against": against,
                        "against_name": against_name, "radius": comp.radius, "shift_dra": comp.shift[0],
                        "shift_ddec": comp.shift[1], "n_matched": comp.n_matched, "n_used": st.n_used, "dra": st.dra,
                        "ddec": st.ddec, "dra_scatter": st.dra_scatter, "ddec_scatter": st.ddec_scatter,
                        "dra_std": st.dra_std, "ddec_std": st.ddec_std,
                        "dra_error": st.dra_error, "ddec_error": st.ddec_error, "pixscale": row["pixscale"]})
        for k, (i, j) in enumerate(zip(comp.i, comp.j, strict=True)):
            matches.append({"file": row["file"], "band": row["band"], "against": against, "id": int(src.ids[i]),
                            "ref_id": int(ref_ids[j]), "ra": float(src.ra[i]), "dec": float(src.dec[i]),
                            "dra": float(comp.dra[k]), "ddec": float(comp.ddec[k]), "snr": float(src.snr[i]),
                            "ref_has_pm": bool(ref_has_pm[j]), "used": bool(st.used[k])})

    # -- Gaia ---------------------------------------------------------------------------------
    def _gaia(self, images: list[dict[str, Any]], cache: Path, services: list[str], pm_only: bool, test: dict[str, Any],
              summary: list[dict[str, Any]], matches: list[dict[str, Any]]) -> dict[str, Any]:
        ra = np.concatenate([np.asarray(e["corners"][0], float) for e in images])
        dec = np.concatenate([np.asarray(e["corners"][1], float) for e in images])
        ra0, dec0 = mean_position(ra, dec)
        radius = float(np.max(angular_distance(ra0, dec0, ra, dec))) + GAIA_MARGIN
        keys: dict[str, Any] = {"JWFASGRA": (ra0, "Gaia query centre RA [deg]"),
                                "JWFASGDE": (dec0, "Gaia query centre Dec [deg]"),
                                "JWFASGRD": (radius, "Gaia query radius [deg]"),
                                "JWFASGPM": (bool(pm_only), "median: Gaia sources with proper motion only")}
        try:
            table, query = load_or_query_gaia(cache, ra0, dec0, radius, services=services)
        except GaiaQueryError as exc:
            log.warning("Gaia DR3 unavailable (%s): the images are compared with each other only", exc)
            keys["JWFASGAI"] = ("failed", "Gaia DR3: ok | cached | empty | failed | off")
            keys["JWFASGER"] = (str(exc)[:45], "Gaia query error")
            return keys
        keys["JWFASGAI"] = ("empty" if len(table) == 0 else "cached" if query.get("reused") else "ok",
                            "Gaia DR3: ok | cached | empty | failed | off")
        keys["JWFASGSV"] = (str(query.get("service", "")), "Gaia DR3 service (esa | stsci)")
        keys["JWFASGQT"] = (str(query.get("queried", "")), "when Gaia DR3 was queried [UTC]")
        keys["JWFASGCT"] = (cache.name, "Gaia DR3 cache in this stage")
        keys["JWFASGN"] = (len(table), "Gaia DR3 sources in the query")
        if len(table) == 0:
            log.warning("Gaia DR3 has no source in the footprint")
            return keys
        ids = np.asarray(table["source_id"], dtype=np.int64)
        for e in images:
            epoch = e["row"]["epoch"]
            if not np.isfinite(epoch):
                log.warning("%s: no observation epoch in the header; not compared with Gaia", e["row"]["file"])
                continue
            gra, gdec, has_pm = propagate(table, epoch)
            src = e["sources"]
            comp = self._compare(src.ra, src.dec, gra, gdec, e["row"]["fwhm"], test,
                                 eligible_ref=has_pm if pm_only else None)
            self._record(summary, matches, e, comp, against="gaia", against_name=GAIA_NAME, ref_ids=ids, ref_has_pm=has_pm)
        return keys

    # -- the product ----------------------------------------------------------------------------
    @staticmethod
    def _base(images: list[dict[str, Any]], ctx: RunContext) -> str:
        """The leading ``_`` fields every input name shares (``jw04290-o003_t001_nircam``), else the target."""
        fields = [split_suffix(Path(e["row"]["file"]).stem)[0].split("_") for e in images]
        common: list[str] = []
        for parts in zip(*fields, strict=False):
            if len(set(parts)) != 1:
                break
            common.append(parts[0])
        if len(images) == 1:
            common = fields[0]
        return "_".join(common) if common else (slug(ctx.target).lower() or "images")

    @staticmethod
    def _keys(ref: dict[str, Any], cuts: dict[str, float], test: dict[str, Any], n_images: int) -> dict[str, Any]:
        return {"JWFASREF": (ref["row"]["file"], "reference image"),
                "JWFASRBD": (ref["row"]["band"], "band of the reference image"),
                "JWFASNIM": (n_images, "images compared"),
                "JWFASSNR": (float(cuts["min_snr"]), "point sources: minimum S/N"),
                "JWFASISO": (float(cuts["isolation"]), "point sources: no neighbour within [FWHM]"),
                "JWFASCOR": (float(cuts["core"]), "point sources: data within [FWHM]"),
                "JWFASMRD": (float(test["radius"]), "match radius [FWHM of the coarser band]"),
                "JWFASSRC": (float(test["search"]), "bulk-shift search radius [arcsec]"),
                "JWFASCLP": (float(test["clip"]), "offsets clipped at [robust scatters]"),
                "JWFASMIN": (int(test["min_n"]), "matches needed for scatter and error")}
