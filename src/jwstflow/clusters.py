"""Bad spaxel clusters: find them in per-dither cubes, flag the detector pixels behind them.

The problem this solves. Combined IFU cubes (``calwebb_spec3``) sometimes show
compact clusters of wildly deviant spaxels over a few wavelength planes --
extreme outliers in morphology as well as in value -- that the official
pipeline never flags. The reason is structural: ``outlier_detection`` for IFU
data (jwst >= 1.11) works in *detector* space and keeps, per detector pixel,
the **minimum** neighbour difference over *all* exposures, so it only catches
defects present in every dither (bad pixels missing from the mask); anything
confined to one dither survives by design. ``cube_build``'s drizzle then
averages the dithers without any clipping, so the artefact lands in the cube,
diluted by the number of dithers but not removed.

The cure is to go back to the exposure the cluster comes from. The chain,
three steps that any workflow can insert between stage 2 and stage 3:

1. **per-dither cubes** -- the official ``cube_build`` step on one association
   per dither (``group_by: [..., PATT_NUM]``), so every dither has its own
   cube (``stage3/cube_build-<variant>/``);
2. :class:`FlagSpaxelClusters` (``flag_spaxel_clusters``) -- for every region
   named in the workflow (sky position, search aperture, wavelength window),
   decides *which dither cubes* carry the cluster and *which spaxels* inside
   the aperture are deviant, by comparing each dither with the others on the
   same sky grid; writes one ``*_clustermask.fits`` product per dither cube;
3. :class:`PropagateClusterFlags` (``propagate_cluster_flags``) -- evaluates
   the cal files' WCS to find the detector pixels whose (RA, Dec, wavelength)
   fall on flagged spaxels and marks them ``DO_NOT_USE | OUTLIER`` in DQ;
   the DQ-edited copies (official ``_cal`` names, JWFCL* header record) feed
   every downstream stage, so the clusters never reach the science cubes.

``qa_spaxel_clusters`` (:mod:`jwstflow.contrib.clusters`) draws every dither
at the affected wavelengths from the products alone.

The product (``<cube base>_clustermask.fits``) follows the mask contract of
:mod:`jwstflow.masks` (MASK/CONT/CONTIMG on the cube's own grid) and adds:

    WAVES     f8 (nw,)         wavelength [um] of every cube plane
    REGIONS   bintable         one row per region evaluated on this cube (see REGION_COLUMNS)
    SLICES    f4 (ns, ny, nx)  EXTVER = region row + 1: the cube planes around the window
    REFS      f4 (ns, ny, nx)  the same planes as seen by the other dithers (median, resampled)
    SIGMA     f4 (ns, ny, nx)  the significance of the deviation from the closest other dither
    SLICEWAV  f8 (ns,)         wavelengths of the SLICES planes
    APER      u1 (ny, nx)      the region's search aperture on this grid

so a QA step never reopens a cube. Heavy imports stay inside the functions.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from .masks import celestial_wcs, dilate, resample_image, strip_wave_axis
from .naming import derived_name
from .steps.base import RunContext, Step, StepParams

log = logging.getLogger(__name__)

CLUSTERMASK_SUFFIX = "clustermask"   # *_clustermask.fits: the per-dither cluster product

#: Header keywords that identify the *band* a dither cube belongs to (never the detector:
#: a per-dither cube combines both NIRSpec detectors, whose cal files must both match it).
BAND_KEYS: tuple[str, ...] = ("PROGRAM", "OBSERVTN", "INSTRUME", "GRATING", "FILTER", "CHANNEL", "BAND")

#: Columns of the REGIONS table (one row per region x cube).
REGION_COLUMNS: tuple[str, ...] = (
    "id", "ra", "dec", "shape", "radius", "radius_minor", "angle", "wave_min", "wave_max",
    "plane_min", "plane_max", "n_planes", "dither", "mode", "included", "refined", "n_aperture",
    "n_deviant", "n_flagged", "score", "noise", "n_others", "reason",
)

#: ``mode`` values of a region as recorded in the product (why a dither was included).
MODE_AUTO, MODE_ALL, MODE_EXPLICIT = "auto", "all", "explicit"


# --------------------------------------------------------------------------- regions
class ClusterRegion(StepParams):
    """One bad cluster to look for: where on the sky, how wide, at which wavelengths."""

    id: str | None = Field(None, description="label of the region (default: region<n>); names the QA figure")
    ra: Any = Field(description="ICRS right ascension of the search aperture centre (degrees or sexagesimal)")
    dec: Any = Field(description="ICRS declination (degrees or sexagesimal)")
    shape: Literal["circle", "ellipse", "polygon"] = Field("circle", description="search-aperture shape")
    radius: float | None = Field(None, gt=0, description="aperture radius [arcsec] (ellipse: semi-major axis)")
    radius_minor: float | None = Field(None, gt=0, description="ellipse semi-minor axis [arcsec]")
    angle: float = Field(0.0, description="ellipse position angle of the major axis [deg, east of north]")
    vertices: list[list[Any]] | None = Field(None, description="polygon vertices [[ra, dec], ...] (degrees or sexagesimal)")
    wave_min: float = Field(gt=0, description="start of the affected wavelength window [um]")
    wave_max: float = Field(gt=0, description="end of the affected wavelength window [um]")
    where: dict[str, Any] = Field(default_factory=dict,
                                  description="restrict to cubes whose primary header matches these keywords, "
                                              "e.g. {GRATING: G395H}; default: every cube covering the window")
    dithers: Literal["auto", "all"] | list[int] = Field(
        "auto", description="which dither cubes to flag: 'auto' = those deviating from the other dithers, "
                            "'all' = every dither, or an explicit list of dither numbers (PATT_NUM)")
    refine: bool = Field(True, description="flag only the deviant spaxels inside the aperture (the cluster's "
                                           "actual shape); false flags the whole aperture")

    @model_validator(mode="after")
    def _consistent(self) -> ClusterRegion:
        if self.wave_max <= self.wave_min:
            raise ValueError(f"region {self.id or ''}: wave_max must exceed wave_min")
        if self.shape in ("circle", "ellipse") and self.radius is None:
            raise ValueError(f"region {self.id or ''}: {self.shape} needs a radius")
        if self.shape == "polygon" and (not self.vertices or len(self.vertices) < 3):
            raise ValueError(f"region {self.id or ''}: a polygon needs at least 3 vertices")
        return self


def region_centre(region: dict[str, Any]) -> tuple[float, float]:
    """ICRS (ra, dec) in degrees of a region (polygons: the centroid of their vertices)."""
    from .targets import parse_coords

    if region.get("shape") == "polygon" and region.get("vertices"):
        pts = np.array([parse_coords(a, b) for a, b in region["vertices"]], dtype=float)
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    return parse_coords(region["ra"], region["dec"])


def sky_offsets(ra: np.ndarray, dec: np.ndarray, ra0: float, dec0: float) -> tuple[np.ndarray, np.ndarray]:
    """Tangent-plane offsets [arcsec] (east, north) of positions from ``(ra0, dec0)``."""
    dra = (np.asarray(ra, float) - ra0 + 180.0) % 360.0 - 180.0
    east = dra * np.cos(np.radians(dec0)) * 3600.0
    north = (np.asarray(dec, float) - dec0) * 3600.0
    return east, north


def region_aperture(region: dict[str, Any], wcs: Any, shape: tuple[int, int]) -> np.ndarray:
    """Boolean image of the region's search aperture on a cube grid (spaxel centres inside)."""
    ny, nx = shape
    yy, xx = np.indices((ny, nx))
    ra, dec = wcs.all_pix2world(xx, yy, 0)
    ra0, dec0 = region_centre(region)
    east, north = sky_offsets(ra, dec, ra0, dec0)
    kind = region.get("shape", "circle")
    if kind == "circle":
        return east**2 + north**2 <= float(region["radius"]) ** 2
    if kind == "ellipse":
        a = float(region["radius"])
        b = float(region.get("radius_minor") or a)
        pa = np.radians(float(region.get("angle", 0.0)))   # east of north: major axis = (sin pa, cos pa)
        u = east * np.sin(pa) + north * np.cos(pa)
        v = -east * np.cos(pa) + north * np.sin(pa)
        return (u / a) ** 2 + (v / b) ** 2 <= 1.0
    from matplotlib.path import Path as MplPath

    from .targets import parse_coords

    verts = np.array([sky_offsets(*parse_coords(a, b), ra0, dec0) for a, b in region["vertices"]], dtype=float)
    inside = MplPath(verts).contains_points(np.column_stack([east.ravel(), north.ravel()]))
    return inside.reshape(ny, nx)


def planes_in_window(wave: np.ndarray, wave_min: float, wave_max: float, pad: int = 0) -> np.ndarray:
    """Indices of the planes whose wavelength *bin* overlaps ``[wave_min, wave_max]`` (never
    empty: at least the nearest plane), widened by ``pad`` planes on each side."""
    wave = np.asarray(wave, float)
    step = float(np.median(np.abs(np.diff(wave)))) if wave.size > 1 else 0.0
    hit = (wave + step / 2 >= wave_min) & (wave - step / 2 <= wave_max)
    if not hit.any():
        hit[int(np.argmin(np.abs(wave - 0.5 * (wave_min + wave_max))))] = True
    idx = np.flatnonzero(hit)
    lo, hi = max(0, idx.min() - pad), min(wave.size - 1, idx.max() + pad)
    return np.arange(lo, hi + 1)


def dither_of(header: Any, path: str | Path | None = None) -> int:
    """Dither number of a cube / cal file: PATT_NUM, else ``dither<N>`` in the file name,
    else the DMS exposure counter, else 0."""
    val = header.get("PATT_NUM") if header is not None else None
    if val is not None and str(val).strip().lstrip("-").isdigit():
        return int(val)
    name = Path(path).name if path is not None else ""
    m = re.search(r"dither(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"_(\d{5})_[a-z][a-z0-9]*_", name)   # the DMS exposure counter before the detector
    return int(m.group(1)) if m else 0


def band_key(header: Any) -> tuple[Any, ...]:
    """The :data:`BAND_KEYS` values of a header (lower-cased strings; None when absent)."""
    out = []
    for key in BAND_KEYS:
        val = header.get(key)
        out.append(None if val is None else str(val).strip().lower())
    return tuple(out)


def matches_where(header: Any, where: dict[str, Any]) -> bool:
    for key, want in (where or {}).items():
        have = header.get(str(key).upper())
        if have is None or str(have).strip().lower() != str(want).strip().lower():
            return False
    return True


# --------------------------------------------------------------------------- the comparison
def robust_sigma(values: np.ndarray) -> float:
    """1.4826 x MAD of the finite values (0.0 when there are none)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def deviation_from_others(plane: np.ndarray, wave_k: float, this_wcs: Any, others: list[dict[str, Any]],
                          ) -> tuple[np.ndarray, np.ndarray, int]:
    """Compare one cube plane with the other dithers at the same wavelength.

    Each other cube's nearest plane (within one plane step) is resampled onto
    this cube's grid. Returns ``(dev, ref, n)``: the *signed deviation with the
    smallest magnitude* over the others (a bad dither deviates from every
    other one; a good dither agrees with at least one), their nanmedian as a
    reference image, and how many others contributed. ``n == 0`` when no
    other dither covers the wavelength.
    """
    import warnings

    ny, nx = plane.shape
    stack = []
    for other in others:
        wave = other["wave"]
        k = int(np.argmin(np.abs(wave - wave_k)))
        step = float(np.median(np.abs(np.diff(wave)))) if wave.size > 1 else np.inf
        if abs(wave[k] - wave_k) > step:
            continue
        stack.append(resample_image(other["data"][k], other["wcs"], this_wcs, (ny, nx)))
    if not stack:
        nan = np.full((ny, nx), np.nan)
        return nan, nan, 0
    imgs = np.stack(stack)
    devs = plane[None] - imgs
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*All-NaN.*")
        absdev = np.where(np.isfinite(devs), np.abs(devs), np.inf)
        idx = np.argmin(absdev, axis=0)
        dev = np.take_along_axis(devs, idx[None], axis=0)[0]
        ref = np.nanmedian(imgs, axis=0)
    return dev, ref, len(stack)


# --------------------------------------------------------------------------- the product
def write_cluster_product(path: Path, cube_path: Path, mask: np.ndarray, waves: np.ndarray, contimg: np.ndarray,
                          regions: list[dict[str, Any]], slices: list[dict[str, Any]],
                          keys: dict[str, Any] | None = None) -> Path:
    """Write a ``*_clustermask.fits`` product (layout in the module docstring)."""
    from astropy.io import fits
    from astropy.table import Table

    with fits.open(cube_path) as hdul:
        prim_hdr = hdul[0].header.copy()
        sci_hdr = hdul["SCI"].header.copy() if "SCI" in hdul else fits.Header()
    prim = fits.PrimaryHDU(header=prim_hdr)
    for k, v in (keys or {}).items():
        prim.header[k] = v
    hdr2d = strip_wave_axis(sci_hdr)
    cont = mask.any(axis=0)
    hdus: list[Any] = [prim,
                       fits.ImageHDU(mask.astype("u1"), header=sci_hdr.copy(), name="MASK"),
                       fits.ImageHDU(cont.astype("u1"), header=hdr2d.copy(), name="CONT"),
                       fits.ImageHDU(np.asarray(contimg, dtype="f4"), header=hdr2d.copy(), name="CONTIMG"),
                       fits.ImageHDU(np.asarray(waves, dtype="f8"), name="WAVES")]
    table = {col: [r[col] for r in regions] for col in REGION_COLUMNS} if regions else {col: [] for col in REGION_COLUMNS}
    hdus.append(fits.BinTableHDU(Table(table), name="REGIONS"))
    for i, s in enumerate(slices, start=1):
        for name, arr, dtype in (("SLICES", s["data"], "f4"), ("REFS", s["ref"], "f4"), ("SIGMA", s["sigma"], "f4"),
                                 ("SLICEWAV", s["wave"], "f8"), ("APER", s["aperture"], "u1")):
            hdu = fits.ImageHDU(np.asarray(arr, dtype=dtype), name=name)
            hdu.header["EXTVER"] = i
            hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return Path(path)


def read_cluster_product(path: Path) -> dict[str, Any]:
    """Read a ``*_clustermask.fits`` product back into plain arrays / a table."""
    from astropy.io import fits
    from astropy.table import Table

    with fits.open(path) as hdul:
        regions = Table(hdul["REGIONS"].data) if "REGIONS" in hdul else Table()
        slices = []
        for i in range(1, len(regions) + 1):
            if ("SLICES", i) not in hdul:
                break
            slices.append({
                "data": np.asarray(hdul["SLICES", i].data, dtype=float),
                "ref": np.asarray(hdul["REFS", i].data, dtype=float),
                "sigma": np.asarray(hdul["SIGMA", i].data, dtype=float),
                "wave": np.asarray(hdul["SLICEWAV", i].data, dtype=float),
                "aperture": np.asarray(hdul["APER", i].data).astype(bool),
            })
        return {
            "path": Path(path),
            "header": hdul[0].header.copy(),
            "mask_header": hdul["MASK"].header.copy(),
            "mask": np.asarray(hdul["MASK"].data).astype(bool),
            "cont": np.asarray(hdul["CONT"].data).astype(bool),
            "contimg": np.asarray(hdul["CONTIMG"].data, dtype=float),
            "waves": np.asarray(hdul["WAVES"].data, dtype=float),
            "regions": regions,
            "slices": slices,
        }


def find_cluster_products(header: Any, path: Path, directory: Path) -> list[dict[str, Any]]:
    """The cluster products of ``directory`` belonging to a cal file's band *and* dither."""
    from astropy.io import fits

    key, dither = band_key(header), dither_of(header, path)
    out = []
    for f in sorted(Path(directory).glob(f"*_{CLUSTERMASK_SUFFIX}.fits")):
        hdr = fits.getheader(f)
        if band_key(hdr) == key and dither_of(hdr, f) == dither:
            out.append(read_cluster_product(f))
    return out


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

        from .spectra import cube_wavelengths

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
def _dq_bits() -> int:
    try:
        from stdatamodels.jwst.datamodels.dqflags import pixel

        return int(pixel["DO_NOT_USE"]) | int(pixel["OUTLIER"])
    except ImportError:  # pragma: no cover - jwst is a dependency
        return 1 | 16


def detector_sky_coordinates(model: Any, windows: list[tuple[float, float]], *, coarse_step: int = 8,
                             ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """(y, x, ra, dec, wavelength) of the detector pixels of ``model`` that may fall
    inside one of the wavelength ``windows`` [um].

    NIRSpec IFU exposures are evaluated slice by slice
    (``jwst.assign_wcs.nirspec.nrs_ifu_wcs``); any other exposure through its
    ``meta.wcs`` and bounding box. A coarse pass (every ``coarse_step``-th
    column, every row) locates the columns whose wavelengths reach a window;
    only those (with a margin) are evaluated at full resolution. Wavelengths
    are returned in microns whatever the WCS emits.
    """
    import warnings

    from gwcs.wcstools import grid_from_bounding_box

    wcs = getattr(model.meta, "wcs", None)
    if wcs is None:
        raise ValueError(f"{model.meta.filename}: no WCS; run assign_wcs (calwebb_spec2) first")
    exp_type = str(getattr(model.meta.exposure, "type", "") or "").upper()
    frames = set(getattr(wcs, "available_frames", []) or [])
    if exp_type == "NRS_IFU" and frames & {"gwa", "slicer"}:
        from jwst.assign_wcs.nirspec import nrs_ifu_wcs

        slice_wcs = nrs_ifu_wcs(model)
    else:
        slice_wcs = [wcs]
    ny_det, nx_det = model.dq.shape

    def evaluate(w: Any, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.errstate(all="ignore"):
            ra, dec, lam = w(x, y, with_bounding_box=True)
        ra, dec, lam = (np.asarray(v, dtype=float) for v in (ra, dec, lam))
        if np.isfinite(lam).any() and np.nanmedian(lam) < 1e-3:   # metres -> microns
            lam = lam * 1e6
        return ra, dec, lam

    for w in slice_wcs:
        bbox = getattr(w, "bounding_box", None)
        if bbox is None:
            xc, yc = np.meshgrid(np.arange(0, nx_det, coarse_step), np.arange(ny_det))
        else:
            xc, yc = grid_from_bounding_box(bbox, step=(coarse_step, 1))
        xc, yc = np.asarray(xc, float), np.asarray(yc, float)
        if xc.ndim == 1:
            xc, yc = xc[None], yc[None]
        _, _, lam_c = evaluate(w, xc, yc)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.filterwarnings("ignore", message=".*All-NaN.*")
            col_lo, col_hi = np.nanmin(lam_c, axis=0), np.nanmax(lam_c, axis=0)
        # a window may lie entirely between two coarse columns: bracket each column with its neighbour
        span_lo, span_hi = col_lo.copy(), col_hi.copy()
        span_lo[:-1] = np.fmin(col_lo[:-1], col_lo[1:])
        span_hi[:-1] = np.fmax(col_hi[:-1], col_hi[1:])
        keep = np.zeros(xc.shape[1], dtype=bool)
        for wlo, whi in windows:
            keep |= np.isfinite(span_lo) & (span_lo <= whi) & (span_hi >= wlo)
        if not keep.any():
            continue
        cols = xc[0]
        idx = np.flatnonzero(keep)
        x_lo = int(np.floor(cols[max(0, idx.min() - 2)]))
        x_hi = int(np.ceil(cols[min(len(cols) - 1, idx.max() + 2)]))
        y_lo, y_hi = int(np.floor(np.nanmin(yc))), int(np.ceil(np.nanmax(yc)))
        x_lo, x_hi = max(0, x_lo), min(nx_det - 1, x_hi)
        y_lo, y_hi = max(0, y_lo), min(ny_det - 1, y_hi)
        yy, xx = np.mgrid[y_lo:y_hi + 1, x_lo:x_hi + 1]
        ra, dec, lam = evaluate(w, xx.astype(float), yy.astype(float))
        ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(lam)
        if ok.any():
            yield yy[ok], xx[ok], ra[ok], dec[ok], lam[ok]


def detector_pixels_in_masks(model: Any, products: list[dict[str, Any]], *, grow: int = 0, plane_pad: int = 1,
                             ) -> tuple[np.ndarray, list[str]]:
    """Boolean detector image of the pixels that land on flagged spaxels of ``products``,
    and the product names that contributed.

    The cube masks are widened first: ``plane_pad`` planes along wavelength and
    ``grow`` spaxels spatially, so every detector pixel drizzled into a
    flagged spaxel is caught. A pixel is flagged when its wavelength's nearest
    plane and its sky position's nearest spaxel are inside the widened mask.
    """
    from astropy.wcs import WCS
    from scipy.ndimage import maximum_filter1d

    flag = np.zeros(model.dq.shape, dtype=bool)
    targets = []
    for p in products:
        mask = p["mask"]
        if not mask.any():
            continue
        wide = mask.astype("u1")
        if plane_pad > 0:
            wide = maximum_filter1d(wide, size=2 * plane_pad + 1, axis=0, mode="constant")
        wide = wide > 0
        if grow > 0:
            wide = np.stack([dilate(pl, grow) if pl.any() else pl for pl in wide])
        waves = p["waves"]
        step = float(np.median(np.abs(np.diff(waves)))) if waves.size > 1 else 0.0
        planes = np.flatnonzero(wide.any(axis=(1, 2)))
        window = (float(waves[planes].min() - step), float(waves[planes].max() + step))
        hdr = p["mask_header"].copy()
        hdr["NAXIS"] = 3
        wcs2d = WCS(hdr).celestial
        targets.append((p["path"].name, wcs2d, waves, step, wide, window))
    if not targets:
        return flag, []
    used: set[str] = set()
    for yy, xx, ra, dec, lam in detector_sky_coordinates(model, [t[-1] for t in targets]):
        for name, wcs2d, waves, step, wide, window in targets:
            inside = (lam >= window[0]) & (lam <= window[1])
            if not inside.any():
                continue
            k = np.clip(np.rint(np.interp(lam[inside], waves, np.arange(waves.size))).astype(int), 0, waves.size - 1)
            ok = np.abs(waves[k] - lam[inside]) <= max(step, 1e-9)
            with np.errstate(all="ignore"):
                px, py = wcs2d.all_world2pix(ra[inside], dec[inside], 0)
            ix, iy = np.rint(px).astype(int), np.rint(py).astype(int)
            _, ny, nx = wide.shape
            ok &= (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            hit = np.zeros(inside.sum(), dtype=bool)
            hit[ok] = wide[k[ok], iy[ok], ix[ok]]
            if hit.any():
                sel = np.flatnonzero(inside)[hit]
                flag[yy[sel], xx[sel]] = True
                used.add(name)
    return flag, sorted(used)


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
                model.dq[pixels] |= _dq_bits()
                model.save(str(dst))
            regions = sorted({str(r["id"]) for p in active for r in p["regions"] if bool(r["included"])})
            log.info("%s: %d detector pixel(s) flagged from %d product(s)", src.name, n_flagged, len(used))
        with fits.open(dst, mode="update") as hdul:
            hdul[0].header["JWFCLNPX"] = (n_flagged, "propagate_cluster_flags: detector pixels flagged")
            hdul[0].header["JWFCLMSK"] = (",".join(used) or "none", "propagate_cluster_flags: products applied")
            hdul[0].header["JWFCLRGS"] = (",".join(regions) or "none", "propagate_cluster_flags: regions applied")
        return [dst]
