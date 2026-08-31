"""The mask-product contract: wavelength-resolved spatial masks for IFU cubes.

Many reductions need "a boolean region per cube plane" -- a source aperture, a
background exclusion zone, a contamination mask. This module fixes one file
layout for such products so that steps from different packages (core,
contributed, project-specific) can produce and consume each other's masks:

    <base>_<suffix>.fits          suffix is the producing step's own (e.g. ``diskmask``)
      PRIMARY                     header copied from the source cube (instrument keys travel along)
      MASK      u1 (nw, ny, nx)   the per-plane mask, on the cube's own s3d pixel grid + WCS
      CONT      u1 (ny, nx)       the wavelength-invariant part of the mask
      CONTIMG   f4 (ny, nx)       a collapsed (nanmedian) image of the cube, for QA overlays
      FEATIMG   f4 (nf, ny, nx)   optional: per-feature local median images
      FEATMASK  u1 (nf, ny, nx)   optional: per-feature local masks
      FEATURES  bintable          optional: one row per evaluated feature window
                                  (id, label, species, wave_min_um, wave_max_um,
                                  nplanes, added_pix, changed)

Producers call :func:`write_mask_product`; consumers call
:func:`read_mask_product` directly, or :func:`mask_from_stage` to locate the
product of a companion stage that matches a cube's instrument configuration.
The product is self-describing (the QA payload travels inside it), so QA steps
never need to reopen a cube.

The module also collects the geometry helpers such steps share: a celestial
WCS from cube metadata, NaN-aware smoothing and reprojection between cube
grids, and morphological cleanup. Everything imports its heavy dependencies
lazily so the module stays importable without astropy/scipy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .steps.base import RunContext

log = logging.getLogger(__name__)

#: Header keywords that identify a cube's instrument configuration; a mask
#: product matches a cube when every keyword present in both headers agrees.
INSTRUMENT_KEYS = ("INSTRUME", "DETECTOR", "GRATING", "FILTER", "CHANNEL", "BAND")


# --------------------------------------------------------------------------- geometry
def celestial_wcs(cube: Any) -> Any:
    """Celestial (2-D) WCS of an IFU cube datamodel, built from ``meta.wcsinfo``."""
    from astropy.io import fits
    from astropy.wcs import WCS

    w = cube.meta.wcsinfo
    hdr = fits.Header()
    for i, axis in ((1, "1"), (2, "2")):
        hdr[f"CTYPE{i}"] = getattr(w, f"ctype{axis}")
        hdr[f"CRVAL{i}"] = getattr(w, f"crval{axis}")
        hdr[f"CRPIX{i}"] = getattr(w, f"crpix{axis}")
        hdr[f"CDELT{i}"] = getattr(w, f"cdelt{axis}")
        hdr[f"CUNIT{i}"] = getattr(w, f"cunit{axis}") or "deg"
    for key in ("pc1_1", "pc1_2", "pc2_1", "pc2_2"):
        val = getattr(w, key, None)
        if val is not None:
            hdr[key.upper()] = val
    hdr["NAXIS"] = 2
    return WCS(hdr).celestial


def nan_smooth(img: np.ndarray, sigma: float) -> np.ndarray:
    """NaN-aware Gaussian smoothing (weights = finite pixels)."""
    from scipy.ndimage import gaussian_filter

    if sigma <= 0:
        return img
    finite = np.isfinite(img)
    v = gaussian_filter(np.where(finite, img, 0.0), sigma)
    w = gaussian_filter(finite.astype(float), sigma)
    return np.where(w > 0.05, v / np.maximum(w, 1e-12), np.nan)


def pixel_map(src_wcs: Any, dst_wcs: Any, dst_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(y, x) coordinates in `src_wcs` pixels of every pixel centre of the destination grid."""
    yy, xx = np.indices(dst_shape)
    ra, dec = dst_wcs.all_pix2world(xx, yy, 0)
    sx, sy = src_wcs.all_world2pix(ra, dec, 0)
    return np.asarray(sy, float), np.asarray(sx, float)


def resample_image(img: np.ndarray, src_wcs: Any, dst_wcs: Any, dst_shape: tuple[int, int]) -> np.ndarray:
    """Bilinear, NaN-aware resampling of `img` onto the destination grid (outside -> NaN)."""
    from scipy.ndimage import map_coordinates

    sy, sx = pixel_map(src_wcs, dst_wcs, dst_shape)
    finite = np.isfinite(img)
    v = map_coordinates(np.where(finite, img, 0.0), [sy, sx], order=1, mode="constant", cval=0.0)
    w = map_coordinates(finite.astype(float), [sy, sx], order=1, mode="constant", cval=0.0)
    return np.where(w > 0.5, v / np.maximum(w, 1e-12), np.nan)


def sample_mask(mask: np.ndarray, src_wcs: Any, dst_wcs: Any, dst_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour sampling of a boolean mask onto the destination grid (outside -> False)."""
    from scipy.ndimage import map_coordinates

    sy, sx = pixel_map(src_wcs, dst_wcs, dst_shape)
    return map_coordinates(mask.astype("u1"), [sy, sx], order=0, mode="constant", cval=0) > 0


def dilate(mask: np.ndarray, n: int) -> np.ndarray:
    """Binary dilation by ``n`` pixels (``n <= 0`` returns the mask unchanged)."""
    from scipy.ndimage import binary_dilation

    return binary_dilation(mask, iterations=n) if n > 0 else mask


def clean_blob(mask: np.ndarray, anchor: np.ndarray | None = None) -> np.ndarray:
    """Close small gaps, fill holes, and keep only structure connected to `anchor`
    (or, without an anchor, the largest connected component)."""
    from scipy.ndimage import binary_closing, binary_fill_holes, label

    if not mask.any():
        return mask
    m = binary_fill_holes(binary_closing(mask, structure=np.ones((3, 3), bool)))
    labels, n = label(m, structure=np.ones((3, 3), int))
    if n <= 1:
        return m
    if anchor is not None and anchor.any():
        keep = np.unique(labels[anchor & (labels > 0)])
    else:
        keep = [int(np.argmax(np.bincount(labels[labels > 0].ravel())))]
    return np.isin(labels, [k for k in keep if k > 0])


def strip_wave_axis(header: Any) -> Any:
    """Copy of a cube SCI header with the 3rd (wavelength) axis removed, for 2-D extensions."""
    hdr = header.copy()
    for key in ("CTYPE3", "CRVAL3", "CRPIX3", "CDELT3", "CUNIT3", "CRDER3", "NAXIS3",
                "PC1_3", "PC2_3", "PC3_1", "PC3_2", "PC3_3", "CD1_3", "CD2_3", "CD3_1", "CD3_2", "CD3_3",
                "PS3_0", "PS3_1", "PV3_0", "PV3_1"):
        hdr.remove(key, ignore_missing=True, remove_all=True)
    if "WCSAXES" in hdr:
        hdr["WCSAXES"] = 2
    return hdr


# --------------------------------------------------------------------------- the product
def write_mask_product(path: Path, cube_path: Path, mask: np.ndarray, cont: np.ndarray, contimg: np.ndarray,
                       *, features: dict[str, list] | None = None, featimg: list[np.ndarray] | None = None,
                       featmask: list[np.ndarray] | None = None, keys: dict[str, Any] | None = None) -> Path:
    """Write a mask product on the cube's own pixel grid (layout in the module docstring).

    ``keys`` adds/overrides PRIMARY keywords (record the producing step's
    parameters there). ``features`` is a column dict for the FEATURES table;
    it, ``featimg`` and ``featmask`` are written only when features were
    evaluated.
    """
    from astropy.io import fits
    from astropy.table import Table

    with fits.open(cube_path) as hdul:
        prim_hdr = hdul[0].header.copy()
        sci_hdr = hdul["SCI"].header.copy() if "SCI" in hdul else fits.Header()
    prim = fits.PrimaryHDU(header=prim_hdr)
    for k, v in (keys or {}).items():
        prim.header[k] = v
    hdr2d = strip_wave_axis(sci_hdr)
    hdus = [prim,
            fits.ImageHDU(mask.astype("u1"), header=sci_hdr.copy(), name="MASK"),
            fits.ImageHDU(cont.astype("u1"), header=hdr2d.copy(), name="CONT"),
            fits.ImageHDU(np.asarray(contimg, dtype="f4"), header=hdr2d.copy(), name="CONTIMG")]
    if features and features.get("id"):
        hdus += [fits.ImageHDU(np.stack(featimg).astype("f4"), name="FEATIMG"),
                 fits.ImageHDU(np.stack(featmask).astype("u1"), name="FEATMASK"),
                 fits.BinTableHDU(Table(features), name="FEATURES")]
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return Path(path)


def read_mask_product(path: Path) -> dict[str, Any]:
    """Read a mask product back into plain arrays (keys mirror the layout)."""
    from astropy.io import fits
    from astropy.table import Table

    with fits.open(path) as hdul:
        return {
            "path": Path(path),
            "header": hdul[0].header.copy(),
            "mask": np.asarray(hdul["MASK"].data).astype(bool),
            "cont": np.asarray(hdul["CONT"].data).astype(bool),
            "contimg": np.asarray(hdul["CONTIMG"].data, dtype=float),
            "features": Table(hdul["FEATURES"].data) if "FEATURES" in hdul else None,
            "featimg": np.asarray(hdul["FEATIMG"].data, dtype=float) if "FEATIMG" in hdul else None,
            "featmask": np.asarray(hdul["FEATMASK"].data).astype(bool) if "FEATMASK" in hdul else None,
        }


def matches_instrument(header_a: Any, header_b: Any) -> bool:
    """True when every :data:`INSTRUMENT_KEYS` keyword present in both headers agrees."""
    for key in INSTRUMENT_KEYS:
        a, b = header_a.get(key), header_b.get(key)
        if a is not None and b is not None and str(a).strip().lower() != str(b).strip().lower():
            return False
    return True


def find_mask_product(cube_path: Path, directory: Path, suffix: str) -> dict[str, Any]:
    """The ``*_<suffix>.fits`` product in ``directory`` matching the cube's instrument config."""
    from astropy.io import fits

    cube_hdr = fits.getheader(cube_path)
    candidates = sorted(Path(directory).glob(f"*_{suffix}.fits"))
    for f in candidates:
        if matches_instrument(cube_hdr, fits.getheader(f)):
            return read_mask_product(f)
    config = {k: cube_hdr.get(k) for k in INSTRUMENT_KEYS if cube_hdr.get(k) is not None}
    raise FileNotFoundError(
        f"no *_{suffix}.fits matching {config} in {directory} ({len(candidates)} product(s) present)"
    )


def mask_from_stage(cube_path: Path, ctx: RunContext, stage: str | None, suffix: str) -> dict[str, Any] | None:
    """The mask product of another stage for this cube, or None when the stage is absent.

    Returns None (with a warning) when ``stage`` is empty or not part of the
    workflow, so consuming steps can fall back to mask-free behaviour; raises
    when the stage exists but holds no matching product (a wiring error --
    remember ``depends_on`` on the consuming stage).
    """
    if not stage:
        return None
    try:
        mask_dir = ctx.dir_of(stage)
    except KeyError:
        log.warning("stage %r is not part of this workflow; falling back to mask-free behaviour", stage)
        return None
    try:
        return find_mask_product(cube_path, mask_dir, suffix)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc}; run the {stage!r} stage first (and declare depends_on: [{stage}])") from exc


def check_mask_matches(product: dict[str, Any], data: np.ndarray, cube_path: Path) -> None:
    """Raise with a helpful message when a mask product's geometry does not fit a cube."""
    if product["mask"].shape != data.shape:
        raise ValueError(
            f"mask {product['path'].name} has shape {product['mask'].shape} but cube {cube_path.name} has "
            f"{data.shape}; the mask was built for a different cube geometry -- rerun its producing stage"
        )
