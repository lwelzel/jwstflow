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

The helpers such steps share live next door: :mod:`.wcs` (a celestial WCS from
cube metadata, reprojection between cube grids), :mod:`.morphology` (NaN-aware
smoothing, morphological cleanup) and :mod:`.instrument` (matching a product
to a cube's configuration). Heavy dependencies are imported lazily.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..steps import RunContext
from .instrument import INSTRUMENT_KEYS, matches_instrument
from .wcs import strip_wave_axis

log = logging.getLogger(__name__)


def write_mask_product(path: Path, cube_path: Path, mask: np.ndarray, cont: np.ndarray, contimg: np.ndarray,
                       *, features: dict[str, list] | None = None, featimg: list[np.ndarray] | None = None,
                       featmask: list[np.ndarray] | None = None, keys: dict[str, Any] | None = None,
                       headers: tuple[Any, Any] | None = None) -> Path:
    """Write a mask product on the cube's own pixel grid (layout in the module docstring).

    ``keys`` adds/overrides PRIMARY keywords (record the producing step's
    parameters there). ``features`` is a column dict for the FEATURES table;
    it, ``featimg`` and ``featmask`` are written only when features were
    evaluated. ``headers`` -- (primary, SCI) -- replaces the headers of
    ``cube_path`` when the mask is on another grid than the file's: a cutout
    of an image (:meth:`jwstflow.images.Image.cutout`) carries its own WCS.
    """
    from astropy.io import fits
    from astropy.table import Table

    if headers is not None:
        prim_hdr, sci_hdr = headers[0].copy(), headers[1].copy()
    else:
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
