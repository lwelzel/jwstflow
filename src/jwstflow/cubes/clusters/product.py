"""The ``*_clustermask.fits`` product: writing, reading, and finding the ones of a cal file.

The layout is described in :mod:`jwstflow.cubes.clusters`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..wcs import strip_wave_axis
from .regions import band_key, dither_of

CLUSTERMASK_SUFFIX = "clustermask"   # *_clustermask.fits: the per-dither cluster product

#: Columns of the REGIONS table (one row per region x cube).
REGION_COLUMNS: tuple[str, ...] = (
    "id", "ra", "dec", "shape", "radius", "radius_minor", "angle", "wave_min", "wave_max",
    "plane_min", "plane_max", "n_planes", "dither", "mode", "included", "refined", "n_aperture",
    "n_deviant", "n_flagged", "score", "noise", "n_others", "reason",
)


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
