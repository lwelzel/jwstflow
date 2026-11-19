"""The ``*_gapcov.fits`` product: the coverage next to the gap on a sky grid, and looking a position up.

The layout is described in :mod:`jwstflow.cubes.gaps`.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

GAPCOV_SUFFIX = "gapcov"   # *_gapcov.fits: the per-group coverage product

#: Columns of the FILES table.
FILE_COLUMNS: tuple[str, ...] = ("file", "exposure", "detector", "n_examined", "n_staircase", "n_tested",
                                 "n_flagged", "edge_min", "edge_max")

#: Image extensions of the product (name, dtype).
PRODUCT_MAPS: tuple[tuple[str, str], ...] = (("LIMLO", "f8"), ("LIMHI", "f8"), ("REACHLO", "f8"), ("REACHHI", "f8"),
                                             ("NEXPLO", "i2"), ("NEXPHI", "i2"))


def write_gap_product(path: Path, header: Any, limits: dict[str, Any], files: list[dict[str, Any]],
                      keys: dict[str, Any] | None = None) -> Path:
    """Write a ``*_gapcov.fits`` product (layout in the module docstring)."""
    from astropy.io import fits
    from astropy.table import Table

    prim = fits.PrimaryHDU(header=fits.Header())
    for card in ("TELESCOP", "INSTRUME", "EXP_TYPE", "PROGRAM", "OBSERVTN", "VISIT", "GRATING", "FILTER",
                 "BKGDTARG", "TARGPROP", "TARG_RA", "TARG_DEC", "PATTTYPE", "NUMDTHPT", "CAL_VER", "CRDS_CTX"):
        if card in header:
            prim.header[card] = header[card]
    for k, v in (keys or {}).items():
        prim.header[k] = v
    hdr2d = limits["grid"].header()
    hdus: list[Any] = [prim]
    for name, dtype in PRODUCT_MAPS:
        h = hdr2d.copy()
        if dtype.startswith("f"):
            h["BUNIT"] = "um"
        hdus.append(fits.ImageHDU(np.asarray(limits[name.lower()], dtype=dtype), header=h, name=name))
    table = {col: [f[col] for f in files] for col in FILE_COLUMNS} if files else {c: [] for c in FILE_COLUMNS}
    hdus.append(fits.BinTableHDU(Table(table), name="FILES"))
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return Path(path)


def read_gap_product(path: str | Path) -> dict[str, Any]:
    """A ``*_gapcov.fits`` product as plain arrays: the maps (lower-case names), ``header``
    (primary), ``map_header`` (the grid's WCS), ``files`` (table)."""
    from astropy.io import fits
    from astropy.table import Table

    with fits.open(path) as hdul:
        out: dict[str, Any] = {"path": Path(path), "header": hdul[0].header.copy(),
                               "map_header": hdul["LIMLO"].header.copy(),
                               "files": Table(hdul["FILES"].data) if "FILES" in hdul else Table()}
        for name, _ in PRODUCT_MAPS:
            out[name.lower()] = np.asarray(hdul[name].data)
    return out


def gap_limits_at(product: dict[str, Any] | str | Path, ra: Any, dec: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(limlo, limhi)`` [um] of a gap product at sky positions (degrees): at these positions all
    (the required fraction of) exposures contribute for wavelengths <= limlo and >= limhi; in
    between only some do (thinner sampling, more noise). NaN outside the grid or where no
    exposure covers the position."""
    from astropy.wcs import WCS

    prod = read_gap_product(product) if not isinstance(product, dict) else product
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wcs = WCS(prod["map_header"])
    px, py = wcs.all_world2pix(np.asarray(ra, float), np.asarray(dec, float), 0)
    ix, iy = np.rint(px).astype(int), np.rint(py).astype(int)
    ny, nx = prod["limlo"].shape
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    lo = np.full(np.shape(ix), np.nan)
    hi = np.full(np.shape(ix), np.nan)
    lo[ok] = prod["limlo"][iy[ok], ix[ok]]
    hi[ok] = prod["limhi"][iy[ok], ix[ok]]
    return lo, hi
