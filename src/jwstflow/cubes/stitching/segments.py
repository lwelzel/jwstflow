"""The segments of a joint cube: one IFU cube per grating / band, read into plain arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..wcs import hdu_wavelengths
from .grid import SpatialGrid, data_bounds, joint_grid


@dataclass
class CubeSegment:
    """One cube of the set that is joined: its planes, their wavelengths and its spatial grid.

    ``data`` is the surface brightness ``(nw, ny, nx)`` with NaN wherever the
    cube holds no usable value (no data, or DO_NOT_USE in DQ); ``err``,
    ``dq`` and ``wmap`` are None when the file has no such extension.
    """

    path: Path
    label: str
    wave: np.ndarray
    data: np.ndarray
    err: np.ndarray | None
    dq: np.ndarray | None
    wmap: np.ndarray | None
    grid: SpatialGrid
    unit: str | None
    primary: Any
    sci_header: Any

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.data)

    @property
    def planes_with_data(self) -> np.ndarray:
        return self.valid.any(axis=(1, 2))

    @property
    def wave_range(self) -> tuple[float, float]:
        """Wavelengths [um] of the first and the last plane that carry data."""
        wave = self.wave[self.planes_with_data]
        return float(wave.min()), float(wave.max())


def segment_label(header: Any, path: Path) -> str:
    """Short name of a segment from its primary header: ``g235h-f170lp`` for a NIRSpec cube,
    ``1-short`` for an MRS band (the labels 1-D stitching uses), else the file stem."""
    for keys in (("GRATING", "FILTER"), ("CHANNEL", "BAND")):
        values = [str(header.get(k, "") or "").strip() for k in keys]
        if all(values) and not any(v.upper() in ("N/A", "MULTIPLE") for v in values):
            return "-".join(v.lower() for v in values)
    return Path(path).stem


def read_segment(path: Path | str) -> CubeSegment:
    """Read an ``*_s3d.fits`` cube as a :class:`CubeSegment`."""
    from astropy.io import fits

    path = Path(path)
    with fits.open(path) as hdul:
        if "SCI" not in hdul or hdul["SCI"].data is None or hdul["SCI"].data.ndim != 3:
            raise ValueError(f"{path.name}: no 3-D SCI extension")
        sci_header = hdul["SCI"].header.copy()
        primary = hdul[0].header.copy()
        data = np.array(hdul["SCI"].data, dtype=np.float32)
        wave = hdu_wavelengths(hdul)

        def plane_array(name: str, dtype: Any) -> np.ndarray | None:
            if name in hdul and hdul[name].data is not None and hdul[name].data.shape == data.shape:
                return np.array(hdul[name].data, dtype=dtype)
            return None

        err, dq, wmap = plane_array("ERR", np.float32), plane_array("DQ", np.uint32), plane_array("WMAP", np.float32)
    if np.any(np.diff(wave) <= 0):
        raise ValueError(f"{path.name}: the wavelengths of its planes do not increase")
    if dq is not None:
        data[(dq & np.uint32(1)) != 0] = np.nan                 # DO_NOT_USE: not a measurement
    if not np.isfinite(data).any():
        raise ValueError(f"{path.name}: no spaxel carries data")
    return CubeSegment(path, segment_label(primary, path), wave, data, err, dq, wmap,
                       SpatialGrid.from_header(sci_header, data.shape), sci_header.get("BUNIT"), primary, sci_header)


def joint_grid_of(paths: list[Path]) -> SpatialGrid:
    """The joint grid (:func:`~jwstflow.cubes.stitching.grid.joint_grid`) of the cubes in these
    files, in order of their wavelengths: the finest spaxel scale over the union of the spaxels
    that carry data. Only the SCI and DQ extensions are read."""
    from astropy.io import fits

    found: list[tuple[float, SpatialGrid, tuple[int, int, int, int] | None]] = []
    for path in paths:
        with fits.open(path) as hdul:
            if "SCI" not in hdul or hdul["SCI"].data is None or hdul["SCI"].data.ndim != 3:
                raise ValueError(f"{Path(path).name}: no 3-D SCI extension")
            valid = np.isfinite(hdul["SCI"].data)
            if "DQ" in hdul and hdul["DQ"].data is not None and hdul["DQ"].data.shape == valid.shape:
                valid &= (np.asarray(hdul["DQ"].data).astype(np.uint32) & np.uint32(1)) == 0
            planes = valid.any(axis=(1, 2))
            if not planes.any():
                continue
            found.append((float(hdu_wavelengths(hdul)[planes].min()), SpatialGrid.from_header(hdul["SCI"].header, valid.shape),
                          data_bounds(valid)))
    if not found:
        raise ValueError("none of the cubes carries data")
    found.sort(key=lambda item: item[0])
    return joint_grid([grid for _, grid, _ in found], [bounds for _, _, bounds in found])
