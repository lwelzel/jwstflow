"""Cluster regions: where to look for a bad cluster, and which planes, dithers and cubes it concerns."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from ...steps import StepParams

#: Header keywords that identify the *band* a dither cube belongs to (never the detector:
#: a per-dither cube combines both NIRSpec detectors, whose cal files must both match it).
BAND_KEYS: tuple[str, ...] = ("PROGRAM", "OBSERVTN", "INSTRUME", "GRATING", "FILTER", "CHANNEL", "BAND")

#: ``mode`` values of a region as recorded in the product (why a dither was included).
MODE_AUTO, MODE_ALL, MODE_EXPLICIT = "auto", "all", "explicit"


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
    from ...data.targets import parse_coords

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

    from ...data.targets import parse_coords

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
