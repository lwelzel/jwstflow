"""The point sources of an image: the catalogue ``source_catalog`` writes next to it, and the cuts.

The cuts are described in :mod:`jwstflow.images.astrometry`; every length is in units of the
band's diffraction-limited FWHM on the image's pixels (:attr:`jwstflow.images.Image.fwhm_pix`).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...steps.naming import split_suffix
from .. import Image

log = logging.getLogger(__name__)

#: ``source_catalog`` (image3) writes ``<base>_cat.ecsv`` next to ``<base>_i2d.fits``.
CATALOG_SUFFIX = "cat"
#: (flux, error) columns the S/N of a source is taken from, in order of preference.
SNR_COLUMNS: tuple[tuple[str, str], ...] = (("aper_total_flux", "aper_total_flux_err"),
                                            ("isophotal_flux", "isophotal_flux_err"))
#: The cuts in the order they are applied: the name of the count of sources left after each.
CUTS: tuple[str, ...] = ("n_catalog", "n_finite", "n_compact", "n_snr", "n_isolated", "n_point")


@dataclass
class PointSources:
    """The point sources of one image that passed every cut."""

    ids: np.ndarray            #: catalogue ``label``
    ra: np.ndarray             #: [deg]
    dec: np.ndarray            #: [deg]
    snr: np.ndarray            #: S/N of the flux in ``snr_column``
    counts: dict[str, int] = field(default_factory=dict)   #: sources left after each cut (:data:`CUTS`)
    snr_column: str = ""

    def __len__(self) -> int:
        return int(self.ra.size)


def catalog_path(image_path: Path | str) -> Path:
    """The ``<base>_cat.ecsv`` ``source_catalog`` writes next to ``<base>_i2d.fits``."""
    image_path = Path(image_path)
    base, _ = split_suffix(image_path.stem)
    return image_path.with_name(f"{base}_{CATALOG_SUFFIX}.ecsv")


def observation_mjd(primary: Any) -> float:
    """Mid-exposure epoch [MJD] of an image from its primary header: the middle of EXPSTART and
    EXPEND, else EXPMID / MJD-AVG, else DATE-OBS (+ TIME-OBS); NaN when none is there."""
    try:
        start, end = float(primary["EXPSTART"]), float(primary["EXPEND"])
        if math.isfinite(start) and math.isfinite(end):
            return 0.5 * (start + end)
    except (KeyError, TypeError, ValueError):
        pass
    for key in ("EXPMID", "MJD-AVG", "MJD-OBS"):
        try:
            value = float(primary[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    date = str(primary.get("DATE-OBS", "") or "").strip()
    if not date:
        return float("nan")
    time = str(primary.get("TIME-OBS", "") or "").strip()
    from astropy.time import Time

    try:
        return float(Time(f"{date}T{time}" if time and "T" not in date else date, scale="utc").mjd)
    except ValueError:
        return float("nan")


def read_point_sources(image: Image, catalog: Any, *, min_snr: float, isolation: float, core: float) -> PointSources:
    """The point sources of ``image`` in its source catalogue (an astropy Table or the path of the
    ``_cat.ecsv``) after the cuts of :mod:`jwstflow.images.astrometry`, with the number left after
    each (:data:`CUTS`)."""
    from astropy.table import Table

    table = Table.read(catalog, format="ascii.ecsv") if isinstance(catalog, (str, Path)) else catalog
    n = len(table)
    counts = dict.fromkeys(CUTS, 0)
    counts["n_catalog"] = n
    empty = PointSources(np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0), counts)
    fwhm = image.fwhm_pix
    if n == 0 or not fwhm:
        if n and not fwhm:
            log.warning("%s: no band or pixel scale, so no resolution to scale the cuts with; no point sources",
                        image.path.name if image.path else "image")
        return empty

    ra, dec = _sky_positions(table, image)
    x, y = _pixel_positions(table, image, ra, dec)
    ids = np.asarray(table["label"], dtype=np.int64) if "label" in table.colnames else np.arange(1, n + 1)
    finite = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(x) & np.isfinite(y)
    counts["n_finite"] = keep = int(finite.sum())
    good = finite.copy()

    if "is_extended" in table.colnames:
        extended = np.asarray(_filled(table["is_extended"], True), dtype=bool)
        good &= ~extended
    else:
        log.warning("%s: the catalogue has no is_extended column; every source counts as compact", _name(catalog))
    counts["n_compact"] = int(good.sum())

    snr, snr_column = _snr(table)
    good &= np.isfinite(snr) & (snr >= min_snr)
    counts["n_snr"] = int(good.sum())

    # isolation against every catalogued source with a centroid, compact or not
    if keep > 1:
        from scipy.spatial import cKDTree

        xy = np.column_stack([x[finite], y[finite]])
        dist, _ = cKDTree(xy).query(xy, k=2)
        nearest = np.full(n, np.inf)
        nearest[finite] = dist[:, 1]
        good &= nearest >= isolation * fwhm
    counts["n_isolated"] = int(good.sum())

    good[good] = core_complete(image, x[good], y[good], core * fwhm)
    counts["n_point"] = int(good.sum())
    return PointSources(ids[good], ra[good], dec[good], snr[good], counts, snr_column)


def core_complete(image: Image, x: np.ndarray, y: np.ndarray, radius_pix: float) -> np.ndarray:
    """Whether every pixel within ``radius_pix`` (at least one pixel) of each (x, y) [0-based] has
    data: finite SCI and, for a resampled image, positive weight. False where the disk leaves the image."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size == 0:
        return np.zeros(0, bool)
    r = max(float(radius_pix), 1.0)
    k = math.ceil(r)
    dy, dx = np.mgrid[-k:k + 1, -k:k + 1]
    inside = dx**2 + dy**2 <= r**2
    dy, dx = dy[inside], dx[inside]
    iy = np.rint(y)[:, None].astype(int) + dy[None, :]
    ix = np.rint(x)[:, None].astype(int) + dx[None, :]
    ny, nx = image.shape
    on = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
    iy_c, ix_c = np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)
    has = np.isfinite(image.data[iy_c, ix_c])
    if image.weight is not None:
        has &= image.weight[iy_c, ix_c] > 0
    return np.all(on & has, axis=1)


def _sky_positions(table: Any, image: Image) -> tuple[np.ndarray, np.ndarray]:
    """(ra, dec) [deg] of every catalogue row: ``sky_centroid``, else the pixel centroid through the WCS."""
    if "sky_centroid" in table.colnames:
        coords = table["sky_centroid"]
        try:
            return (np.asarray(_filled(coords.ra.deg, np.nan), float), np.asarray(_filled(coords.dec.deg, np.nan), float))
        except AttributeError:   # not a SkyCoord column (an older or hand-made catalogue)
            pass
    if "xcentroid" in table.colnames and image.wcs is not None:
        ra, dec = image.wcs.all_pix2world(_column(table, "xcentroid"), _column(table, "ycentroid"), 0)
        return np.asarray(ra, float), np.asarray(dec, float)
    raise KeyError("the catalogue has neither sky_centroid nor xcentroid/ycentroid")


def _pixel_positions(table: Any, image: Image, ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) [0-based pixels of the image] of every catalogue row."""
    if "xcentroid" in table.colnames and "ycentroid" in table.colnames:
        return _column(table, "xcentroid"), _column(table, "ycentroid")
    if image.wcs is None:
        raise KeyError("the catalogue has no pixel centroids and the image no WCS")
    x, y = image.wcs.all_world2pix(ra, dec, 0)
    return np.asarray(x, float), np.asarray(y, float)


def _snr(table: Any) -> tuple[np.ndarray, str]:
    for flux, err in SNR_COLUMNS:
        if flux in table.colnames and err in table.colnames:
            f, e = _column(table, flux), _column(table, err)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(e > 0, f / e, np.nan), flux
    log.warning("the catalogue has none of the flux columns %s; no source passes the S/N cut",
                [f for f, _ in SNR_COLUMNS])
    return np.full(len(table), np.nan), ""


def _column(table: Any, name: str) -> np.ndarray:
    col = table[name]
    value = getattr(col, "value", col)          # Quantity -> plain values
    return np.asarray(_filled(value, np.nan), dtype=float)


def _filled(values: Any, fill: Any) -> Any:
    return values.filled(fill) if hasattr(values, "filled") else values


def _name(catalog: Any) -> str:
    return Path(catalog).name if isinstance(catalog, (str, Path)) else "catalogue"
