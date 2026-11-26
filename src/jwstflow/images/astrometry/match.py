"""Matching two lists of sky positions and the robust statistics of their offsets.

:func:`compare` is the whole comparison of :mod:`jwstflow.images.astrometry`: the bulk shift
(:func:`bulk_shift`), one-to-one matches around it (:func:`match_pairs`), the offsets of the
matched pairs (:func:`sky_offsets_mas`, image minus reference) and their median, robust scatter and
error of the median after clipping (:func:`robust_offset`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Milliarcseconds per degree.
MAS_PER_DEG = 3.6e6
#: Robust scatter of a Gaussian from its median absolute deviation.
MAD_TO_SIGMA = 1.4826
#: Standard error of the median of N Gaussian samples, in units of sigma / sqrt(N): sqrt(pi / 2).
MEDIAN_ERROR = math.sqrt(math.pi / 2.0)


def tangent_plane(ra: np.ndarray, dec: np.ndarray, ra0: float, dec0: float) -> np.ndarray:
    """(n, 2) gnomonic projection [arcsec] about (ra0, dec0): x to the east, y to the north."""
    a, d = np.radians(np.asarray(ra, float)), np.radians(np.asarray(dec, float))
    a0, d0 = math.radians(ra0), math.radians(dec0)
    cos_c = math.sin(d0) * np.sin(d) + math.cos(d0) * np.cos(d) * np.cos(a - a0)
    x = np.cos(d) * np.sin(a - a0) / cos_c
    y = (math.cos(d0) * np.sin(d) - math.sin(d0) * np.cos(d) * np.cos(a - a0)) / cos_c
    return np.degrees(np.column_stack([x, y])) * 3600.0


def mean_position(ra: np.ndarray, dec: np.ndarray) -> tuple[float, float]:
    """The direction of the mean unit vector of some positions [deg]."""
    a, d = np.radians(np.asarray(ra, float)), np.radians(np.asarray(dec, float))
    v = np.array([np.sum(np.cos(d) * np.cos(a)), np.sum(np.cos(d) * np.sin(a)), np.sum(np.sin(d))])
    return float(np.degrees(np.arctan2(v[1], v[0])) % 360.0), float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))))


def sky_offsets_mas(ra: np.ndarray, dec: np.ndarray, ra_ref: np.ndarray, dec_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(dra, ddec) [mas] of positions relative to reference positions: the offset in right
    ascension times cos(dec) (east positive) and in declination (north positive). Exact to
    microarcseconds for offsets of arcseconds."""
    ra, dec, ra_ref, dec_ref = (np.asarray(v, float) for v in (ra, dec, ra_ref, dec_ref))
    dra = ((ra - ra_ref + 180.0) % 360.0 - 180.0) * np.cos(np.radians(0.5 * (dec + dec_ref)))
    return dra * MAS_PER_DEG, (dec - dec_ref) * MAS_PER_DEG


def bulk_shift(xy: np.ndarray, xy_ref: np.ndarray, *, search: float, radius: float) -> tuple[float, float, int]:
    """The common shift [arcsec] of ``xy`` against ``xy_ref`` (tangent-plane positions in arcsec):
    every pair within ``search`` votes with its offset, and the shift is the median offset of the
    largest group of votes that lie within ``radius`` of one of them. Returns ``(dx, dy, votes)``;
    (0, 0, 0) without any pair."""
    from scipy.spatial import cKDTree

    if len(xy) == 0 or len(xy_ref) == 0:
        return 0.0, 0.0, 0
    pairs = cKDTree(xy).query_ball_tree(cKDTree(xy_ref), r=search)
    offsets = np.array([xy[i] - xy_ref[j] for i, js in enumerate(pairs) for j in js]).reshape(-1, 2)
    if offsets.size == 0:
        return 0.0, 0.0, 0
    tree = cKDTree(offsets)
    support = np.array([len(n) for n in tree.query_ball_point(offsets, r=radius)])
    best = int(np.argmax(support))
    group = offsets[tree.query_ball_point(offsets[best], r=radius)]
    dx, dy = np.median(group, axis=0)
    return float(dx), float(dy), len(group)


def match_pairs(xy: np.ndarray, xy_ref: np.ndarray, *, radius: float, shift: tuple[float, float] = (0.0, 0.0)
                ) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one matches ``(i, j)`` of ``xy`` to ``xy_ref``: mutual nearest neighbours within
    ``radius`` once ``shift`` is taken off ``xy`` (all in arcsec)."""
    from scipy.spatial import cKDTree

    empty = (np.zeros(0, int), np.zeros(0, int))
    if len(xy) == 0 or len(xy_ref) == 0:
        return empty
    moved = np.asarray(xy, float) - np.asarray(shift, float)
    dist, j = cKDTree(xy_ref).query(moved, k=1, distance_upper_bound=radius)
    _, back = cKDTree(moved).query(xy_ref, k=1, distance_upper_bound=radius)
    i = np.arange(len(moved))
    ok = np.isfinite(dist) & (j < len(xy_ref))
    ok[ok] &= back[j[ok]] == i[ok]
    return i[ok], j[ok]


@dataclass
class Offset:
    """The robust statistics of one set of offsets."""

    used: np.ndarray          #: which samples entered the median (finite, eligible, not clipped)
    dra: float                #: median [mas]
    ddec: float
    dra_scatter: float        #: 1.4826 x MAD of the used samples [mas]
    ddec_scatter: float
    dra_std: float            #: standard deviation of the used samples [mas]
    ddec_std: float
    dra_error: float          #: error of the median, 1.2533 x max(scatter, std) / sqrt(N) [mas]
    ddec_error: float

    @property
    def n_used(self) -> int:
        return int(self.used.sum())


#: Clipping needs at least this many samples: below it no sample can be told an outlier.
MIN_CLIP = 5


def robust_offset(dra: np.ndarray, ddec: np.ndarray, *, eligible: np.ndarray | None = None, clip: float = 3.0,
                  min_n: int = 3, max_iter: int = 10) -> Offset:
    """Median, robust scatter (1.4826 x MAD), standard deviation and error of the median of offsets.

    While at least :data:`MIN_CLIP` samples are in use, those farther than ``clip`` robust scatters
    from the median in either coordinate are clipped, iteratively (never below ``min_n``).
    ``eligible`` leaves samples out from the start. The error of the median is 1.2533 x sigma /
    sqrt(N) with sigma the larger of the robust scatter and the standard deviation of the samples
    used: for a few samples the MAD collapses (of three it is the smaller of two differences) and
    would claim a precision the data do not have; for many they agree. With fewer than ``min_n``
    samples the median is given without scatter, standard deviation or error (NaN)."""
    dra, ddec = np.asarray(dra, float), np.asarray(ddec, float)
    used = np.isfinite(dra) & np.isfinite(ddec)
    if eligible is not None:
        used &= np.asarray(eligible, bool)
    nan = float("nan")
    if not used.any():
        return Offset(used, nan, nan, nan, nan, nan, nan, nan, nan)
    for _ in range(max_iter):
        if used.sum() < max(MIN_CLIP, min_n):
            break
        med = np.array([np.median(dra[used]), np.median(ddec[used])])
        sig = MAD_TO_SIGMA * np.array([np.median(np.abs(dra[used] - med[0])), np.median(np.abs(ddec[used] - med[1]))])
        keep = used.copy()
        for k, values in enumerate((dra, ddec)):
            if sig[k] > 0:
                keep &= np.abs(values - med[k]) <= clip * sig[k]
        if np.array_equal(keep, used) or keep.sum() < min_n:
            break
        used = keep
    n = int(used.sum())
    med = np.array([np.median(dra[used]), np.median(ddec[used])])
    if n < min_n:
        return Offset(used, float(med[0]), float(med[1]), nan, nan, nan, nan, nan, nan)
    sig = MAD_TO_SIGMA * np.array([np.median(np.abs(dra[used] - med[0])), np.median(np.abs(ddec[used] - med[1]))])
    std = np.array([np.std(dra[used], ddof=1), np.std(ddec[used], ddof=1)])
    err = MEDIAN_ERROR * np.maximum(sig, std) / math.sqrt(n)
    return Offset(used, float(med[0]), float(med[1]), float(sig[0]), float(sig[1]), float(std[0]), float(std[1]),
                  float(err[0]), float(err[1]))


@dataclass
class Comparison:
    """One image compared with a reference (another image, or Gaia): the matches and their statistics."""

    i: np.ndarray             #: indices of the matched sources in the image's list
    j: np.ndarray             #: ... and in the reference list
    dra: np.ndarray           #: per-pair offsets, image minus reference [mas]
    ddec: np.ndarray
    stats: Offset
    shift: tuple[float, float]   #: the bulk shift of the first pass [mas]
    radius: float             #: the match radius [arcsec]

    @property
    def n_matched(self) -> int:
        return int(self.i.size)


def compare(ra: np.ndarray, dec: np.ndarray, ra_ref: np.ndarray, dec_ref: np.ndarray, *, radius: float, search: float,
            eligible_ref: np.ndarray | None = None, clip: float = 3.0, min_n: int = 3) -> Comparison:
    """Match a list of positions [deg] to a reference list and measure the offsets: bulk shift
    within ``search``, mutual nearest neighbours within ``radius`` of it (both arcsec), offsets
    image minus reference, and their :func:`robust_offset` (``eligible_ref`` marks the reference
    entries allowed into the median)."""
    ra, dec, ra_ref, dec_ref = (np.asarray(v, float) for v in (ra, dec, ra_ref, dec_ref))
    if ra.size == 0 or ra_ref.size == 0:
        z = np.zeros(0)
        return Comparison(np.zeros(0, int), np.zeros(0, int), z, z, robust_offset(z, z), (0.0, 0.0), radius)
    ra0, dec0 = mean_position(np.concatenate([ra, ra_ref]), np.concatenate([dec, dec_ref]))
    xy, xy_ref = tangent_plane(ra, dec, ra0, dec0), tangent_plane(ra_ref, dec_ref, ra0, dec0)
    dx, dy, _ = bulk_shift(xy, xy_ref, search=max(search, radius), radius=radius)
    i, j = match_pairs(xy, xy_ref, radius=radius, shift=(dx, dy))
    dra, ddec = sky_offsets_mas(ra[i], dec[i], ra_ref[j], dec_ref[j])
    eligible = None if eligible_ref is None else np.asarray(eligible_ref, bool)[j]
    stats = robust_offset(dra, ddec, eligible=eligible, clip=clip, min_n=min_n)
    return Comparison(i, j, dra, ddec, stats, (dx * 1e3, dy * 1e3), radius)
