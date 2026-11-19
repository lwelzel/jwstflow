"""The NIRSpec detector gap: test the pixels next to it against the other exposures, keep the flux.

Next to the gap a cube is thinly sampled. With the high-resolution gratings
every NIRSpec IFU slice runs off NRS1 and resumes on NRS2, the 30 slices
cross the gap at staggered wavelengths (G235H/F170LP: NRS1 ends at
2.36-2.43 um, NRS2 starts at 2.42-2.49 um), and every dither moves a sky
position onto other slices. In the combined cube the gap of a spaxel is the
union of the gaps of the slices feeding it, and its edges are a staircase:
approaching the gap the exposures drop out one after the other, so the
voxels of the staircase (15-30 planes on each side, more at the field edges)
are built from a few exposures down to single pixels. Their noise grows
accordingly (2-5 times the noise of the same spaxel away from the gap in
ESO-Ha 569/574) and nothing marks them in the cube.

What those voxels hold is nevertheless physical. Compared with the other
exposures at the same sky position and wavelength, the detector pixels that
build them -- the "staircase pixels" -- agree as well as any pixel away from
the gap (ESO-Ha 574 G235H: 93 % of them testable, robust scatter 0.90 of
their errors, 0.01 % beyond 5 sigma; interior pixels 0.92 and 0.02 %), and
the detector-edge pixels are not biased. Masking by coverage -- dropping what
not all exposures reach, or voxels below some fraction of the usual weight --
removes real flux: the emission of the part of the disk the moving gap has
not yet covered (5-70 % of the H2 1-0 Q-line flux next to the gap in
ESO-Ha 574, even at 5 % of the usual weight). So this module never flags by
coverage.

:class:`FlagGapOutliers` (``flag_gap_outliers``) tests instead. For every
group of exposures combined into one cube (``group_by``) it measures on a
fine sky grid how far towards the gap each exposure reaches and from that
the staircase pixels -- those beyond the wavelength all (``fraction``)
exposures covering their position reach, where ``outlier_detection`` has the
fewest exposures to compare. Each staircase pixel is compared with the other
exposures' pixels at its sky position (within ``footprint``), interpolated to
its wavelength: real emission is in every exposure that sees the position,
an artefact (an unflagged hot pixel, a cosmic ray, a detector-edge effect)
in one. Pixels deviating by more than ``sigma`` (their errors plus a
``rel_tol`` fraction of the reference, for spatial structure below the
footprint) from the median of at least ``min_others`` other exposures get
``DO_NOT_USE | OUTLIER``; pixels with fewer exposures to compare are kept --
they are the only data there -- and counted as unverified.

The product (``<group>_gapcov.fits``, one per group) records the coverage on
the sky grid, with a celestial TAN WCS, so later steps can tell how thinly a
position is sampled at a wavelength (:func:`gap_limits_at`) without the cal
files, and ``qa_gap_coverage`` can draw it:

    LIMLO    f8 (ny, nx)  last wavelength [um] before the gap that the required exposures reach
    LIMHI    f8 (ny, nx)  first wavelength [um] after the gap that they reach
    REACHLO  f8 (ny, nx)  the reddest wavelength any exposure reaches before the gap
    REACHHI  f8 (ny, nx)  the bluest wavelength any exposure reaches after it
    NEXPLO   i2 (ny, nx)  exposures covering the cell on the blue side (NRS1)
    NEXPHI   i2 (ny, nx)  exposures covering the cell on the red side (NRS2)
    FILES    bintable     one row per cal file: pixels examined, in the staircase, tested, flagged

(NaN / 0 where no exposure covers a cell; between LIMLO and REACHLO, and
between REACHHI and LIMHI, a position is seen by only some exposures.)
Heavy imports stay inside the functions.

The package, by module:

* :mod:`.edges` -- where the spectra meet the gap on the detector, the gap window, the pixels in it;
* :mod:`.coverage` -- each exposure's reach on a sky grid, the staircase pixels;
* :mod:`.compare` -- the test against the other exposures;
* :mod:`.product` -- the ``*_gapcov.fits`` product and :func:`gap_limits_at`;
* :mod:`.groups` -- the exposures that make one cube;
* :mod:`.steps` -- ``flag_gap_outliers``.
"""

from .compare import cross_exposure_deviation
from .coverage import CoverageGrid, coverage_limits, staircase_pixels
from .edges import (
    BLUE_DETECTOR,
    RED_DETECTOR,
    gap_samples,
    gap_window,
    slice_edges,
    slice_edges_from_wcs,
)
from .groups import DEFAULT_GROUP_BY, exposure_of, group_key, group_name
from .product import (
    FILE_COLUMNS,
    GAPCOV_SUFFIX,
    PRODUCT_MAPS,
    gap_limits_at,
    read_gap_product,
    write_gap_product,
)
from .steps import FlagGapOutliers

__all__ = [
    "BLUE_DETECTOR",
    "DEFAULT_GROUP_BY",
    "FILE_COLUMNS",
    "GAPCOV_SUFFIX",
    "PRODUCT_MAPS",
    "RED_DETECTOR",
    "CoverageGrid",
    "FlagGapOutliers",
    "coverage_limits",
    "cross_exposure_deviation",
    "exposure_of",
    "gap_limits_at",
    "gap_samples",
    "gap_window",
    "group_key",
    "group_name",
    "read_gap_product",
    "slice_edges",
    "slice_edges_from_wcs",
    "staircase_pixels",
    "write_gap_product",
]
