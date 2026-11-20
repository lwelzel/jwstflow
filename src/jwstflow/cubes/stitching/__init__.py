"""Joint cubes: the band cubes of one instrument joined along wavelength, on one spatial grid.

**Status: in development.** The steps of this package (``cube_build_matched``,
``stitch_cubes``, ``qa_cube_stitch``) are still being worked on: a joint cube
is not a science-ready product, and no reduction recipe builds one. For
science use the band cubes and the stitched spectra.

An IFU observation comes out of the pipeline as one cube per grating (NIRSpec)
or per band (MIRI MRS: twelve), each on a grid of its own. A *joint cube* is
one cube for the whole wavelength range of the instrument -- the cube
counterpart of a stitched spectrum, built by the same rules
(:mod:`jwstflow.spectra.stitching`) and with one aim: every number in it is a
number the pipeline measured, in the place where it was measured.

**The spatial grid** (:mod:`.grid`). The joint grid has the finest spaxel scale
among the segments and covers the union of their fields: for the MRS the
0.13" spaxels of channel 1 over the field of channel 4. It extends the pixel
lattice of the finest segment (tangent point, orientation and scale are that
cube's), laid out as ``cube_build`` lays cubes out, so the grid can be handed
to ``cube_build`` as parameters.

**Getting the segments onto it** (:mod:`.regrid`, :mod:`.matched`). There are
two ways, and the product records which one each segment took:

* ``cube_build_matched`` builds the band cubes on the joint grid in the first
  place: the official ``cube_build``, given the grid, drizzles the detector
  pixels of every band straight onto it. The data are resampled once, as in
  any cube, and ``stitch_cubes`` then copies spaxels. This is the way to
  build a joint cube whenever the calibrated exposures are at hand.
* a finished cube on any other grid is resampled with the overlap areas
  between its spaxels and the joint spaxels as weights. That conserves flux
  exactly and invents nothing, but it is a second resampling: a spaxel is
  spread over the joint spaxels it straddles (the three sub-bands of an MRS
  channel sit up to three quarters of a spaxel apart), and coarser spaxels
  remain visible as blocks. Errors are propagated as for independent
  spaxels; ``correlation`` in the SEGMENTS table says by how much sums over
  many spaxels are then more uncertain than their errors in quadrature.

**The wavelength axis.** Nothing is resampled along wavelength. Segments are
ordered blue to red and the cube switches from one to the next at a crossover
(default: the midpoint of their overlap); each plane is one plane of one
segment at its own wavelength, and the axis is the table of those wavelengths
(``CTYPE3 = 'WAVE-TAB'``, as in the official multi-band cubes), sampled
differently from segment to segment.

**The overlaps** (:mod:`.measure`, :mod:`jwstflow.spectra.overlaps`). For each
pair of neighbours the flux of both segments is measured over one region of
sky -- where both are complete and the source stands out, clear of the rim of
the field (a source the field cuts is held differently by each band) -- as the excess over
the level of the field in each plane (the cube counterpart of an on/off
extraction), and the ratio of the two region spectra is taken as for stitched
spectra: around the crossover, as a ratio of medians with a propagated
uncertainty. Measured on the excess, the ratio is the factor between the two
bands whatever their levels are; the levels themselves (the sky, or what a
sky subtraction left behind -- rarely the same in two bands) are recorded
next to it, because a difference there is additive and no factor removes it.
The ratios are always recorded. Applied (``rescale``), they are one
multiplicative factor per segment, chained onto a reference segment, with the
factor's uncertainty folded into ERR; the spatial distribution within a band
is never touched.

**Positions.** A segment is placed by its own WCS, so a source sits where
that band's astrometry puts it. Whether two neighbours agree is measured, not
assumed: for every crossover the two segments are registered against each
other at the same wavelengths, over one window around the bright core they
share, and the shift is recorded.

The product (``<common name>_s3dcomb.fits``) is an ``IFUCubeModel`` -- it
opens with ``stdatamodels`` like any ``_s3d`` cube, and every step that reads
cubes through :func:`jwstflow.cubes.cube_wavelengths` reads it -- followed by
the records of the stitch:

    SCI, ERR, DQ, WMAP  (nw, ny, nx)  the joint cube; DQ 513 (DO_NOT_USE | NON_SCIENCE) where no segment has data
    WCS-TABLE                        the wavelength [um] of every plane
    SEGMENTS   bintable              one row per segment: label, file, wavelength range, the planes it fills,
                                     scale and its error, native spaxel scale, placement (identity | regridded),
                                     lattice offset, correlation
    OVERLAPS   bintable              one row per neighbouring pair: crossover, overlap, ratio window, ratio and
                                     its error, planes and spaxels measured, the level of the field in both
                                     segments, the shift of b against a [arcsec east, north] with its error,
                                     the position of the shared core [pix] and the size of the window
    SEGSPEC    bintable              every segment over one region of the field all segments cover, as measured
                                     (SOURCE: the excess over the level of COMMON; or the plain sum of COMMON)
    OVLSPEC    bintable              per pair, both segments over the pair's measurement region, as measured
    OVLIMG     f4 (2 pairs, ny, nx)  per pair, both segments collapsed over the ratio window
    OVLREG     u1 (pairs, ny, nx)    per pair, the measurement region
    OVLCORE    u1 (2 pairs, ny, nx)  per pair, the bright core of each segment inside the window of the registration
    COMMON     u1 (ny, nx)           the field all segments cover
    SOURCE     u1 (ny, nx)           the region of it that SEGSPEC is taken over

PRIMARY carries the keywords of the reference cube with the band keywords set
to what the joint cube is (``MULTIPLE``; channels ``1234``) and the JWFJC*
record of the stitch. ``qa_cube_stitch`` (:mod:`jwstflow.qa.steps.cube_stitch`)
draws the records; the stock cube figures read the cube like any other.

The package, by module:

* :mod:`.grid` -- :class:`SpatialGrid`, :func:`joint_grid`, the grid as ``cube_build`` parameters;
* :mod:`.regrid` -- :class:`Placement`: copying on a shared lattice, exact-overlap resampling otherwise;
* :mod:`.segments` -- reading the cubes; the joint grid of a set of files;
* :mod:`.measure` -- footprints, regions, region spectra, the registration of two segments;
* :mod:`.product` -- the ``*_s3dcomb.fits`` product;
* :mod:`.steps` -- ``stitch_cubes``;
* :mod:`.matched` -- ``cube_build_matched`` (a subclass of the official step: importing it imports jwst).

Heavy imports stay inside the functions.
"""

from .grid import (
    LATTICE_TOLERANCE_PIX,
    SpatialGrid,
    data_bounds,
    joint_grid,
    reference_index,
)
from .measure import (
    footprint,
    image_shift,
    pair_shift,
    region_spectrum,
    rim,
    shared_window,
    source_region,
)
from .product import (
    JOINT_SUFFIX,
    OVERLAP_COLUMNS,
    SEGMENT_COLUMNS,
    read_joint_records,
    region_flux,
    write_joint_cube,
)
from .regrid import IDENTITY, NO_DATA_DQ, REGRIDDED, Placement, overlap_matrix
from .segments import CubeSegment, joint_grid_of, read_segment, segment_label
from .steps import StitchCubes

__all__ = [
    "IDENTITY",
    "JOINT_SUFFIX",
    "LATTICE_TOLERANCE_PIX",
    "NO_DATA_DQ",
    "OVERLAP_COLUMNS",
    "REGRIDDED",
    "SEGMENT_COLUMNS",
    "CubeSegment",
    "Placement",
    "SpatialGrid",
    "StitchCubes",
    "data_bounds",
    "footprint",
    "image_shift",
    "joint_grid",
    "joint_grid_of",
    "overlap_matrix",
    "pair_shift",
    "read_joint_records",
    "read_segment",
    "reference_index",
    "region_flux",
    "region_spectrum",
    "rim",
    "segment_label",
    "shared_window",
    "source_region",
    "write_joint_cube",
]
