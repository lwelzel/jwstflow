"""``cube_build_matched``: the official ``cube_build`` on one spatial grid for every band.

Defined as a subclass of the official step, so it runs like any pipeline step of a workflow
(associations, CRDS references, official ``_s3d`` products). Why the joint cube wants it is
described in :mod:`jwstflow.cubes.stitching`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from jwst.step import CubeBuildStep

from .segments import joint_grid_of

log = logging.getLogger(__name__)


class CubeBuildMatched(CubeBuildStep):
    """The official cube_build with the spatial grid prescribed: every band cube on the joint grid.

    ``cube_build`` gives each band its own grid -- its own tangent point,
    field and, for the MRS, spaxel scale -- so the band cubes of one
    observation never share a pixel lattice (the two NIRSpec gratings of one
    visit come out a fraction of a spaxel apart, the twelve MRS bands at
    four scales with tangent points up to half an arcsecond apart). Joining
    them along wavelength would mean resampling finished cubes a second time.

    This step builds the cubes on the joint grid instead: the grid that
    ``stitch_cubes`` would choose for the cubes found in ``grid_from`` (the
    stage directory of the ordinary band cubes; ``grid_pattern`` selects
    among them) -- the finest spaxel scale among them over the union of
    their fields, on the pixel lattice of the finest. It passes that grid to
    the official step as ``scalexy``, ``ra_center``, ``dec_center``,
    ``nspax_x``, ``nspax_y`` (and ``cube_pa``), and everything else is
    ``cube_build`` as it stands: the detector pixels of every band are
    drizzled once, straight onto the common grid -- as the official
    multi-band cubes are built, but with the bands kept apart, so that
    ``stitch_cubes`` can choose the crossovers and measure the overlap
    ratios. Every other parameter of ``cube_build`` (``weighting``,
    ``output_type``, ``channel``, ...) is accepted as usual.

    Feed it the exposures the ordinary cubes were built from, in the state
    they were built from: the ``*_crf.fits`` files of ``calwebb_spec3``
    (outlier flags and a master-background subtraction are in them), after
    the same ``pixel_replace`` if the ordinary cubes had one. The products
    are named like the ordinary cubes and live in
    ``stage3/cube_build_matched/``.
    """

    class_alias = "cube_build_matched"

    spec = """
        grid_from = string(default=None)             # directory of the band cubes whose grids define the common grid
        grid_pattern = string(default="*_s3d.fits")  # which files of grid_from count
    """

    def process(self, input_data):
        if not self.grid_from:
            raise ValueError("cube_build_matched needs `grid_from`: the directory of the band cubes that define the grid "
                             "(e.g. ${run_dir}/stage3/calwebb_spec3)")
        cubes = sorted(Path(self.grid_from).glob(self.grid_pattern))
        if not cubes:
            raise FileNotFoundError(f"no {self.grid_pattern} in {self.grid_from}: run the stage that builds the band "
                                    "cubes first (and declare depends_on)")
        grid = joint_grid_of(cubes)
        for name, value in grid.cube_build_parameters().items():
            setattr(self, name, value)
        self.coord_system = "skyalign"            # the orientation is part of the grid (cube_pa when rotated)
        log.info("cube_build_matched: grid of %d cube(s) in %s: %s", len(cubes), self.grid_from, grid.describe())
        return super().process(input_data)
