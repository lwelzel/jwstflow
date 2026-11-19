"""Bad spaxel clusters: find them in per-dither cubes, flag the detector pixels behind them.

The problem this solves. Combined IFU cubes (``calwebb_spec3``) sometimes show
compact clusters of wildly deviant spaxels over a few wavelength planes --
extreme outliers in morphology as well as in value -- that the official
pipeline never flags. The reason is structural: ``outlier_detection`` for IFU
data (jwst >= 1.11) works in *detector* space and keeps, per detector pixel,
the **minimum** neighbour difference over *all* exposures, so it only catches
defects present in every dither (bad pixels missing from the mask); anything
confined to one dither survives by design. ``cube_build``'s drizzle then
averages the dithers without any clipping, so the artefact lands in the cube,
diluted by the number of dithers but not removed.

The cure is to go back to the exposure the cluster comes from. The chain,
three steps that any workflow can insert between stage 2 and stage 3:

1. **per-dither cubes** -- the official ``cube_build`` step on one association
   per dither (``group_by: [..., PATT_NUM]``), so every dither has its own
   cube (``stage3/cube_build-<variant>/``);
2. :class:`FlagSpaxelClusters` (``flag_spaxel_clusters``) -- for every region
   named in the workflow (sky position, search aperture, wavelength window),
   decides *which dither cubes* carry the cluster and *which spaxels* inside
   the aperture are deviant, by comparing each dither with the others on the
   same sky grid; writes one ``*_clustermask.fits`` product per dither cube;
3. :class:`PropagateClusterFlags` (``propagate_cluster_flags``) -- evaluates
   the cal files' WCS to find the detector pixels whose (RA, Dec, wavelength)
   fall on flagged spaxels and marks them ``DO_NOT_USE | OUTLIER`` in DQ;
   the DQ-edited copies (official ``_cal`` names, JWFCL* header record) feed
   every downstream stage, so the clusters never reach the science cubes.

``qa_spaxel_clusters`` (:mod:`jwstflow.qa.steps.clusters`) draws every dither
at the affected wavelengths from the products alone.

The product (``<cube base>_clustermask.fits``) follows the mask contract of
:mod:`jwstflow.cubes.masks` (MASK/CONT/CONTIMG on the cube's own grid) and adds:

    WAVES     f8 (nw,)         wavelength [um] of every cube plane
    REGIONS   bintable         one row per region evaluated on this cube (see REGION_COLUMNS)
    SLICES    f4 (ns, ny, nx)  EXTVER = region row + 1: the cube planes around the window
    REFS      f4 (ns, ny, nx)  the same planes as seen by the other dithers (median, resampled)
    SIGMA     f4 (ns, ny, nx)  the significance of the deviation from the closest other dither
    SLICEWAV  f8 (ns,)         wavelengths of the SLICES planes
    APER      u1 (ny, nx)      the region's search aperture on this grid

so a QA step never reopens a cube. Heavy imports stay inside the functions.

The package, by module:

* :mod:`.regions` -- the regions of the workflow: aperture, planes, dither and band of a cube;
* :mod:`.compare` -- one dither against the others;
* :mod:`.product` -- the ``*_clustermask.fits`` product;
* :mod:`.detector` -- from flagged spaxels back to detector pixels;
* :mod:`.steps` -- ``flag_spaxel_clusters`` and ``propagate_cluster_flags``.
"""

from .compare import deviation_from_others, robust_sigma
from .detector import detector_pixels_in_masks
from .product import (
    CLUSTERMASK_SUFFIX,
    REGION_COLUMNS,
    find_cluster_products,
    read_cluster_product,
    write_cluster_product,
)
from .regions import (
    BAND_KEYS,
    MODE_ALL,
    MODE_AUTO,
    MODE_EXPLICIT,
    ClusterRegion,
    band_key,
    dither_of,
    matches_where,
    planes_in_window,
    region_aperture,
    region_centre,
)
from .steps import FlagSpaxelClusters, PropagateClusterFlags

__all__ = [
    "BAND_KEYS",
    "CLUSTERMASK_SUFFIX",
    "MODE_ALL",
    "MODE_AUTO",
    "MODE_EXPLICIT",
    "REGION_COLUMNS",
    "ClusterRegion",
    "FlagSpaxelClusters",
    "PropagateClusterFlags",
    "band_key",
    "detector_pixels_in_masks",
    "deviation_from_others",
    "dither_of",
    "find_cluster_products",
    "matches_where",
    "planes_in_window",
    "read_cluster_product",
    "region_aperture",
    "region_centre",
    "robust_sigma",
    "write_cluster_product",
]
