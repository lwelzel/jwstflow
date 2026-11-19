"""IFU cubes: what steps that read or write cubes build on, and the cube features shipped with jwstflow.

The toolkit, importable from here (``from jwstflow.cubes import celestial_wcs, write_mask_product``):

* :mod:`.wcs` -- the celestial WCS and the wavelengths of a cube, regridding between cubes;
* :mod:`.morphology` -- NaN-aware smoothing, dilation, cleaning a mask down to one blob;
* :mod:`.instrument` -- matching a product to the instrument configuration of a cube;
* :mod:`.masks` -- the mask-product contract: wavelength-resolved spatial masks in one FITS layout;
* :mod:`.apcorr` -- the CRDS aperture-correction reference (MIRI MRS, NIRSpec IFU), evaluated per plane.

The features, each a package with its library, its product and its steps:

* :mod:`.psf` -- PSF cubes (``psf_cube``);
* :mod:`.clusters` -- bad spaxel clusters (``flag_spaxel_clusters``, ``propagate_cluster_flags``).
"""

from .apcorr import apcorr_factor, load_apcorr, mrs_apcorr_table, nrs_apcorr_table
from .instrument import INSTRUMENT_KEYS, matches_instrument
from .masks import (
    check_mask_matches,
    find_mask_product,
    mask_from_stage,
    read_mask_product,
    write_mask_product,
)
from .morphology import clean_blob, dilate, nan_smooth
from .wcs import (
    celestial_wcs,
    cube_wavelengths,
    pixel_map,
    resample_image,
    sample_mask,
    strip_wave_axis,
)

__all__ = [
    "INSTRUMENT_KEYS",
    "apcorr_factor",
    "celestial_wcs",
    "check_mask_matches",
    "clean_blob",
    "cube_wavelengths",
    "dilate",
    "find_mask_product",
    "load_apcorr",
    "mask_from_stage",
    "matches_instrument",
    "mrs_apcorr_table",
    "nan_smooth",
    "nrs_apcorr_table",
    "pixel_map",
    "read_mask_product",
    "resample_image",
    "sample_mask",
    "strip_wave_axis",
    "write_mask_product",
]
