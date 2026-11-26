"""How well the images of a run are registered: to each other, and to Gaia DR3.

Science that compares the images of one source across filters -- where the dark lane of an
edge-on disk lies at 1, 4 and 21 um -- needs to know how well the mosaics agree on the sky.
``calwebb_image3`` aligns the exposures of one filter to each other and, where it finds Gaia
stars, to Gaia DR3 (``tweakreg``, ``abs_refcat``), but every filter separately: nothing aligns
the mosaics of two filters to each other, and the absolute step fails quietly where a field
holds few unsaturated Gaia stars. :class:`ImageAstrometry` (``image_astrometry``) measures what
came out, from the point sources the pipeline already found.

**Point sources.** The source catalogue ``image3``'s ``source_catalog`` writes next to every
mosaic (``<base>_cat.ecsv`` beside ``<base>_i2d.fits``) is read and cut, every cut in units of the
band's diffraction-limited FWHM (:attr:`jwstflow.images.Band.fwhm_arcsec`) so that one set of
numbers serves 1 and 21 um alike:

* a finite centroid (``sky_centroid``; the pixel centroid ``xcentroid``/``ycentroid``);
* compact: ``is_extended`` false (the catalogue's two concentration indices below the star
  thresholds of ``source_catalog``);
* ``S/N >= min_snr``, from the aperture-corrected total flux and its error
  (``aper_total_flux / aper_total_flux_err``; the isophotal flux where those are missing);
* isolated: no other catalogue source -- compact or not -- within ``isolation`` FWHM (a neighbour
  pulls a segment's centroid);
* a complete core: every pixel within ``core`` FWHM of the centroid has data (finite SCI, WHT > 0),
  which drops stars with saturated, blanked cores and sources cut by the edge of the mosaic.

The number of sources left after every cut is recorded per image.

**Matching**, of an image to the reference image and of an image to Gaia, in two passes. First the
bulk shift: every pair within ``search`` arcsec votes with its offset, and the shift is the median
offset of the largest group of votes that agree within the match radius -- so a misregistration
of a few tenths of an arcsec, larger than the match radius, is found rather than missed. Then
one-to-one matches: mutual nearest neighbours within ``match_radius`` FWHM (of the coarser band of
the pair) of the shifted positions. Offsets are measured on the positions as they are, image
minus reference: ``dra`` is the offset in right ascension times cos(dec), ``ddec`` in declination,
both in mas. Per comparison: the number matched, the median of each offset over the matches kept
after clipping at ``clip`` times the robust scatter (``1.4826 x MAD``; only with five matches or
more -- with fewer no match can be told an outlier), that scatter, the standard deviation, and the
error of the median ``1.2533 x sigma / sqrt(N)`` with sigma the larger of the two (the MAD of a few
matches collapses: of three it is the smaller of two differences); with fewer than ``min_matches``
matches the median is reported without scatter and error (NaN).

**The reference image** is the one named by ``reference`` (a band, e.g. ``F200W``, or a file
name), else the image with the most point sources (the bluer one on a tie): every other image is
compared with it.

**Gaia DR3** is queried once for the union of the images' footprints (:mod:`.gaia`: the ESA Gaia
archive through ``astroquery``, falling back to the STScI catalogue service that ``tweakreg``'s
``abs_refcat`` reads) and cached as a product of the stage (``<base>_gaiadr3.ecsv``), reused by
later runs whose footprint it covers. Positions are propagated with their proper motions from
the Gaia epoch (J2016.0) to each image's mid-exposure epoch; sources without a proper motion
(two-parameter solutions) are matched and recorded but, with ``gaia_pm_only`` (default), left out
of the median -- their position years later is uncertain by their unknown motion. A query that
fails is logged and the relative comparison goes on without it.

**The product** (``<base>_astrom.fits``, one per run; ``<base>`` is what the input names share,
else the target) is self-describing -- what ``qa_image_astrometry`` draws it from::

    PRIMARY   the reference, the cuts, the Gaia status and query, the conventions (JWFAS* keys)
    IMAGES    one row per input image: band, pivot wavelength, FWHM, pixel scale, epoch, the
              sources left after each cut, the S/N column used, whether it is the reference
    SUMMARY   one row per comparison (image against the reference image, image against Gaia):
              match radius, bulk shift, N matched / used, median, scatter, std, error [mas]
    MATCHES   one row per matched source: both ids, the image position, the offset [mas], S/N,
              whether it is in the median

The package, by module:

* :mod:`.catalogs` -- the catalogue of an image and its point sources;
* :mod:`.match` -- the two-pass match and the robust statistics of the offsets;
* :mod:`.gaia` -- the Gaia DR3 query, its cache and the propagation to an epoch;
* :mod:`.product` -- the ``*_astrom.fits`` product;
* :mod:`.steps` -- ``image_astrometry``.
"""

from .catalogs import CATALOG_SUFFIX, PointSources, catalog_path, observation_mjd, read_point_sources
from .gaia import GAIA_EPOCH, GAIA_SUFFIX, GaiaQueryError, load_or_query_gaia, propagate, query_gaia
from .match import Comparison, bulk_shift, compare, match_pairs, robust_offset, sky_offsets_mas
from .product import ASTROM_SUFFIX, write_astrometry_product
from .steps import ImageAstrometry

__all__ = [
    "ASTROM_SUFFIX",
    "CATALOG_SUFFIX",
    "GAIA_EPOCH",
    "GAIA_SUFFIX",
    "Comparison",
    "GaiaQueryError",
    "ImageAstrometry",
    "PointSources",
    "bulk_shift",
    "catalog_path",
    "compare",
    "load_or_query_gaia",
    "match_pairs",
    "observation_mjd",
    "propagate",
    "query_gaia",
    "read_point_sources",
    "robust_offset",
    "sky_offsets_mas",
    "write_astrometry_product",
]
