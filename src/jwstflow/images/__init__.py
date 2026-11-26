"""Images: what steps that read imaging products build on.

The toolkit, importable from here (``from jwstflow.images import read_image, band_of``):

* :mod:`.io` -- an imaging product (``*_i2d.fits``, ``*_cal.fits``) read into an :class:`Image`:
  SCI / ERR / WHT / CON / DQ, the celestial WCS, the pixel solid angle, the band; cutouts around
  a sky position that keep a WCS of their own;
* :mod:`.filters` -- the imaging filters of NIRCam, NIRISS and MIRI (pivot wavelength, width,
  edges; ``refdata/filters``), their throughput curves and the band average of a spectrum through
  them (synthetic photometry), and the diffraction-limited resolution at a wavelength.

One feature is built on it: :mod:`.astrometry` (``image_astrometry``), how well the images of a
run are registered to each other and to Gaia DR3. The stock figures of imaging products are QA
steps (``jwstflow.qa.steps.images``), the science of particular sources lives in contributed
packages.
"""

from .filters import Band, all_bands, band_average, band_of, diffraction_fwhm_arcsec, throughput
from .io import Image, read_image

__all__ = ["Band", "Image", "all_bands", "band_average", "band_of", "diffraction_fwhm_arcsec", "read_image",
           "throughput"]
