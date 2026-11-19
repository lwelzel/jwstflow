"""The PSF-cube product contract: spectrally sub-sampled instrument PSFs for IFU cubes.

Forward-modeling an IFU observation -- rendering a model of the target (a
disk, an envelope) and comparing it with an observed ``*_s3d.fits`` cube --
needs the instrument PSF at the observed wavelengths, sampled on the model's
own pixel grid. This module fixes one file layout for such products so that
the generating step (``psf_cube``), QA and downstream modeling code all speak
the same product:

    <base>_psfcube.fits           derived from the s3d cube it matches
      PRIMARY                     header copied from the source cube (instrument keys
                                  travel along) + JWFPSF* provenance keywords
      PSF       f4 (nw, ny, nx)   PSF slices, each centred on the geometric array
                                  centre ((ny-1)/2, (nx-1)/2); slice sums record the
                                  in-field energy fraction (stpsf ``normalize='first'``)
      WAVETAB   bintable          one row per slice: wavelength_um, psf_sum,
                                  fwhm_arcsec (measured on the slice)

Producers build a :class:`PsfCubeProduct` and call :meth:`PsfCubeProduct.write`;
consumers call :meth:`PsfCubeProduct.read` directly, or :func:`psf_from_stage`
to locate the product of a companion stage that matches a cube's instrument
configuration. :meth:`PsfCubeProduct.at` interpolates a kernel at any
wavelength and :meth:`PsfCubeProduct.convolve` convolves a whole model cube,
so modeling code never re-implements the interpolation or the centring rules.

The design choices (docs/psf_cubes.md carries the full rationale):

* **Wavelengths are sub-sampled.** The PSF varies smoothly with wavelength
  (FWHM ~ lambda/D), so slices at log-spaced wavelengths whose adjacent ratio
  is at most ``1 + max_fractional_step`` (default 2%) bound the FWHM error of
  the nearest slice to ~1% and of linear interpolation to O(step^2/8) ~ 1e-4.
* **The product is a distance-independent instrument library.** Slices live
  on a fixed, fine angular grid -- the cube's native spaxel scale divided by
  ``oversample`` (default 4), over a fixed angular field -- and know nothing
  about any particular model of the target. Scaling onto a model's angular
  grid (e.g. the 256 px / +-300 AU rendering convention,
  :func:`model_pixel_scale`) is the *consumer's* single resampling step,
  :meth:`PsfCubeProduct.resampled`, so many models at many trial distances
  reuse one library.
* **Orientation stays instrument-native** (``frame='ideal'``: array +y along
  the aperture ideal +y axis, the frame both engines produce), so no rotation
  ever resamples the stored slices; the sky position angle of that axis
  (``PA_APER``, else ``ROLL_REF + V3I_YANG``) is recorded in ``JWFPSFPA`` and
  :meth:`PsfCubeProduct.resampled` folds the rotation to the sky frame of
  ``skyalign`` cubes (north up, east left) into the same interpolation as the
  scale change. ``frame='sky'`` rotates at generation instead, for direct
  overlay on skyalign cubes. Orientation matters most for MIRI MRS, whose
  empirical broadening is anisotropic (a slice-width box along beta).
* **Engines are pluggable**: :class:`StpsfEngine` drives stpsf's IFU mode
  (``pip install 'jwstflow[psf]'`` plus the stpsf data files) and
  :class:`GaussianPsfEngine` is an analytic, offline approximation that also
  powers the tests.

The package, by module:

* :mod:`.grid` -- the spatial grid and the wavelength sub-sampling;
* :mod:`.instrument` -- the instrument configuration and geometry of a cube,
  the PSF widths to expect;
* :mod:`.engines` -- :class:`StpsfEngine` and :class:`GaussianPsfEngine`;
* :mod:`.product` -- :class:`PsfCubeProduct` and the lookup of products;
* :mod:`.steps` -- the ``psf_cube`` step (its figure, ``qa_psf_cube``, is in
  :mod:`jwstflow.qa.steps.psf`).

Everything imports its heavy dependencies lazily so the package stays
importable without astropy/scipy/stpsf.
"""

from .engines import GaussianPsfEngine, StpsfEngine, stpsf_version
from .grid import (
    MODEL_FOV_AU,
    MODEL_NPIX,
    PsfGrid,
    model_pixel_scale,
    subsample_wavelengths,
)
from .instrument import (
    JWST_DIAMETER_M,
    MRS_SUBBAND,
    NIRSPEC_IFU_BROADENING_SIGMA,
    CubeGeometry,
    IfuConfig,
    approx_fwhm_arcsec,
    diffraction_fwhm_arcsec,
    read_cube_geometry,
)
from .product import (
    PSFCUBE_SUFFIX,
    PsfCubeProduct,
    find_psf_product,
    measure_fwhm_arcsec,
    psf_from_stage,
    rotate_cube_to_sky,
)
from .steps import PsfCube

__all__ = [
    "JWST_DIAMETER_M",
    "MODEL_FOV_AU",
    "MODEL_NPIX",
    "MRS_SUBBAND",
    "NIRSPEC_IFU_BROADENING_SIGMA",
    "PSFCUBE_SUFFIX",
    "CubeGeometry",
    "GaussianPsfEngine",
    "IfuConfig",
    "PsfCube",
    "PsfCubeProduct",
    "PsfGrid",
    "StpsfEngine",
    "approx_fwhm_arcsec",
    "diffraction_fwhm_arcsec",
    "find_psf_product",
    "measure_fwhm_arcsec",
    "model_pixel_scale",
    "psf_from_stage",
    "read_cube_geometry",
    "rotate_cube_to_sky",
    "stpsf_version",
    "subsample_wavelengths",
]
