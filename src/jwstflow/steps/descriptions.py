"""What each built-in jwst alias does (hand-written, so nothing is imported).

``jwstflow steps`` must stay fast and work without ``jwst`` in the parent
process, so the aliases of :data:`~jwstflow.steps.base.BUILTIN_ALIASES` are
documented here by hand as ``name -> (short, detailed)``. Custom steps need no
such table: their docstring is their description (first paragraph = short,
rest = detailed), read straight from the source file.

The detailed texts sketch what the official pipeline or step computes, in
execution order, not every instrument-specific branch; the authoritative
reference is https://jwst-pipeline.readthedocs.io.
"""

from __future__ import annotations

BUILTIN_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    # -- pipelines ----------------------------------------------------------
    "detector1": (
        "Stage 1: turns the raw non-destructive ramps into count-rate images "
        "(uncal -> rate/rateints).",
        "Applies the detector-level corrections shared by all instruments and\n"
        "fits the up-the-ramp accumulation of every pixel. Per exposure, roughly:\n"
        "\n"
        "1. initialise the data-quality arrays from the bad-pixel mask (dq_init)\n"
        "   and flag saturated groups (saturation);\n"
        "2. remove electronic signatures: superbias and reference-pixel drifts\n"
        "   for the NIR detectors (superbias, refpix), reset/RSCD/firstframe/\n"
        "   lastframe effects for MIRI;\n"
        "3. correct classical non-linearity (linearity) and subtract the dark\n"
        "   current (dark_current);\n"
        "4. flag cosmic-ray hits as jumps between consecutive groups (jump) and\n"
        "   fit a slope to the surviving ramp segments of every pixel (ramp_fit).\n"
        "\n"
        "The slope images are written as *_rate.fits (integrations averaged) and\n"
        "*_rateints.fits (one image per integration), in DN/s.",
    ),
    "image2": (
        "Stage 2 imaging: assigns the WCS, flat-fields and flux-calibrates each "
        "rate image (rate -> cal).",
        "Per exposure: attach the pixel->sky WCS transforms (assign_wcs),\n"
        "subtract a background exposure when the association provides one\n"
        "(bkg_subtract), divide by the flat field (flat_field), and convert DN/s\n"
        "to surface brightness in MJy/sr (photom) -> *_cal.fits. A resampled\n"
        "single-exposure preview (*_i2d.fits) is written for quick looks.",
    ),
    "spec2": (
        "Stage 2 spectroscopy: WCS + wavelength solution, 2-D extraction and "
        "flux calibration per exposure (rate -> cal).",
        "The spectroscopic sibling of image2; the sub-steps run depend on the\n"
        "exposure type. Per exposure, roughly:\n"
        "\n"
        "1. assign_wcs attaches the transforms detector pixel -> sky + wavelength;\n"
        "2. backgrounds are handled: bkg_subtract (nodded/offset exposures),\n"
        "   imprint_subtract and msa_flagging (NIRSpec IFU/MOS);\n"
        "3. extract_2d cuts out slit stamps (NIRSpec, WFSS), srctype decides\n"
        "   POINT vs EXTENDED (steering later corrections), wavecorr refines\n"
        "   NIRSpec wavelength zero-points;\n"
        "4. flat_field applies the flats, followed by the geometric corrections:\n"
        "   straylight and fringe (MIRI MRS), pathloss and barshadow (NIRSpec);\n"
        "5. photom converts to MJy/sr -> *_cal.fits; depending on the mode a\n"
        "   resampled preview (*_s2d) and a quick extraction (*_x1d) follow.",
    ),
    "image3": (
        "Stage 3 imaging: aligns, background-matches and combines the exposures "
        "of an association into one mosaic (cal -> i2d).",
        "Works on an association of calibrated exposures:\n"
        "\n"
        "1. tweakreg refines the relative (and, against GAIA, absolute)\n"
        "   alignment by matching point sources;\n"
        "2. skymatch measures and equalises the sky level between exposures;\n"
        "3. outlier_detection compares each exposure against the median of all\n"
        "   of them and flags cosmic rays / transients in the DQ arrays;\n"
        "4. resample drizzles everything onto one undistorted grid -> *_i2d.fits,\n"
        "   and source_catalog measures the detected sources (*_cat.ecsv).",
    ),
    "spec3": (
        "Stage 3 spectroscopy: combines the calibrated exposures of an "
        "association into cubes / 2-D spectra and extracts 1-D spectra.",
        "Works on an association of *_cal files (the dithers/nods of a target):\n"
        "\n"
        "1. master_background builds one background spectrum and subtracts it\n"
        "   (when background exposures or nods are available);\n"
        "2. outlier_detection flags pixels that deviate between the dithers;\n"
        "3. the exposures are combined: cube_build for IFU data (-> *_s3d\n"
        "   spectral cubes), resample_spec for slit spectra (-> *_s2d);\n"
        "4. extract_1d extracts 1-D spectra (-> *_x1d) and combine_1d merges\n"
        "   them where applicable (-> *_c1d); MIRI MRS additionally gets the\n"
        "   spectral_leak correction.",
    ),
    "tso3": (
        "Stage 3 time series: per-integration outlier flagging, then photometry "
        "or spectral extraction into light curves.",
        "Works on the *_calints (per-integration) products of a time-series\n"
        "observation. outlier_detection flags deviant pixels across the stack of\n"
        "integrations; imaging TSOs then get aperture photometry per integration\n"
        "(tso_photometry -> *_phot.ecsv), spectroscopic TSOs a 1-D spectrum per\n"
        "integration (extract_1d -> *_x1dints) and a band-averaged white-light\n"
        "curve (white_light -> *_whtlt.ecsv).",
    ),
    "coron3": (
        "Stage 3 coronagraphy: KLIP PSF subtraction of the science exposures "
        "using the PSF reference exposures.",
        "Works on an association of science and PSF-reference *_calints\n"
        "exposures: stack_refs collects the PSF references, align_refs registers\n"
        "them onto the science frames, klip builds an optimal PSF model per\n"
        "science integration (Karhunen-Loeve decomposition) and subtracts it\n"
        "(-> *_psfsub), then outlier_detection and resample combine the\n"
        "PSF-subtracted integrations into the final image (*_i2d.fits).",
    ),
    "dark": (
        "Stage 1 processing of dark exposures: the initial detector corrections "
        "only, to build dark reference files.",
        "Runs the leading detector1 steps (dq_init, saturation and the\n"
        "electronic corrections appropriate for the detector) on raw dark\n"
        "exposures and stops before dark subtraction and ramp fitting. The\n"
        "output ramps are input for building dark-current reference files, not\n"
        "science products.",
    ),
    # -- individual steps ---------------------------------------------------
    "assign_wcs": (
        "Attaches the WCS: the transforms from detector pixels to sky "
        "coordinates (and wavelength, for spectra).",
        "Builds a chain of transforms (a gWCS object) from the CRDS distortion\n"
        "and spectral reference files plus the telescope pointing, and stores it\n"
        "in the file. Afterwards every pixel maps to RA/Dec -- and to wavelength\n"
        "for spectroscopic modes, per slit for NIRSpec. All later resampling,\n"
        "cube building and extraction consume this WCS, not raw FITS keywords.",
    ),
    "extract_1d": (
        "Collapses 2-D spectra or IFU cubes into 1-D flux-vs-wavelength tables "
        "(x1d).",
        "For slit-like data: sums the flux over the source aperture wavelength\n"
        "bin by wavelength bin and subtracts a background estimated from\n"
        "adjacent regions. For IFU cubes: aperture photometry on every\n"
        "wavelength plane -- a circular aperture at the source (growing with\n"
        "wavelength for point sources) with an annulus background, then an\n"
        "aperture correction to total flux. Writes an EXTRACT1D table\n"
        "(WAVELENGTH, FLUX, FLUX_ERROR, surface brightness, background)\n"
        "-> *_x1d.fits.",
    ),
    "cube_build": (
        "Resamples IFU exposures onto a rectified RA x Dec x wavelength "
        "spectral cube (s3d).",
        "Maps every detector pixel of the input IFU exposures through its WCS to\n"
        "(RA, Dec, wavelength) and accumulates the flux onto a regular 3-D voxel\n"
        "grid with a distance-weighted kernel (emsm/msm, or drizzle).\n"
        "Overlapping dithers and exposures are averaged with their weights;\n"
        "errors and DQ are propagated. One cube per band/grating -- or one\n"
        "combined cube, as configured -- is written as *_s3d.fits.",
    ),
    "resample_spec": (
        "Drizzles 2-D slit spectra onto a rectified wavelength x slit-position "
        "grid (s2d).",
        "Uses the WCS of each input to drizzle the slit exposures onto a common\n"
        "rectified grid -- wavelength along one axis, position along the slit on\n"
        "the other -- combining dithers/nods with their weights into one\n"
        "undistorted 2-D spectrum per source (*_s2d.fits).",
    ),
    "resample": (
        "Drizzles imaging exposures onto a common undistorted sky grid, "
        "combining dithers into one image (i2d).",
        "Projects each input image through its WCS onto the output pixel grid\n"
        "with the drizzle algorithm: flux is distributed proportionally to pixel\n"
        "overlap and combined weighted by the exposure weight maps, removing the\n"
        "optical distortion. DQ-flagged pixels are left out, so previously\n"
        "flagged outliers disappear from the combined *_i2d.fits.",
    ),
    "outlier_detection": (
        "Flags pixels (cosmic rays, transients) that deviate between the "
        "dithered exposures of an association.",
        "Resamples all input exposures onto a common grid, builds their\n"
        "(weighted) median image, projects that median back into each exposure's\n"
        "own frame and compares: pixels deviating by more than the noise-scaled\n"
        "thresholds are flagged OUTLIER in that exposure's DQ. IFU and TSO data\n"
        "use variants that difference along the exposure/integration stack\n"
        "instead of resampling. Downstream steps (resample, cube_build,\n"
        "extract_1d) then ignore the flagged pixels.",
    ),
    "master_background": (
        "Combines background exposures/nods into one master background spectrum "
        "and subtracts it from the science data.",
        "Collects the 1-D background spectra (from dedicated background\n"
        "exposures or nod positions), combines them into one master background\n"
        "spectrum, expands that spectrum back into the 2-D frame of every\n"
        "science exposure using each pixel's wavelength, and subtracts it.\n"
        "NIRSpec MOS uses a dedicated variant inside spec2\n"
        "(master_background_mos).",
    ),
    "photom": (
        "Converts count rates to physical units (MJy/sr) using the CRDS "
        "photometric calibration.",
        "Multiplies the data by the scalar conversion factor for the\n"
        "detector/filter (PHOTMJSR) and, for spectroscopic modes, by the\n"
        "wavelength-dependent relative response from the photom reference file;\n"
        "attaches the pixel-area map. Afterwards SCI is surface brightness in\n"
        "MJy/sr (BUNIT updated), so fluxes summed over apertures come out in Jy.",
    ),
    "jump": (
        "Flags cosmic-ray hits in the up-the-ramp groups via two-point "
        "differences.",
        "For every pixel, computes the differences between consecutive groups of\n"
        "each integration and compares them with the expected noise; a\n"
        "difference beyond rejection_threshold sigmas marks that group JUMP_DET\n"
        "in GROUPDQ (ramp_fit later splits the ramp there). Large events can\n"
        "additionally trigger neighbour and snowball/shower flagging.",
    ),
    "ramp_fit": (
        "Fits a slope to every pixel's ramp segments -> count-rate images "
        "(rate, rateints).",
        "Splits each pixel's ramp at the jumps and saturated groups flagged\n"
        "earlier, fits each clean segment with weighted ordinary least squares,\n"
        "and combines the segment slopes weighted by their inverse variance.\n"
        "Writes the per-integration slopes (*_rateints.fits) and their\n"
        "exposure-level average (*_rate.fits) in DN/s, with Poisson and\n"
        "read-noise variance arrays.",
    ),
    "clean_flicker_noise": (
        "Removes correlated 1/f read noise (banding) from the ramps before "
        "ramp fitting.",
        "Works on the group images near the end of detector1 (generalising\n"
        "NSClean): masks the illuminated and flagged pixels to keep only\n"
        "background, models the remaining correlated read noise per amplifier\n"
        "along the slow read direction (median or FFT fit), and subtracts that\n"
        "model from every group.",
    ),
}
