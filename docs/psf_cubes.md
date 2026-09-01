# PSF cubes

`psf_cube` generates, per observed IFU cube, the instrument PSF your model of
the target must be convolved with before it can be compared to that cube:
one `*_psfcube.fits` per `*_s3d.fits`, spectrally sub-sampled, on the model's
own pixel grid, oriented like the cube. `qa_psf_cube` plots every product.
Supported today: **NIRSpec IFU** and **MIRI MRS**.

```yaml
  - step: psf_cube                       # -> stage4/psf_cube
    inputs: [{stage: calwebb_spec3, pattern: "*_s3d.fits"}]
    parameters:
      distance_pc: 190.0                 # the only required knob: your target's distance

  - step: qa_psf_cube                    # -> qa/qa_psf_cube
    inputs: [{stage: psf_cube, pattern: "*_psfcube.fits"}]
    tags: [qa]
```

Downstream, in your modeling code:

```python
from jwstflow.psf import PsfCubeProduct

psf = PsfCubeProduct.read("jw01751-o010_t005_miri_ch1-short_psfcube.fits")
convolved = psf.convolve(model_cube, model_wavelengths_um)   # (nw, 256, 256) in, same out
kernel = psf.at(5.3)                                         # one interpolated slice
```

## Why it lives where it lives

The generation engine is [stpsf](https://stpsf.readthedocs.io) (formerly
WebbPSF) -- a heavy optional dependency with its own reference-data download.
That argues for a contributed package; but a PSF product is only useful if
*everything* agrees on its layout: the generating step, the QA step, and the
modeling code of every program (MIDAS and JOYS+ both consume cubes of disks).
Shared product layouts are exactly what `jwstflow` core fixes for the
ecosystem (the mask contract in `jwstflow.masks`, stitching in
`jwstflow.stitching`, `jwstflow.apcorr` -- see "Shared step contracts" in the
guide). So the split follows the graduation ladder:

* **the contract and every consumer** (`jwstflow.psf`: product I/O,
  wavelength interpolation, convolution, the offline Gaussian engine, and
  `qa_psf_cube`) ship with the core and add **no dependencies** -- the base
  install is unchanged;
* **the stpsf engine** hides behind the `psf` extra and a lazy import:

  ```bash
  uv pip install "jwstflow[psf]"       # adds stpsf
  ```

  plus the stpsf reference files: set `STPSF_PATH` (an `.env` line in your
  project root works) or let stpsf download them on first use. Without the
  extra, `psf_cube` fails with exactly this instruction, and
  `engine: gaussian` still works.

If the engine ever warrants its own release cadence, it can graduate into a
`jwstflow-psf` contributed package without breaking anything: the YAML name,
the product contract and the consumers all stay in place.

## The product

```
<cube base>_psfcube.fits
  PRIMARY               copied from the source cube (instrument keys travel along)
                        + JWFPSF* provenance: engine, stpsf version, method, OPD,
                        broadening, grid, pixel scale, applied PA, distance
  PSF   f4 (nw, ny, nx) PSF slices; each centred on the geometric array centre
                        ((n-1)/2 -- a half-pixel for even n, matching stpsf/poppy)
  WAVETAB               wavelength_um, psf_sum (in-field energy fraction),
                        fwhm_arcsec (measured) per slice
```

Slices are stored **as computed** (stpsf `normalize='first'`: unit flux enters
the telescope, so a slice sum is the fraction landing inside the field --
QA-relevant, and the wing loss an extended-source aperture correction cares
about). `PsfCubeProduct.at()/.convolve()` renormalize each slice to unit sum
by default, so convolution conserves surface brightness; pass
`normalized=False` to keep the absolute normalization. `convolve()` shifts
the kernel centre to the origin analytically (Fourier shift theorem), so even
grids introduce **no half-pixel displacement** -- pinned by a test.

`find_psf_product` / `psf_from_stage` locate the product matching a cube's
instrument configuration (same `INSTRUMENT_KEYS` matching as the mask
contract), so a consuming step never hardcodes file names.

## Spectral sub-sampling

The PSF core scales as lambda/D, so storing every cube plane (~1000 per
grating) would be ~50x redundant. Slices are computed at **log-spaced
wavelengths** covering the cube exactly, adjacent wavelengths within
`max_fractional_step` (default 2%):

| product | range [um] | slices at 2% |
|---|---|---|
| MRS 1A | 4.90-5.74 | 9 |
| MRS 4C | 24.4-28.6 | 9 |
| NIRSpec G235H/F170LP | 1.66-3.17 | 34 |
| NIRSpec G395H/F290LP | 2.87-5.27 | 32 |

Between stored slices the nearest-slice FWHM error is at most step/2 (1%),
and the **linear interpolation** `at()`/`convolve()` perform cuts the
residual to O(step^2/8) ~ 1e-4 -- far below the fidelity of the PSF model
itself. Endpoints are included, so interpolation never extrapolates.
`n_wavelengths` overrides the count (a PRISM cube at 2% would want
`max_fractional_step: 0.05` or `method: fast`).

## The spatial grid

Model slices in this ecosystem are rendered at 1 pc with 256x256 pixels
spanning -300..+300 AU. At 1 pc, 1 AU subtends exactly 1 arcsec, so for a
target at distance d the rendered pixel is

    (600 AU / 256 px) / d  =  2.34375 / d  arcsec

-- at the ~190 pc of Cha I: 12.3 mas pixels, a 3.2" field. That is a lucky
(or not so lucky) match to the NIRSpec IFU field (3.0") and MRS ch1 (~3.4"),
and 4-10x finer than any PSF FWHM in range, so **the PSF is generated
directly on that grid** (`grid: model`, the default; `distance_pc` sets it)
and convolution needs no resampling step -- resampling kernels is where flux
conservation quietly dies. In stpsf terms the model grid *is* the detector
plane (`oversample=1`); at >=4 px per FWHM the difference between
pixel-integrated and point-sampled kernels is negligible next to the
IFU broadening.

`grid: native` uses the cube's own spaxel scale instead (PSF photometry on
the cube itself), and an explicit `pixelscale_arcsec` overrides both.
`npix`/`fov_au` reshape the model convention if yours differs.

## What the slices contain, and orientation

stpsf's IFU mode is used with its **as-built broadening** left on: for MIRI
MRS the empirical model tuned to commissioning cubes (Argyriou et al. 2023;
Law et al. 2023: FWHM = 0.033 lambda + 0.106") -- a Gaussian along the
along-slice axis plus a **slice-width boxcar across slices**, so the kernel
is anisotropic -- and for NIRSpec a 0.05"-sigma Gaussian. The slices
therefore approximate the PSF **as realised in drizzled s3d cubes**, which is
what a model compared against s3d planes must be convolved with; the bare
optical PSF is available with `broadening: none`.

Because the MRS kernel is anisotropic, orientation matters. stpsf computes in
the instrument frame; the step rotates every slice to the **sky frame of
`skyalign` cubes (north up, east left)** using the aperture position angle
from the cube header (`PA_APER`, else `ROLL_REF + V3I_YANG`;
`position_angle_deg` overrides, and the angle used is recorded in
`JWFPSFPA`). NIRSpec's IFU-align output rotation is disabled so both
instruments share the "array +y = aperture ideal +y" convention that the
rotation assumes. Fidelity notes: the MRS aperture ideal frame is itself an
approximation (stpsf averages the skewed slice geometry), and parity of the
faint speckle pattern is not guaranteed -- treat sub-degree PA effects and
speckle-level structure as beyond this product's fidelity. If your model is
rendered in a frame other than sky (e.g. disk major axis along x), rotate
the *model* to sky before convolving, as you would for the data comparison.

## Engines, methods, OPD

| parameter | choices | notes |
|---|---|---|
| `engine` | `stpsf` (default) / `gaussian` | `gaussian` is an offline analytic stand-in at the empirical FWHM -- no spikes/rings; dry-runs, tests |
| `method` | `exact` (default) / `fast` | `exact` = one full `calc_psf` per wavelength; `fast` = stpsf's `calc_datacube_fast` (one pupil propagation for all wavelengths, ~100x faster) + stpsf's broadening applied per slice; assumes a wavelength-independent exit-pupil wavefront |
| `opd` | `default` / `by_date` / file | `by_date` fetches the measured in-flight wavefront nearest DATE-OBS from MAST (network!); `default` is offline |
| `broadening` | `default` / `gaussian` / `none` | `default` = empirical (MRS) / Gaussian (NIRSpec) |

Generation cost with `exact` is minutes per band product (and band products
run in parallel like any other per-file stage); results are checkpointed
like every jwstflow task, so reruns are free.

## Testing

`tests/test_psf.py` runs entirely offline: the Gaussian engine exercises the
full product path (generation -> write -> read -> interpolate -> convolve ->
QA), the geometry/rotation/centring conventions are pinned numerically
(rotation sign, the (n-1)/2 kernel centre, flux conservation, zero
displacement), and a stub of the verified stpsf 2.x IFU API pins how
`StpsfEngine` must drive the real package -- in particular that the pixel
scale is applied *after* band selection (stpsf resets it on any IFU aperture
change) and that the broadened `DET_DIST` plane is the one stored.
