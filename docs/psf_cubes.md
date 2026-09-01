# PSF cubes

`psf_cube` generates, per observed IFU cube, a **distance-independent library
of the instrument PSF** your models of the target are convolved with before
they can be compared to that cube: one `*_psfcube.fits` per `*_s3d.fits`,
spectrally sub-sampled, on a fixed fine angular grid, in the instrument's own
frame. `qa_psf_cube` plots every product. Supported today: **NIRSpec IFU**
and **MIRI MRS**.

```yaml
  - step: psf_cube                       # -> stage4/psf_cube; the defaults just work
    inputs: [{stage: calwebb_spec3, pattern: "*_s3d.fits"}]

  - step: qa_psf_cube                    # -> qa/qa_psf_cube
    inputs: [{stage: psf_cube, pattern: "*_psfcube.fits"}]
    tags: [qa]
```

Downstream, in your modeling code -- the library is generated once, and each
model (at each trial distance) resamples it onto its own grid in one step:

```python
from jwstflow.psf import PsfCubeProduct, model_pixel_scale

library = PsfCubeProduct.read("jw01751-o010_t005_miri_ch1-short_psfcube.fits")
for d in trial_distances_pc:                        # e.g. your 256 px / +-300 AU rendering
    psf = library.resampled(model_pixel_scale(d), 256, to_sky=True)
    convolved = psf.convolve(model_cube_at(d), model_wavelengths_um)
```

`resampled()` performs the scale change and the rotation to the sky frame in
a *single* interpolation; `convolve()` then matches kernels to your model
wavelengths. `at()` gives one interpolated slice when that is all you need.

## Why a library, not a per-model product

The observation fixes the PSF (instrument configuration, wavelengths,
wavefront, orientation); the model fixes the grid (its distance and pixel
convention). Baking a model's distance into the PSF product would conflate
the two -- every trial distance would need its own generation run. So the
step knows nothing about your models: it emits the instrument PSF at a fixed
fine angular sampling, and distance scaling lives in the one resampling the
consumer does anyway. `model_pixel_scale(d)` (the `(600 AU / 256 px) / d`
arcsec convention of this ecosystem) stays available as a modeling-side
convenience for picking the target grid.

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
  resampling, wavelength interpolation, convolution, the offline Gaussian
  engine, and `qa_psf_cube`) ship with the core and add **no dependencies**
  -- the base install is unchanged;
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
                        broadening, sampling (oversample / native spaxel scale /
                        pixel scale / field), frame, recorded PA
  PSF   f4 (nw, ny, nx) PSF slices; each centred on the geometric array centre
                        ((n-1)/2 -- an integer pixel: the grid is rounded up to
                        odd, matching stpsf/poppy centring)
  WAVETAB               wavelength_um, psf_sum (in-field energy fraction),
                        fwhm_arcsec (measured) per slice
```

Slices are stored **as computed** (stpsf `normalize='first'`: unit flux enters
the telescope, so a slice sum is the fraction landing inside the field --
QA-relevant, and the wing loss an extended-source aperture correction cares
about). `PsfCubeProduct.at()/.convolve()` renormalize each slice to unit sum
by default, so convolution conserves surface brightness; pass
`normalized=False` to keep the absolute normalization. `convolve()` shifts
the kernel centre to the origin analytically (Fourier shift theorem), so no
half-pixel displacement ever enters -- pinned by a test. `resampled()`
conserves each slice's total flux (values scaled by the pixel-area ratio;
cubic-spline undershoot clipped to zero).

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

## The library grid

The sampling is the cube's own spaxel scale divided by `oversample`
(default 4): 26 mas for NIRSpec, 33/43/61/87 mas for MRS ch1-ch4 -- read
from the s3d WCS, so per-band scales come out right automatically. Is /4
fine enough? The *optical* PSF is band-limited at lambda/2D (15 mas at
NIRSpec's shortest wavelengths), but the stored slices are the *broadened*
cube-frame PSF: NIRSpec's 50 mas-sigma Gaussian leaves ~1e-8 of the power at
the 26 mas Nyquist frequency, and every MRS scale sits below even the
optical lambda/2D of its band. Interpolating this library onto a much finer
model grid (12 mas at 190 pc for the 256 px / +-300 AU convention) is
therefore lossless in practice, while the library stays small and cheap.

The angular field is `fov_arcsec` (default 6", generous for the PSF wings
that matter in convolution); the pixel count follows, rounded up to odd so
the kernel centre is an integer pixel. An explicit `pixelscale_arcsec`
overrides the oversample rule.

## What the slices contain, and orientation

stpsf's IFU mode is used with its **as-built broadening** left on: for MIRI
MRS the empirical model tuned to commissioning cubes (Argyriou et al. 2023;
Law et al. 2023: FWHM = 0.033 lambda + 0.106") -- a Gaussian along the
along-slice axis plus a **slice-width boxcar across slices**, so the kernel
is anisotropic -- and for NIRSpec a 0.05"-sigma Gaussian. The slices
therefore approximate the PSF **as realised in reconstructed s3d cubes**;
the bare optical PSF is available with `broadening: none`.

stpsf computes (and its docs recommend comparing) in the instrument-aligned
frame, and the default `frame: ideal` keeps exactly that: **array +y along
the aperture ideal +y axis** for both instruments (for NIRSpec, stpsf's
extra 90-degree IFU-align display rotation is disabled so the two
instruments share one convention), and *nothing is ever interpolated at
generation*. The sky position angle of that axis -- `PA_APER`, else
`ROLL_REF + V3I_YANG`, overridable via `position_angle_deg` -- is recorded
in `JWFPSFPA`, and `resampled(to_sky=True)` folds the rotation to the
north-up/east-left frame of `skyalign` cubes into the same interpolation as
the scale change. `frame: sky` instead rotates the slices once at
generation, for direct overlay on skyalign cubes (PSF photometry, quick
looks).

Fidelity notes: the MRS aperture ideal frame is itself an approximation
(stpsf averages the skewed slice geometry), parity of the faint speckle
pattern is not guaranteed, and a NIRSpec `ifualign` *cube* may be rotated by
a further +-90 degrees relative to the library frame (use
`resampled(rotation_deg=...)` if you compare in that frame). Treat
sub-degree PA effects and speckle-level structure as beyond this product's
fidelity. If your model is rendered in a frame other than sky (e.g. disk
major axis along x), compose that angle into `rotation_deg` yourself.

## Engines, methods, OPD

| parameter | choices | notes |
|---|---|---|
| `engine` | `stpsf` (default) / `gaussian` | `gaussian` is an offline analytic stand-in at the empirical FWHM -- no spikes/rings; dry-runs, tests |
| `method` | `exact` (default) / `fast` | `exact` = one full `calc_psf` per wavelength; `fast` = stpsf's `calc_datacube_fast` (one pupil propagation for all wavelengths, ~100x faster) + stpsf's broadening applied per slice; assumes a wavelength-independent exit-pupil wavefront |
| `opd` | `default` / `by_date` / file | `by_date` fetches the measured in-flight wavefront nearest DATE-OBS from MAST (network!); `default` is offline |
| `broadening` | `default` / `gaussian` / `none` | `default` = empirical (MRS) / Gaussian (NIRSpec) |

Generation cost with `exact` is minutes per band product (and band products
run in parallel like any other per-file stage); results are checkpointed
like every jwstflow task, so reruns are free -- and because the library is
distance-independent, model iteration never triggers regeneration.

## Testing

`tests/test_psf.py` runs entirely offline: the Gaussian engine exercises the
full product path (generation -> write -> read -> resample -> interpolate ->
convolve -> QA), the geometry/rotation/centring conventions are pinned
numerically (rotation sign, the (n-1)/2 kernel centre, flux conservation,
zero displacement, resampling that preserves angular FWHM and flux), and a
stub of the verified stpsf 2.x IFU API pins how `StpsfEngine` must drive the
real package -- in particular that the pixel scale is applied *after* band
selection (stpsf resets it on any IFU aperture change) and that the broadened
`DET_DIST` plane is the one stored.
