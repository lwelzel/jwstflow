# Changelog

All notable changes to `jwstflow` will be documented in this file.

The format follows the principles of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project intends to use [Semantic Versioning](https://semver.org/) once public releases begin.

## [Unreleased]

### Added

- QA figure standard v2 as the `jwstflow.qa` package (style guide in `docs/qa_figures.md`): one set of font sizes, line widths and colors by role; mathtext axis labels from shared constants; NaN pixels in grey, `asinh`/`sqrt`/`log`/`linear` stretches and fixed-width colorbars with pointed ends; step lines and error bands that break at gaps; legends with a notes column that fit the figure; plain-number log axes. The package is split by what a figure is made of (`style`, `units`, `lines`, `frames`, `annotation` with `labels` / `placement` / `layout`, `io`), with the stock QA steps in `jwstflow.qa.steps`.
- Two figure classes to build QA steps on: `qa.Spectrum` (a flux-like quantity over wavelength on logarithmic axes, 1-sigma bands, spectral features annotated, optional panels below, wavelength zoom) and `qa.Frame` (an image with its colorbar, mask outlines and markers). Both stand alone or draw onto an axes of a larger figure.
- Frames carry two sets of axes with one common zero: offsets in pixels (bottom/left) and offsets on the sky in arcsec (top/right) from the same reference position (`SkyFrame`; the target, else the image centre); the legend states the array pixel of the zero. Panels of a grid label only their outer sides (`sides=`).
- Feature annotation that keeps the spectrum the main element (`qa.annotate_features`): every line in range is marked whether the spectrum shows it or not, each gas label a constant factor above its own line with a stem, colored by species; a line that towers over its neighbours keeps its label on top and the neighbours dodge left or right; crowded stretches use two tiers; converging members of a series share one label with a stem per member; broad gas emission is outlined above the plot; stellar photospheric lines are dashed and labelled below, side-by-side lines of one species merged into one named multiplet; ices, silicates and PAH are rectangles in rows at the foot of the plot inside its frame, all bands of a species in one row, labelled with typeset formulae. Connectors are smooth curves that cannot cross; a figure with more labels gets wider; strong peaks no catalogue line explains are marked with a grey `?`. No legend entries. `series="combs"` draws the series as named rows of ticks instead (`plot_stitch` writes that as `*_series.png`).
- A figure of the lines a spectrum shows: `plot_spectrum` and `plot_stitch` write, next to the figure that labels every catalogue line, `*_detected.png` (with `features`; `detected: false` leaves it out) -- gas lines labelled where an emission peak stands within one sample or 50 km/s of them (`qa.IDENTIFY_VELOCITY`), photospheric lines where a dip does, broad gas bands drawn where the spectrum shows their forest (`qa.FOREST_MIN_PEAKS` unidentified peaks within `qa.FOREST_WINDOW` samples), the notes saying so (`qa.DETECTED_NOTE`). In both figures a broad gas band explains the peaks inside it (no `?`) only where its forest is seen, and inside a seen forest a catalogue line counts as found only where its peak stands above the forest around it. `qa.Spectrum.decorate` / `finish` take `detected_only`.
- Axis labels name quantities by their symbols (`F_nu [mJy]`, `lambda [um]`, `I_nu`, `tau`).
- Line tables generated from primary lists by `scripts/build_line_tables.py` (H I: Rydberg formula; H2: HITRAN; He I and forbidden lines: NIST ASD Ritz wavelengths; all vacuum). `gas_lines` keeps its set of lines -- it is what data steps mask -- with wavelengths corrected: H I Pa-alpha (was the air wavelength), H2 0-0 S(12) and S(13). The further lines (H I series up to n = 25, all branches of the H2 0-0, 1-0 and 2-1 bands, He I, rarer forbidden lines) are the new opt-in `gas_lines_extended` table, annotated in QA figures and never part of a bare `features: all`.
- `photospheric_lines` feature dataset (stellar photospheric absorption lines and bandheads, 0.85-4.1 um; atomic lines from NIST ASD, CO 2-0 and 3-0, 4-1 heads from HITRAN, other bandheads as quoted), opt-in: annotated in QA figures, never part of a bare `features: all`.
- New QA steps: `qa_cube_summary` (median image, coverage and signal-to-noise maps, flux and filled footprint per plane), `qa_cube_slices` (channel maps), `plot_snr` (signal-to-noise per sample), `plot_spectrum_stages` (a spectrum across 1-D processing stages and their ratios).
- `flag_gap_outliers` (`jwstflow.cubes.gaps`): the detector pixels next to the NRS1/NRS2 gap of NIRSpec IFU exposures -- the ones that build the thinly sampled voxels at the gap edges of a cube -- are tested against the other exposures at their sky position and wavelength, and only those that disagree are flagged; nothing is masked for coverage. One `*_gapcov.fits` product per group of exposures maps the coverage on the sky (`gap_limits_at`), drawn by `qa_gap_coverage`.
- Imaging reductions end to end. `download.instrument` takes `NIRCAM` and `NIRISS`; `PUPIL` and `MODULE` are cached from the headers (cached records carry the keyword set they were read with, so a new keyword re-reads them). `{OPTELEM}` names an imager's filter and pupil as DMS does (`clear-f200w`, `f405n-f444w`, `f770w`) and the new `{SUBARRAY_SUFFIX}` adds DMS's subarray part (`-brightsky`), so level-3 imaging products carry MAST's names. Preset `nircam_imaging` (detector1 with snowball flagging -> image2 -> image3, one mosaic per filter, pupil and subarray); `miri_imaging` groups by subarray and names with `{OPTELEM}{SUBARRAY_SUFFIX}`. Every shipped preset is validated by a test.
- `jwstflow.images`, the toolkit of imaging products: `read_image` (SCI / ERR / WHT / CON / DQ, celestial WCS, pixel solid angle, band; `n_exposures` from CON), `Image.cutout` (a box around a sky position with its WCS and header shifted, its origin recorded as `JWFCUTX0/Y0`), `band_of` (the filter from either wheel; the narrower wins), `diffraction_fwhm_arcsec`, `throughput` and `band_average` (the photon-weighted mean F_nu of a spectrum through a band: synthetic photometry). `refdata/filters/`: 54 NIRCam, NIRISS and MIRI filters (pivot, equivalent width, half-maximum edges) and their throughput curves, from stpsf by `scripts/build_filter_table.py`. `testing.synthetic_i2d`.
- `qa_image_summary`: a resampled image, the exposures behind every pixel and its S/N above the sky, with the sigma-clipped sky level and scatter. `mast_compare` compares mosaics (`_i2d`) with MAST's: around the target, MAST's interpolated onto jwstflow's grid, the difference, the flux ratio above each sky, the sky difference and the shift between them (`*_i2ddiff.fits` and a figure).
- `write_mask_product(..., headers=)`: a mask on another grid than its source file (the cutout of an image) carries that grid's WCS.
- `image_astrometry` (`jwstflow.images.astrometry`): how well the images of a run are registered, to each other and to Gaia DR3. From the source catalogues `image3` writes next to its mosaics, the point sources of every image (`is_extended` false, S/N, isolation, a complete core; cuts in units of the band's FWHM) are matched in two passes -- a bulk shift from all pairs within `search`, then mutual nearest neighbours within `match_radius` FWHM -- to a reference image (`reference`; default the one with the most point sources) and to Gaia DR3, queried once for the run's footprint (ESA archive via `astroquery`, else the STScI service `tweakreg` reads), cached as `<base>_gaiadr3.ecsv` and propagated with the proper motions to each image's epoch. Per comparison the clipped median offset, the robust scatter and the error of the median [mas], and every matched source, in one `<base>_astrom.fits`; a failed Gaia query leaves the relative comparison. `qa_image_astrometry` draws the offsets per image and their medians against wavelength.
- Joint cubes (`jwstflow.cubes.stitching`), **in development: the joint cubes are not science-ready products and no reduction recipe builds them yet**: `stitch_cubes` joins the band cubes of one instrument -- NIRSpec gratings, the twelve MRS bands -- along wavelength into one `*_s3dcomb.fits`, an `IFUCubeModel` with a tabular wavelength axis followed by the records of the stitch. One spatial grid (the finest spaxel scale over the union of the fields, on the pixel lattice of the finest band); no spectral resampling (every plane is one plane of one band, bands switch at crossovers); per neighbouring pair the flux ratio around the crossover, measured like `stitch_segments` on the excess of both bands over the level of the field where the source stands out (so a level that differs between bands is not taken for a factor; whole sources only -- a bright object the edge of the field cuts is left out), always recorded and optionally applied as one factor per band with its uncertainty in ERR; the level of the field in both bands and the shift between them (the two registered against each other over one window around the core they share) recorded per crossover. `scales_from` adopts the scales of a stitched spectrum of the same data instead (for a faint or extended source, whose ratio depends on where it is measured). Cubes on the joint pixel lattice are copied; any other cube is resampled once by exact pixel overlap (flux-conserving, nothing interpolated), which the product and a warning say.
- `cube_build_matched`: the official `cube_build` with the spatial grid prescribed -- the joint grid of the band cubes in `grid_from` -- so every band is drizzled once, straight onto the grid of the joint cube, and `stitch_cubes` resamples nothing. On the lattice of the reference band the cube is the pipeline's own, value for value.
- `qa_cube_stitch`: the figures of a joint cube from its records -- the flux over the common field (the cube over its segments), every overlap with its ratio window and factor, the bright core of the source in both segments of every pair with the region the ratio was measured over, and the shifts between neighbouring segments against wavelength. `qa_cube_summary` and `qa_cube_slices` read joint cubes too.
- `jwstflow.spectra.overlaps`: the measurement behind every stitch -- `measure_overlap_ratio`, `chain_scales`, `overlap_midpoints`, `mask_windows`, `OverlapRatio` -- as toolkit functions that the stitch of spectra and the stitch of cubes share. `jwstflow.cubes`: `celestial_wcs_from_header` and `hdu_wavelengths` (a cube's celestial WCS and plane wavelengths from its FITS extensions, whatever its spectral axis).
- `jwstflow.spectra.lines`: the search for lines in a spectrum and the decision which catalogue lines it shows, moved from the QA figures (`qa/labels.py`, `FeatureLayout.detect`) into the spectra toolkit so that data steps decide as the `*_detected.png` figures do. `detect_lines` and its constants (`DETECT_SIGMA`, `UNIDENTIFIED_SIGMA`, `UNIDENTIFIED_IN_BAND`, `IDENTIFY_VELOCITY`, `GAP_STEP`, `EDGE_SAMPLES`, `FOREST_MIN_PEAKS`, `FOREST_WINDOW`, `FEATURE_NARROW_BAND`) live there, with `merge_peaks`, `identify_lines` (the catalogue lines found, the gas bands whose forest is seen, the unidentified peaks: `LineIdentification`), `line_wavelength` (the narrow bands of the gas catalogue are lines: one rule for the annotation and the search) and, on `Feature` objects, `search_features` (`LineSearch`). `jwstflow.qa` still exports every moved name. A pure refactor: the QA figures are pixel-identical, and no module that defines a step changed, so no step reruns.

### Changed

- `subtract_psf_neighbour` v2: `line_planes: core` (opt-in; default `field` reproduces v1 to the bit) -- in the planes of gas lines the neighbour's model is the neighbouring continuum planes' model scaled by its brightness on its core, line plane over continuum, so extended line emission of the field is no longer fitted as the neighbour's light and subtracted from the target (HH 48 NE: NE's line fluxes lost 2-15 % to it). Continuum planes are unchanged; `*_nbrfit.fits` records `LINE`, `LINE_SCALE`, `LINE_SCALE_ERR`, `FLUX_FIELD`; which spaxels of the neighbour's core are blanked is decided on the fit over the field in either mode.
- Spectral-feature annotation: a strong peak no catalogue line explains is labelled with its wavelength, `? [4.954]` (grey), in every annotated figure. A peak identifies a catalogue line within one sample or 50 km/s (was two samples or 0.05 % of the wavelength -- 150 km/s, where 10-25 % of the matches in NIRSpec spectra of edge-on disks were chance coincidences with other peaks; next to a gap in the spectrum the old tolerance spanned the gap); the three samples at either end of the spectrum and of every stretch between gaps are no longer searched (a detector edge rising into a gap is no line). No connector crosses a label or another connector any more: a label pinned over a towering line moves just aside where another line's stem would rise inside its column, a moved label stays between the pinned labels on either side of its line, and a series shares one label only between members with no other labelled line between them (`H₂ 2-1 Q(1, 3–5)` names exactly the members with stems). Crowded figures carry more labels and are wider. Versions bumped: `plot_spectrum` 8, `plot_stitch` 8, `plot_snr` 3, `plot_spectrum_stages` 3, `plot_stitch_background` 4, `qa_cube_stitch` 2.
- All stock QA steps redrawn to the standard and rebuilt on `qa.Spectrum` / `qa.Frame` (versions bumped, figures are re-rendered on the next run): `plot_spectrum`, `plot_stitch`, `plot_stitch_background` and `plot_snr` are spectrum-like figures with logarithmic axes (`plot_snr` and `plot_stitch_background` annotate the features by default); `quicklook_image` defaults to the `asinh` stretch; `qa_cube_summary` puts its spectra over the full width under the three maps; `plot_stitch_overlaps` notes pair and factor inside each panel; the background estimate of `plot_stitch_background` is drawn in its role color from the standard colormap.
- The stock QA steps moved from `jwstflow.contrib.qa` to `jwstflow.qa.steps` (`spectra`, `stitch`, `images`, `cubes`, `summary`); entry-point names are unchanged. Reinstall the package (`pip install -e .`) so that the entry points name the new modules.
- Package layout (`docs/guide.md`, "How the package is laid out"; the dependency rules are checked by `tests/test_architecture.py`). No behaviour changed, and a workflow that names its steps by their registered names needs no editing, but the package directory now holds `cli.py` and subpackages only, so most module paths moved:

  | was | is now |
  |---|---|
  | `jwstflow.naming` | `jwstflow.steps.naming` |
  | `jwstflow.steps.base` | `jwstflow.steps` (split into `interface`, `adapters`, `registry`, `plugins`, `descriptions`) |
  | `jwstflow.spectra` (module) | `jwstflow.spectra` (package; same names from `jwstflow.spectra`, code in `spectra/x1d.py`) |
  | `jwstflow.spectra.cube_wavelengths` | `jwstflow.cubes.cube_wavelengths` |
  | `jwstflow.spectra.mrs_psf_fwhm_arcsec` | `jwstflow.cubes.psf.mrs_psf_fwhm_arcsec` |
  | `jwstflow.features` | `jwstflow.spectra.features` |
  | `jwstflow.stitching` | `jwstflow.spectra.stitching` |
  | `jwstflow.masks` | `jwstflow.cubes` (all names), split into `cubes/wcs.py`, `cubes/morphology.py`, `cubes/instrument.py` and `cubes/masks.py` (the product contract) |
  | `jwstflow.apcorr` | `jwstflow.cubes.apcorr` (names also from `jwstflow.cubes`) |
  | `jwstflow.psf` | `jwstflow.cubes.psf` (a package: `grid`, `instrument`, `engines`, `product`, `steps`) |
  | `jwstflow.clusters` | `jwstflow.cubes.clusters` (a package: `regions`, `compare`, `product`, `detector`, `steps`) |
  | `jwstflow.targets` | `jwstflow.data.targets` |
  | `jwstflow.associations`, `jwstflow.crds`, `jwstflow.dagviz` | `jwstflow.engine.associations`, `.crds`, `.dagviz` |
  | `jwstflow.project` | `jwstflow.config.project` |
  | `jwstflow.scaffold` (module) | `jwstflow.scaffold` (package: `package`, and `step` with `scaffold_step`, formerly `jwstflow.testing.scaffold`) |
  | `jwstflow.testing.mock` (module) | `jwstflow.testing.mock` (package, same names) |

  A data step is defined with its feature and every QA step in `jwstflow.qa.steps`: `stitch_segments` in `jwstflow.spectra.stitching`, `psf_cube` in `jwstflow.cubes.psf.steps`, the cluster steps in `jwstflow.cubes.clusters.steps`, `fix_msa_metafile` in `jwstflow.data.msa`; `qa_psf_cube`, `qa_spaxel_clusters` and `mast_compare` in `jwstflow.qa.steps`. No feature imports another: `detector_sky_coordinates`, `slice_wcs`, `dq_value` (`jwstflow.cubes.detector`) and `sky_offsets` (`jwstflow.cubes.wcs`) are toolkit functions, importable from `jwstflow.cubes`.
- jwstflow's own steps are registered in one place, the `jwstflow.steps` entry points in `pyproject.toml`; the second, in-package table of the same names (`CONTRIB_ALIASES`) is gone. A source tree therefore has to be installed (`uv sync`, `pip install -e .`) for its step names to resolve -- as was always the case for step packages.
- Checkpoints: the task ids contain a hash of the file that defines a step, and the file of every step defined in jwstflow changed (its imports at least). In an existing reduction all custom stages therefore run again on the next `jwstflow run`, and so does every stage downstream of a product they rewrite -- in practice everything after the first custom stage of a workflow; the official pipeline stages before it (addressed by their built-in aliases) keep their checkpoints. Reinstall editable installs (`uv sync`): the entry points name new modules.
- `chain_scales` (and so `stitch_segments` with `min_overlap_snr: 0`) takes a ratio that is not positive for no measurement: a flux scale cannot be negative.
- QA: a (nearly) constant curve on a logarithmic flux axis no longer asks matplotlib for billions of minor ticks. The tick step of a narrow log axis was fixed from the view before the limits were set; it now follows the major ticks whatever the limits become, and the limits are set first.
- Mock workflows (`jwstflow.testing.mock`): the spec3 stubs leave `_crf` files next to their cubes like the real stage (a step run on its own returns them too), and `StubCubeBuildMatched` / `StubPixelReplace` stand in for the two official steps of the joint-cube chain.
- `stitch_segments` measures and chains through `jwstflow.spectra.overlaps` (same numbers; `OverlapRatio` is still importable from `jwstflow.spectra.stitching`, the private `StitchSegments._scales` is gone). The file of the step changed, so its tasks rerun.
- Frames of a cube whose wavelength axis is a table (`CTYPE3 = 'WAVE-TAB'`: joint cubes, the official multi-band cubes) get their sky axes: `SkyFrame.from_header` reads the two celestial axes only.
- `plot_stitch_overlaps` draws its panels with the module functions `overlap_panel` / `overlap_legend` (`jwstflow.qa.steps.stitch`), which `qa_cube_stitch` uses as well; the figure is unchanged.
- A step that reads MAST's own products declares it (`needs_reference_products = True`); the run waits for their background download before such a stage instead of recognising `mast_compare` by its stage name.
- `jwstflow check-step <name>` (and `jwstflow.testing.check_step`) reports when a name or dotted path leads to a module that only re-exports the step: the defining file is what is hashed into the task ids.
- `jwstflow.__version__` is read from the installed package metadata (it had stayed at 0.1.0).
- `qa.colorbar` takes `width`/`pad` in inches (was: fractions of the axes); `qa.imshow(axes_labels=False)` became `sides=None`.

### Removed

- The `jwstflow.contrib` package. Nothing distinguished its contents from the rest of jwstflow, and its name collided with "contributed packages" -- the separate distributions that plug into jwstflow. Its steps are filed with their features (see Changed).
- The old module paths in the table above and the `jwstflow.qafig` alias of `jwstflow.qa`: importing them fails; step packages import the new paths (jwstflow-midas and jwstflow-joys were updated in step). The same holds for dotted *step addresses* in workflows (`jwstflow.contrib.qa:PlotSpectrum`, `jwstflow.stitching:StitchSegments`, ...): name steps by their registered names.
- `jwstflow.testing.scaffold` (it is `jwstflow.scaffold.scaffold_step`), and code nothing used: `copy_inputs` (a function-step template among the QA steps), `features.dataset_metadata`, `associations.write_all`.

## [0.2.0] - 2026-09-29

The repository currently identifies the development version as `0.1.0`. Until the first public release is tagged, release-preparation changes should remain in this section. At release time, move the relevant entries into a versioned section and add the release date.

### Added

- YAML-driven orchestration of the official JWST calibration pipeline.
- Workflow planning, validation, execution, checkpointing, resume, and status reporting.
- MAST data acquisition and CRDS reference-data setup/prefetch support.
- Association construction for pipeline stages that require grouped inputs.
- Serial, process-based, and Dask-oriented execution infrastructure.
- Deterministic task state and provenance information for reproducible reductions.
- DMS-style product naming and structured per-target/per-run output trees.
- Workflow graph generation and QA-oriented contributed steps.
- A validated custom-step API and entry-point mechanism for external contributed-step packages.
- Scaffolding commands for custom steps and contributed packages.
- Presets and workflow support for NIRSpec and MIRI use cases.
- Offline unit, engine, configuration, and mock-observation integration testing infrastructure.
- BSD 3-Clause licensing.
- Contributor guidelines, citation metadata, changelog, and project code of conduct in preparation for public development.

### Changed

- Nothing yet.

### Fixed

- Nothing yet.

### Deprecated

- Nothing yet.

### Removed

- Nothing yet.

### Security

- Nothing yet.

<!--
Release template:

## [X.Y.Z] - YYYY-MM-DD

### Added
### Changed
### Fixed
### Deprecated
### Removed
### Security

When a public repository URL is available for release comparisons, add links such as:
[Unreleased]: https://github.com/lwelzel/jwstflow/compare/vX.Y.Z...HEAD
[X.Y.Z]: https://github.com/lwelzel/jwstflow/releases/tag/vX.Y.Z
-->
