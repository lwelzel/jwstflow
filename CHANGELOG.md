# Changelog

All notable changes to `jwstflow` will be documented in this file.

The format follows the principles of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project intends to use [Semantic Versioning](https://semver.org/) once public releases begin.

## [Unreleased]

### Added

- QA figure standard v2 as the `jwstflow.qa` package (style guide in `docs/qa_figures.md`): one set of font sizes, line widths and colors by role; mathtext axis labels from shared constants; NaN pixels in grey, `asinh`/`sqrt`/`log`/`linear` stretches and fixed-width colorbars with pointed ends; step lines and error bands that break at gaps; legends with a notes column that fit the figure; plain-number log axes. The package is split by what a figure is made of (`style`, `units`, `lines`, `frames`, `annotation` with `labels` / `placement` / `layout`, `io`), with the stock QA steps in `jwstflow.qa.steps`.
- Two figure classes to build QA steps on: `qa.Spectrum` (a flux-like quantity over wavelength on logarithmic axes, 1-sigma bands, spectral features annotated, optional panels below, wavelength zoom) and `qa.Frame` (an image with its colorbar, mask outlines and markers). Both stand alone or draw onto an axes of a larger figure.
- Frames carry two sets of axes with one common zero: offsets in pixels (bottom/left) and offsets on the sky in arcsec (top/right) from the same reference position (`SkyFrame`; the target, else the image centre); the legend states the array pixel of the zero. Panels of a grid label only their outer sides (`sides=`).
- Feature annotation that keeps the spectrum the main element (`qa.annotate_features`): every line in range is marked whether the spectrum shows it or not, each gas label a constant factor above its own line with a stem, colored by species; a line that towers over its neighbours keeps its label on top and the neighbours dodge left or right; crowded stretches use two tiers; converging members of a series share one label with a stem per member; broad gas emission is outlined above the plot; stellar photospheric lines are dashed and labelled below, side-by-side lines of one species merged into one named multiplet; ices, silicates and PAH are rectangles in rows at the foot of the plot inside its frame, all bands of a species in one row, labelled with typeset formulae. Connectors are smooth curves that cannot cross; a figure with more labels gets wider; strong peaks no catalogue line explains are marked with a grey `?`. No legend entries. `series="combs"` draws the series as named rows of ticks instead (`plot_stitch` writes that as `*_series.png`).
- Axis labels name quantities by their symbols (`F_nu [mJy]`, `lambda [um]`, `I_nu`, `tau`).
- Line tables generated from primary lists by `scripts/build_line_tables.py` (H I: Rydberg formula; H2: HITRAN; He I and forbidden lines: NIST ASD Ritz wavelengths; all vacuum). `gas_lines` keeps its set of lines -- it is what data steps mask -- with wavelengths corrected: H I Pa-alpha (was the air wavelength), H2 0-0 S(12) and S(13). The further lines (H I series up to n = 25, all branches of the H2 0-0, 1-0 and 2-1 bands, He I, rarer forbidden lines) are the new opt-in `gas_lines_extended` table, annotated in QA figures and never part of a bare `features: all`.
- `photospheric_lines` feature dataset (stellar photospheric absorption lines and bandheads, 0.85-4.1 um; atomic lines from NIST ASD, CO 2-0 and 3-0, 4-1 heads from HITRAN, other bandheads as quoted), opt-in: annotated in QA figures, never part of a bare `features: all`.
- New QA steps: `qa_cube_summary` (median image, coverage and signal-to-noise maps, flux and filled footprint per plane), `qa_cube_slices` (channel maps), `plot_snr` (signal-to-noise per sample), `plot_spectrum_stages` (a spectrum across 1-D processing stages and their ratios).

### Changed

- All stock QA steps redrawn to the standard and rebuilt on `qa.Spectrum` / `qa.Frame` (versions bumped, figures are re-rendered on the next run): `plot_spectrum`, `plot_stitch`, `plot_stitch_background` and `plot_snr` are spectrum-like figures with logarithmic axes (`plot_snr` and `plot_stitch_background` annotate the features by default); `quicklook_image` defaults to the `asinh` stretch; `qa_cube_summary` puts its spectra over the full width under the three maps; `plot_stitch_overlaps` notes pair and factor inside each panel; the background estimate of `plot_stitch_background` is drawn in its role color from the standard colormap.
- The stock QA steps moved from `jwstflow.contrib.qa` to `jwstflow.qa.steps` (`spectra`, `stitch`, `images`, `cubes`, `summary`); entry-point names are unchanged, `jwstflow.contrib.qa` re-exports the classes, and `jwstflow.qafig` remains as an alias of `jwstflow.qa`. Reinstall the package (`pip install -e .`) so that the entry points name the new modules.
- `qa.colorbar` takes `width`/`pad` in inches (was: fractions of the axes); `qa.imshow(axes_labels=False)` became `sides=None`.

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
