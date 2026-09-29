# Changelog

All notable changes to `jwstflow` will be documented in this file.

The format follows the principles of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project intends to use [Semantic Versioning](https://semver.org/) once public releases begin.

## [Unreleased]

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
