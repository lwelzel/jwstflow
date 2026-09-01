# Testing strategy

One strategy serves all four repositories — `jwstflow` (the engine),
`jwstflow-midas` and `jwstflow-joys` (contributed steps), and
`jwstflow-reducer` (the reduction project). Everything runs offline, in
seconds, from the flat side-by-side checkout the `tool.uv.sources` entries
already assume:

```
work/
  jwstflow/           the engine + the shared testing toolkit
  jwstflow-midas/     NIRSpec disk steps
  jwstflow-joys/      MIRI MRS steps
  jwstflow-reducer/   the ESO-Ha 569 reduction record
```

## The three levels

**1. Step unit tests** (existing pattern, per repo). One step at a time on
synthetic products, through `jwstflow.testing.run_step` — exactly the code
path a worker uses (parameter validation, output checks). Milliseconds each.
See any `tests/test_*.py` in the step packages; `jwstflow new-step`
scaffolds this kind of test.

**2. Engine and config tests** (`jwstflow/tests/`). The orchestration
semantics on toy steps and text files: planning, deterministic task ids,
checkpoint/resume and invalidation, failure policy, orphan detection, stage
selection, association building, discovery filters, naming grammar, config
composition (`extends`, interpolation, overrides). No FITS, no astropy —
sub-second.

**3. Mock-observation workflow tests** (all repos, `-m integration`).
Complete workflow YAMLs through the real engine on **mock ESO-Ha 569
observations** — the heart of the strategy, built on
`jwstflow.testing.mock`:

* `nirspec_ifu_observation()` / `miri_mrs_observation()` describe
  observations with the *structure* of program 1751 (DMS file names, header
  keywords, gratings/bands x dithers x detectors, a dedicated sky
  observation, the MAST download log that carries the DMS target id) but
  **strongly reduced sizes**: 2 dithers instead of 4, ~100–150 spectral
  planes instead of thousands, ~31 px fields. `write_observation()`
  materialises the `_uncal` files.
* A deterministic **scene** (`MockScene`: edge-on disk and/or point source,
  flat sky, emission-line halos at real feature wavelengths, a hot pixel)
  travels in `JWFMK*` header keywords from the uncal files into every
  product, so tests assert *science* against known truth: the disk mask
  agrees across gratings and excludes the hot pixel, the in-field
  background recovers the injected sky, extraction recovers the injected
  flux to a percent, the bands stitch at unit ratio.
* The MIRI mock optionally carries the simultaneous imager frames
  (`imager=True`): a deterministic star field displaced by an injected
  pointing error, with `write_gaia_catalog()` providing the truth table so
  the `gaia_offset`/`wcs_offset` astrometry loop closes offline. MRS band
  cubes use the real per-channel spaxel scales (0.13–0.35 arcsec), so
  PSF-scaled apertures and sky annuli behave as on real cubes.
* The official pipelines are replaced by **scene-faithful stubs**
  (`StubDetector1/StubSpec2/StubSpec3/StubImage2/StubImage3`) that keep the
  interfaces exact:
  same stage names and directories, same DMS product names (`cube_build`'s
  band suffix included), consuming the same association files, honouring
  the parameters that shape the data flow (`steps.extract_1d.skip`,
  background members → sky subtraction, `background_dir` →
  `*_bkgspec.fits` lookup). Everything that is *ours* — discovery,
  associations, checkpointing, naming, and every jwstflow / contributed
  step — runs for real.
* `stub_pipelines()` shadows the step aliases in-process (serial backend /
  `workers=1`); by dotted path (`jwstflow.testing.mock:StubSpec3`) the
  stubs also run in spawned workers, which is how the core repo tests the
  process-pool executor. `offline_overrides()` pins CRDS and disables the
  network; `run_mock_workflow()` bundles all of it for one-line tests.

Who tests what at level 3:

| repo | workflow under test | focus |
|---|---|---|
| jwstflow | `tests/data/mini_nirspec.yaml` | engine end to end: process pool, associations, DMS naming, resume, `--only`, graph rendering |
| jwstflow-midas | `tests/data/nirspec_ifu_mock.yaml` (mirrors the production NIRSpec YAML) | the disk chain: flag_frames → pass-1 cubes → disk_mask → in_field_background → background-aware spec3 → extract_extended → stitch + QA |
| jwstflow-joys | `tests/data/miri_mrs_mock.yaml` (mirrors the production MIRI YAML) | the MRS chain: the simultaneous-imager astrometry loop (image2/image3 stubs → gaia_offset on a local truth catalogue → wcs_offset on the rates, closing on an injected pointing error) → band cubes → on/off annulus extraction → defringe → clean → per-aperture stitch_bands → LSRK + QA |
| jwstflow-reducer | **the production YAMLs themselves** | the dress rehearsal: `nirspec_ifu.yaml`, `miri_mrs.yaml` and `combine_nirspec_miri.yaml` unmodified (two mock-specific `--set`-style overrides), full product tree, flux recovery, cross-instrument 1.7–28 µm combine, cached reruns |

The reducer additionally has `tests/test_workflows_validate.py` — a cheap
standing guard that every workflow YAML still loads, resolves against the
*installed* step packages, orders correctly, and finds its plugins. Run it
after any dependency bump.

## Running

Each repo's suite is self-contained: `python -m pytest` in the repo (using
the shared venv, e.g. `jwstflow-reducer/.venv`). Deselect the workflow
tests with `-m "not integration"` when iterating on a unit.

From the flat checkout, `jwstflow/scripts/test-all.sh` syncs one
environment in `jwstflow-reducer` (whose `tool.uv.sources` point at the
sibling checkouts, editable) and runs every repo's suite with it:

```bash
./jwstflow/scripts/test-all.sh                 # everything
./jwstflow/scripts/test-all.sh -m "not integration"   # fast lane, args go to pytest
```

CI is provided as a `ci/test.yml` template in each repo: it reproduces the
flat layout by checking out the sibling repositories next to the repo under
test, then runs the same suite. Copy it to `.github/workflows/test.yml` from
a clone whose credential has the `workflow` scope (automation tokens often
lack it, and GitHub rejects pushes of workflow files without it). For the
private packages the checkout needs a token that can read the siblings
(`SIBLING_REPOS_TOKEN` secret, falling back to the workflow token).

## Adding a test

* A new step → scaffold the unit test (`jwstflow new-step`), assert numbers
  against a constructed truth, not against the step's own output.
* New behaviour of a chain (a header a later step needs, grouping keys, a
  stage contract) → extend the repo's workflow test; the mock observation
  is deterministic, so exact assertions are safe.
* A new observing mode → add a builder next to `nirspec_ifu_observation()`
  and give the stubs the band table; keep real wavelength ranges and DMS
  naming so contributed steps meet realistic metadata.
* Reduce sizes, never physics: fewer planes/dithers/pixels is fine, but
  keep the scene features that the step under test keys on (contrast
  ratios, line wavelengths, overlaps between bands).

## Known gaps (deliberate, for now)

* The Typer CLI is exercised only through its underlying functions
  (`load_config`, `Runner`, `scaffold`), not via `CliRunner`.
* MAST download and CRDS prefetch are not integration-tested (network);
  `build_query` and `target_ids_from_log` have unit coverage.
* The dask executor is not exercised (serial and process are).
* `mast_compare` runs in the workflow tests but only its no-reference path;
  its diff math has no unit test yet.
