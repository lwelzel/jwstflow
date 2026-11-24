# jwstflow guide

Everything beyond the [README](../README.md): the run model, naming, data
products, scheduling and the full YAML reference.

## Why this design

The requirements were: light-weight, maintainable, compartmentalised, a
clean YAML interface, and built on well-maintained libraries. The choices,
and the alternatives that were rejected:

| concern | choice | why |
|---|---|---|
| configuration | **pydantic v2** models are the single source of truth; YAML via PyYAML | validation with readable errors, defaults, docstrings, and a JSON Schema for editor autocompletion all come from one class hierarchy |
| composition | ~150 lines of own code: `extends:` (files or `preset:<name>`), deep merge with stages merged **by name**, `${a.b}` / `${env:VAR,default}` interpolation, CLI dot-list overrides | these are the four Hydra features people actually use; no framework needed |
| unit of work | `stpipe.Step.call()` | pulls CRDS parameter-reference files (`pars-*`), so results match STScI's defaults; any `Step` subclass works, including subclasses of the official pipelines |
| user steps | tiny `jwstflow.Step` ABC or a plain function `f(inputs, ctx, **params) -> list[Path]`; discovered through the `jwstflow.steps` entry-point group or a dotted path | no plugin framework, just `importlib.metadata` |
| execution | stage DAG from `graphlib`, tasks in a `concurrent.futures` process pool (`spawn`) or `dask.distributed` | matches the jwst multiprocessing guidance; dask gives clusters/SLURM (via `dask-jobqueue`) without changing the model |
| checkpoints | one JSON record per task keyed by a hash of (stage, step, parameters, input fingerprints, jwst version, CRDS context) | inspectable and diffable with plain tools; no database |

**Why not Hydra / hydra-zen as the core?** Hydra is an *application*
framework: it owns `main()`, the working directory, logging and sweeps, and it
has no notion of a task graph or checkpoints. That fights an orchestrator that
must also run inside notebooks and be embeddable. OmegaConf alone is a weak
foundation too (stable release 2.3 dates from 2022; 2.4 is still a
pre-release). jwstflow therefore keeps the useful ideas (presets, overrides,
interpolation) in a few dependency-free functions. It stays **compatible**:
compose with OmegaConf or Hydra and hand the resulting dict to
`jwstflow.config_from_dict()`, see `examples/python_api.py`.

**Why not Snakemake / Prefect / Dagster?** They are excellent but heavy for
this scope, require learning their DSL/UI, and their task model does not map
cleanly onto "one association file -> one pipeline call". jwstflow tasks are
plain dictionaries executed by one function (`jwstflow.engine.executor.execute_task`),
so wrapping them in one of those systems later is straightforward.

## Projects, targets, runs

jwstflow is an installed library/CLI; your reductions live in a **project
repository of their own**, not inside the jwstflow checkout. A project is the
directory tree that holds your workflows (the nearest ancestor with `.git`,
`pyproject.toml`, `uv.lock` or a `.jwstflow-root` marker). Two things come
from the project root: every `.env` / `.env.*` file there is loaded (they win
over the shell unless `env_file_override: false`), and the default
`workspace` is `<project root>/reductions`. A minimal project:

```
my-survey/                              your repo; depends on jwstflow (+ contributed packages)
  pyproject.toml
  src/my_survey_steps/                  steps shared across the project's targets (optional)
  reductions/eso-ha-569/…               targets, workflows, and their products (below)
```

A workflow names a **target** and a **run**; a target directory holds the raw
data of all its observations and one run per reduction, and a run holds its
products by calibration level (version the YAMLs and `steps.py`, ignore
`raw/` and the run directories):

```
reductions/eso-ha-569/                 target: ESO-Ha 569  (slugified)
  nirspec_ifu.yaml  miri_mrs.yaml  combine.yaml  steps.py     the workflows and target-specific steps
  raw/                                  downloads of every instrument/observation of the target
  raw/mast_observations.json            MAST obs_ids (carry the DMS target id, e.g. t010)
  targets.json                          cached name -> coordinates lookups
  mast_reference/<run>/stage3/...       MAST's own products, opt-in, mirroring the run layout
  nirspec_ifu/                          one run
    stage2/flag_frames/  stage3/calwebb_spec3-pass1/  stage3/calwebb_spec3/
    stage4/in_field_background/  stage4/extract_extended/  qa/quicklook_image/
    associations/  logs/  .jwstflow/    per-stage associations, per-task logs, checkpoints
  miri_mrs/                             another run of the same target
  combine/                              a level-4 run reading from both (`inputs: [{run: nirspec_ifu, ...}]`)
```

## Where step code lives: the graduation ladder

Step code has a life cycle; keep each step at the lowest tier that fits it,
and promote it when it is reused:

1. **`steps.py` next to the workflow** (`plugins:`) -- genuinely one-off,
   target-specific science: a rejection list, a hand-tuned subclass of a
   contributed step. It is part of the reduction's provenance (its content is
   hashed into the task ids) and must stay **single-file and self-contained**;
   the moment it wants to import a sibling file, promote it.
2. **Your project's package** (`src/my_survey_steps/`) -- steps shared by
   several targets of one project. An ordinary package; reference steps by
   dotted path (`my_survey_steps.disks:DiskMask`) or register entry points.
3. **A contributed package** (`jwstflow-<name>` on PyPI or a git host) --
   steps useful beyond one project. See the next section.
4. **jwstflow itself** -- only broadly useful machinery: the toolkits steps
   are built from (`jwstflow.spectra`, `jwstflow.cubes`), the shared contracts
   below (`jwstflow.spectra.stitching`, `jwstflow.cubes.masks`,
   `jwstflow.cubes.psf`, `jwstflow.cubes.clusters`) and the stock QA steps
   (`jwstflow.qa.steps`).

There is no tier in between inside jwstflow: a step either is part of the
package, filed with the feature it implements (see "How the package is laid
out"), or it lives in a contributed package. "Contributed" always means a
separate distribution.

## Contributed packages

A contributed package is an ordinary distribution that **depends on
jwstflow** and registers its steps in the `jwstflow.steps` entry-point
group -- the core never references it, so absent or private packages can
never break anyone's install:

```toml
[project]
name = "jwstflow-midas"
dependencies = ["jwstflow"]

[project.entry-points."jwstflow.steps"]
disk_mask = "jwstflow_midas.steps.disk_mask:DiskMask"
```

Start a new package with the generator -- it writes the whole boilerplate
(pyproject with entry points, src layout with one module per step, step stubs whose `run()` awaits
your science, a passing declaration test, README with a publish-to-GitHub
walkthrough, CI workflow) and initialises a git repository:

```bash
jwstflow new-package jwstflow-mysteps --steps defringe,stitch_bands [--private]
```

The templates live inside jwstflow, so generated boilerplate always matches
the plugin contract of the installed version. Users install whichever
packages they have access to and reference the steps by entry-point name in
YAML; `jwstflow steps` lists everything installed, each with a short
description (`jwstflow steps <name>` or `--describe` shows the detailed
explanation, taken from the step's docstring).
Public packages come from PyPI or a public git URL; proprietary ones (e.g.
JOYS+) live in private repositories and install with
`uv pip install git+ssh://git@github.com/<org>/jwstflow-joys` (pin them in
your *project's* `tool.uv.sources`, never in jwstflow's). Entry-point names
are global across installed packages, so pick distinctive names and keep a
step's canonical `name` equal to its entry-point name. Contributed steps
should bump their `version` attribute when results change -- the module file
hash covers direct edits, but not edits to helper modules.

## Shared step contracts

Some product shapes recur across instruments and packages, so their layout is
fixed in core and plugins build on them instead of inventing variants:

* **Mask products** (`jwstflow.cubes.masks`): wavelength-resolved spatial masks for
  IFU cubes (a source aperture, a background exclusion zone, ...) in one FITS
  layout (MASK/CONT/CONTIMG + optional per-feature extensions), with
  `write_mask_product` / `read_mask_product` / `mask_from_stage` and the
  geometry helpers (WCS resampling, NaN-aware smoothing, morphological
  cleanup) such steps share. Any step can consume any other step's mask.
* **Stitching** (`jwstflow.spectra.stitching.StitchSegments`, YAML name
  `stitch_segments`): splice N overlapping 1-D segments (NIRSpec gratings,
  MRS bands) with flux ratios measured in a small window around each
  crossover (`ratio_window_frac` of the crossover wavelength) whose
  uncertainty -- propagated from the flux errors in that window -- travels
  into the stitched FLUX_ERROR, optional rescaling onto a reference
  segment, and configurable crossovers. Usable directly from YAML;
  contributed packages subclass it for mode-specific behaviour (naming via
  `segment_label`, the ratio measurement via `overlap_ratio`, grouping via a
  `run` wrapper -- jwstflow-joys' `stitch_bands` does all three).
* **PSF cubes** (`jwstflow.cubes.psf`): a distance-independent library of
  spectrally sub-sampled instrument PSFs matching an s3d cube, on a fixed
  fine angular grid in the instrument frame, in one FITS layout (PSF slices
  + WAVETAB) with resampling onto any model grid (scale + rotation to sky in
  one interpolation), wavelength interpolation and flux-conserving
  convolution built in -- what forward-modeling code convolves its model
  cubes with, at every trial distance, from one product. Generated by the
  `psf_cube` step (stpsf behind the optional `jwstflow[psf]` extra; NIRSpec
  IFU + MIRI MRS), plotted by `qa_psf_cube`; the rationale and conventions
  live in [psf_cubes.md](psf_cubes.md).

* **Bad spaxel clusters** (`jwstflow.cubes.clusters`): compact clusters of wildly
  deviant spaxels over a few planes that survive the official pipeline
  (IFU `outlier_detection` keeps the *minimum* neighbour difference over all
  exposures, so anything confined to one dither passes; drizzle then averages
  it in). Three steps go between stage 2 and stage 3 of any IFU workflow:
  the official `cube_build` on one association per dither
  (`group_by: [..., PATT_NUM]`, `variant: perdither` ->
  `stage3/cube_build-perdither/`), `flag_spaxel_clusters` (regions from the
  YAML -- sky position, search aperture, wavelength window -- compared
  across the per-dither cubes: which dithers carry the cluster, which
  spaxels, in one `*_clustermask.fits` product per dither cube) and
  `propagate_cluster_flags` (the cal files' WCS evaluated to find the
  detector pixels behind the flagged spaxels; DQ-flagged `_cal` copies that
  every later stage consumes). `qa_spaxel_clusters` draws every dither at the
  affected wavelengths. The product carries the mask contract plus the
  per-region decisions and QA slices.
* **The NIRSpec detector gap** (`jwstflow.cubes.gaps`): every IFU slice crosses
  the NRS1/NRS2 gap at its own wavelength and every dither moves a sky
  position onto other slices, so in a combined cube the exposures drop out
  one after another approaching the gap: the voxels of that staircase (some
  15-30 planes on each side, more at the field edges) are built from a few
  exposures down to single pixels, and are 2-5 times noisier than the same
  spaxels away from the gap. Their content is physical, though -- the part
  of the disk the moving gap has not yet covered, lines included -- so
  nothing is masked for coverage. `flag_gap_outliers` (after stage 2,
  `batch: all`) finds the detector pixels that build those voxels and tests
  each against the other exposures at its sky position and wavelength
  (where `outlier_detection` has the fewest exposures to compare): only
  pixels deviating by more than 5 sigma from at least two other exposures
  are flagged `DO_NOT_USE | OUTLIER` in DQ-edited `_cal` copies; untestable
  pixels are kept. One `*_gapcov.fits` product per group of exposures maps
  the coverage on the sky (`jwstflow.cubes.gaps.gap_limits_at` looks a
  position up: where only some exposures contribute at a wavelength), drawn
  by `qa_gap_coverage`. Files without a detector gap (MIRI, PRISM, medium
  gratings) pass unchanged:

  ```yaml
  - step: flag_gap_outliers            # -> stage2/flag_gap_outliers
    inputs: [{stage: calwebb_spec2, pattern: "*_cal.fits"}]
  - step: spec3
    inputs: [{stage: flag_gap_outliers, pattern: "*_cal.fits"}]
    # ...
  - step: qa_gap_coverage
    inputs: [{stage: flag_gap_outliers, pattern: "*_gapcov.fits"}]
    tags: [qa]
  ```

* **Joint cubes** (`jwstflow.cubes.stitching`) -- *in development: joint
  cubes are not science-ready products, and no reduction recipe builds them
  yet* --: the band cubes of one
  instrument -- the gratings of a NIRSpec IFU observation, the twelve bands
  of an MRS observation -- joined along wavelength into one cube
  (`*_s3dcomb.fits`), by the rules of the spectrum stitch and with every
  number left where the pipeline measured it:

  * *one spatial grid*: the finest spaxel scale among the bands over the
    union of their fields (for the MRS the 0.13" spaxels of channel 1 over
    the field of channel 4), on the pixel lattice of the finest band;
  * *no spectral resampling*: bands are ordered blue to red and the cube
    switches from one to the next at a crossover (default: the midpoint of
    their overlap); every plane is one plane of one band at its own
    wavelength, and the wavelength axis is the table of those wavelengths
    (`CTYPE3 = 'WAVE-TAB'`, as in the official multi-band cubes);
  * *overlap ratios as for spectra*: for each neighbouring pair the flux of
    both bands is measured over one region of sky -- where both are complete
    and the source stands out, whole sources only: a bright object the edge
    of the field cuts is held differently by each band and is left out --
    as the excess over the level of the field
    in each plane (the cube counterpart of an on/off extraction), and the
    ratio is taken in the window around the crossover with its propagated
    uncertainty (the same function `stitch_segments` uses,
    `jwstflow.spectra.overlaps`). Always recorded; with `rescale` applied as
    one factor per band, chained onto a reference band, its uncertainty
    folded into ERR. The level of the field in both bands is recorded next
    to it: two bands rarely agree in it, the difference is additive, and no
    factor removes it. Such a factor is well defined for a bright, compact
    source (it is then the factor of a stitched aperture spectrum of the
    same data, to about a percent, for NIRSpec gratings and MRS bands
    alike); for a faint or extended source it depends on where it is
    measured -- there `scales_from` adopts the scales of one stitched
    spectrum of the same data instead, or the bands are left as calibrated;
  * *positions checked, not adjusted*: each band is placed by its own WCS,
    and for every crossover the two bands are registered against each other
    at the same wavelengths (one window around the bright core they share,
    scale and level free); the shift is recorded. Two NIRSpec gratings agree
    to a few milliarcseconds; MRS bands differ by tens, and the joint cube
    carries that.

  The segments follow one another in wavelength (one lying inside another's
  range is refused) and a crossover lies inside the overlap of its pair, so
  no wavelength a band holds is left out.

  `cube_build` gives every band a grid of its own, so finished band cubes
  never share a pixel lattice. `stitch_cubes` takes them as they are and
  resamples those that are off the joint lattice with the overlap areas
  between spaxels as weights -- exact, flux-conserving, nothing interpolated
  -- but that is a second resampling and costs resolution (the three
  sub-bands of an MRS channel sit up to three quarters of a spaxel apart).
  The better way is to build the band cubes on the joint grid in the first
  place: `cube_build_matched` is the official `cube_build` with the grid
  prescribed (the joint grid of the cubes in `grid_from`, handed over as
  `scalexy`, `ra_center`, `dec_center`, `nspax_x`, `nspax_y`). It drizzles
  the detector pixels of every band once, straight onto the common grid,
  and `stitch_cubes` then copies spaxels; on the lattice of the reference
  band its cube is the pipeline's own, value for value. The product says
  for every band which way it came (SEGMENTS: `identity` or `regridded`):

  ```yaml
  - step: cube_build_matched           # -> stage3/cube_build_matched
    inputs: [{stage: calwebb_spec3, pattern: "*_crf.fits"}]     # what the band cubes were built from
    association: {mode: group, level: 3, asn_type: spec3, group_by: [PROGRAM, OBSERVTN, GRATING, FILTER],
                  product_name: "jw{PROGRAM}-o{OBSERVTN}_{TARGID}_nirspec_{GRATING}"}
    parameters: {grid_from: "${run_dir}/stage3/calwebb_spec3"}
  - step: stitch_cubes                 # -> stage4/stitch_cubes
    inputs: [{stage: cube_build_matched, pattern: "*_s3d.fits"}]
    parameters: {crossovers: [2.890], rescale: true}
  - step: qa_cube_stitch
    inputs: [{stage: stitch_cubes, pattern: "*_s3dcomb.fits"}]
    tags: [qa]
  ```

  (For the MRS: one association for all bands, `group_by: [PROGRAM,
  OBSERVTN]`, after the same `pixel_replace` the band cubes had.) The joint
  cube is an `IFUCubeModel` followed by the records of the stitch (SEGMENTS,
  OVERLAPS, the region spectra and images of every overlap), so it opens
  like any `_s3d` cube, the stock cube figures read it, and
  `qa_cube_stitch` draws the join, the overlaps and the positions from the
  product alone.

Extraction steps additionally share `jwstflow.cubes.apcorr`: the CRDS
aperture-correction reference (MIRI MRS and NIRSpec IFU layouts) normalised
to one plain-array table and evaluated per plane, so contributed extraction
steps do not each carry their own loader.

Stage names are **derived from the step**, never chosen in the YAML: the stpipe
`class_alias` (`calwebb_spec3`, `extract_1d`; a subclass inherits it unless it
sets its own), or a jwstflow step's `name` / entry-point name. Use
`variant: pass1` when the same step appears twice (`calwebb_spec3-pass1`). The
level directory comes from the step as well (1/2/3 for the jwst stages, 4 for
derived products, `qa` for plots). This keeps runs comparable across targets,
users and reductions; `--only`, `depends_on` and `inputs: [{stage: ...}]` use
the derived names, and `--set stages.<name>.…` accepts the name or the step.

## How the package is laid out

`src/jwstflow/` holds the command line and ten subpackages -- no loose
modules. Each subpackage is one of three things:

```
jwstflow/
  cli.py        the command line

                -- the step contract
  steps/        what a step is (interface.py), the file-naming rules it follows (naming.py),
                the adapter for the official pipelines (adapters.py), names -> steps
                (registry.py), user code by path (plugins.py), the texts of `jwstflow steps`
                (descriptions.py). No step is defined here.

                -- orchestration: runs any workflow, knows no science
  config/       the YAML: schema, loader (extends, interpolation, overrides), project root
                and env files (project.py), presets/
  data/         what is known about the observation outside the YAML: MAST (download.py), files
                and headers (discovery.py), target positions (targets.py), the MSA metafile (msa.py)
  engine/       the run model: runner, executor, stage graph, checkpoints (state.py),
                associations, the CRDS environment (crds.py), the workflow DAG figure (dagviz.py)

                -- science, by the kind of data: a toolkit for step authors, plus the bundled features
  spectra/      1-D spectra: the x1d format (x1d.py), line and band datasets (features.py),
                overlap ratios and chained scales (overlaps.py), the stitching contract and
                its step (stitching.py)
  cubes/        IFU cubes: WCS and regridding (wcs.py), the detector pixels behind a cube
                (detector.py), image operations (morphology.py), instrument matching
                (instrument.py), the mask-product contract (masks.py), aperture corrections
                (apcorr.py); psf/, clusters/, gaps/ and stitching/ (joint cubes), each with
                its library, its product and its steps.py
  qa/           the QA figure standard; qa/steps/ holds every QA step

                -- tools around it
  scaffold/     the templates of `new-step` and `new-package`
  testing/      run_step, synthetic products, check_step; testing/mock/: mock observations
                and stub pipelines
  refdata/      reference tables shipped with the package
```

**Who may import whom.** The dependencies run one way -- an arrow reads
"imports" -- and `tests/test_architecture.py` checks them on the source:

```
cli · testing · scaffold                            tools: may import anything below

engine              qa.steps · cubes.psf · cubes.clusters · cubes.gaps · cubes.stitching · spectra.stitching    the features
   |                qa · cubes · spectra                                                                        the toolkits
   |                     |
   +------> data <-------+           files and headers, MAST, target positions
              |
              v
            config                   the YAML, the project root
              |
              v
            steps                    the step contract: imports nothing
```

The two columns never import each other:

* `steps` imports no other part of the package.
* Orchestration (`config`, `data`, `engine`) never imports a science domain:
  the engine runs steps it knows only through the step contract.
* The science domains never import the engine: a step gets everything it
  needs through its `RunContext`.
* Within a domain, the toolkit does not depend on the features built with it
  (`cubes/wcs.py` knows nothing of `cubes/psf/`), and no feature imports
  another: what two of them need moves into the toolkit (`cubes/detector.py`
  serves both `clusters` and `gaps`, `spectra/overlaps.py` the stitch of
  spectra and the stitch of cubes). Only the QA steps look across -- they
  draw the products of the others.

**Where a step is defined.**

* A data step lives with the feature it implements: `stitch_segments` in
  `jwstflow.spectra.stitching`, `psf_cube` and `subtract_psf_neighbour` in
  `jwstflow.cubes.psf.steps`, the
  cluster steps in `jwstflow.cubes.clusters.steps`, `flag_gap_outliers` in
  `jwstflow.cubes.gaps.steps`, `stitch_cubes` in
  `jwstflow.cubes.stitching.steps` and `cube_build_matched` -- a subclass of
  the official `cube_build`, so a module of its own that imports jwst -- in
  `jwstflow.cubes.stitching.matched`, `fix_msa_metafile` in
  `jwstflow.data.msa`.
* Every QA step lives in `jwstflow.qa.steps`, one module per kind of product
  (`spectra`, `stitch`, `images`, `cubes`, `cube_stitch`, `psf`, `clusters`,
  `gaps`, `mast_compare`, `summary`).

jwstflow registers its own steps the way any step package does: as
`jwstflow.steps` entry points in `pyproject.toml`, and nowhere else. Each
entry names the module that *defines* the step, because that file's hash
enters the task ids and its docstring is what `jwstflow steps` prints
(`jwstflow check-step <name>` tells when an address leads to a module that
merely re-exports the class).

**What a step author imports.**

```python
from jwstflow import Step, StepParams, RunContext, qa
from jwstflow.steps.naming import derived_name, slug
from jwstflow.spectra import Spectrum1D, read_x1d, write_x1d
from jwstflow.spectra.features import feature_windows
from jwstflow.spectra.stitching import StitchSegments
from jwstflow.spectra.overlaps import measure_overlap_ratio, chain_scales
from jwstflow.cubes import celestial_wcs, cube_wavelengths, write_mask_product, load_apcorr
from jwstflow.cubes.stitching import SpatialGrid, joint_grid, Placement
from jwstflow.cubes.psf import PsfCubeProduct
from jwstflow.data.targets import resolve_source
```

**A step package** (`jwstflow-midas`, `jwstflow-joys`, yours) follows the same
conventions at its own scale:

```
jwstflow_<name>/
  steps/        one module per data step, named like the step (steps/disk_mask.py -> `disk_mask`)
  qa/           the QA steps
  <library>.py  code several steps share, by concern
```

## File names

Official products keep the names the pipeline gives them. Level-3 product
names follow the DMS convention `jw<PPPPP>-o<OOO>_<tTTT>_<instrument>` with the
target id taken from MAST at download time, so cubes come out named exactly as
in the archive. Custom steps derive names with `jwstflow.steps.naming.derived_name()`
and may not use a suffix the pipeline reserves (`_x1d`, `_s3d`, `_cal`, …);
jwstflow's own product types are `_s1d` (extracted spectra), `_s1dcomb`
(stitched spectra), `_s3dcomb` (joint cubes), `_bkgspec` (background
spectra), `_lsr`, `_psfcube` (PSF cubes), `_nbrfit` (a neighbour's PSF
fitted to and subtracted from a cube), `_clustermask` (bad spaxel
clusters of a per-dither cube) and `_gapcov` (coverage at the NIRSpec
detector gap). Edited copies of an official product (a DQ-flagged `_cal`) keep the
official name and record the edit in the header.

## Data products shipped with jwstflow

Curated reference tables (`spectral_features/*.ecsv` today) ship **inside**
the package (`jwstflow/refdata/`), so every install has them. A project can
override them with a `data/spectral_features/` directory at its project root,
or point `$JWSTFLOW_DATA_DIR` somewhere else; lookup order is env var,
project override, package. Observational data and run products are never
mixed into either.

## Prefetching and what runs in parallel

A run overlaps its network work with compute where that is safe:

* **MAST reference products** download in a background thread while the
  stages run; the run only waits for them when a `mast_compare` stage needs
  them (or at the very end).
* **CRDS references** are synced once, up front, for the raw files
  (`crds bestrefs --sync-references`); reference selection depends only on
  header keys the raw files already carry, so this one sync covers every
  later stage -- there is nothing left to look ahead for during the run.
  It runs *before* the first stage on purpose: several worker processes
  populating a shared (often NFS-mounted) CRDS cache concurrently is the
  classic way to corrupt it, so jwstflow never races the cache against
  running tasks.
* **Raw data** must exist before stage 1 by definition, and is verified for
  completeness at download time.

None of this needs to be requested: a plain `jwstflow run` on a clean
checkout downloads the data, syncs CRDS (on by default whenever an official
pipeline stage is in the workflow; a failed sync degrades to on-demand
fetching with a warning), fetches the reference products and runs the
stages -- one command is the whole reduction. `validate` and `prefetch` are
optional conveniences on top. To move all of the waiting off your compute
time entirely:

```bash
jwstflow prefetch reductions/eso-ha-569/miri_mrs.yaml   # network only: raw + CRDS + reference products
jwstflow run      reductions/eso-ha-569/miri_mrs.yaml   # starts computing immediately, needs no network
```

## The workflow DAG figure

Every run starts by rendering its own DAG into `qa/workflow_graph/`
(`workflow_graph: false` turns it off), and `jwstflow graph <wf>` draws it
without running anything. Nodes are the data products per file pattern
(cylinder: the MAST query; notes: patterns, dashed when they come from a
sibling run), and the steps coloured by calibration level (dashed when
disabled); dotted edges are `depends_on`, and a legend explains every colour
and style. The figure is drawn with matplotlib (already a dependency:
nothing extra to install, no system binaries) using a full Sugiyama layered
layout -- crossing minimisation, edges routed through waypoints between the
rows, and a straight main spine; any format matplotlib can save works
(`--format svg,pdf,png`). The `.dot` source is written alongside as a
portable text artifact.

## QA figures

All figures of a reduction come from QA steps (`level = "qa"`), never from
data steps, so each one lands in `qa/<step name>/` -- the subdirectory names
the step that made it. They all follow one standard (no titles, mJy flux
units, mid-point step plots, height-matched colorbars, nan-aware cube
collapses, the cmasher `torch` palette with black main lines, units in
square brackets), set in this repo and implemented by the `jwstflow.qa` package;
[docs/qa_figures.md](qa_figures.md) spells out the rules and shows how a QA
step uses the module. Contributed packages build their QA figures through
the same module, so `qa/` reads as one consistent report.

## Comparing with the archive (opt-in)

`download.reference_products: true` fetches MAST's own calibrated products of
the same observations into `<target>/mast_reference/<run>/` in the
background, laid out like the run itself (`stage2/calwebb_spec2/`,
`stage3/calwebb_spec3/`, …) so that a jwstflow product and its MAST
reference sit at the same relative path. A `provenance.json` records each
file's `CAL_VER`/`CRDS_CTX` and download time; the files are never mixed with
`raw/` or run outputs. A `mast_compare` stage then differences jwstflow's
`_s3d`/`_x1d` products against the reference file of the same relative path
and DMS name (a `calwebb_spec3-pass1` variant maps to `calwebb_spec3`):
`_s3ddiff.fits` (DIFF/RATIO cubes when the grids match, always a per-plane
SUMMARY table) and `_x1ddiff.fits` plus a PNG with both spectra and their
ratio, all carrying both provenances in the header. If MAST is unreachable
and the raw files are already on disk, the run continues with those.

## Positions and spectral features

Positions are never read from ad-hoc files. `target:` is resolved by name
(SIMBAD/Sesame, cached in `targets.json`) unless the workflow gives
`target_coords: {ra, dec}`, and steps can also use the observation's own
`TARG_RA/TARG_DEC`. Line and band lists ship as datasets in
`jwstflow/refdata/spectral_features/` (`gas_lines`, `pah_bands`, `ice_bands`;
ECSV with per-feature label, transition, source and notes, plus file-level
references and version). Steps select them with a small grammar:

```yaml
features: all
features: [H2, NeII, pah_7.7]                 # species or ids
features: {gas_lines: [H2], pah_bands: all, ice_bands: [co2_4.27], wave_min: 2.8}
```

## YAML reference

Top level:

```yaml
version: 1
target: ESO-Ha 569           # required; directory slug + name resolution
run: nirspec_ifu             # required; <workspace>/<target>/<run>/
target_coords: {ra: 167.79, dec: -76.70}   # optional: a position that is not a catalogue object
workspace: ./reductions      # default <project root>/reductions; `root:` sets an explicit run dir
extends: preset:nirspec_ifu  # or a path; chains are allowed
env: {OMP_NUM_THREADS: "1"}  # exported to every worker

crds:
  path: ~/crds_cache         # CRDS_PATH
  server_url: https://jwst-crds.stsci.edu
  context: null              # 'jwst_1364.pmap', or 'latest' (resolved once, pinned in manifest)
  prefetch: false            # crds bestrefs --sync-references before running
  disable_steppars: false    # STPIPE_DISABLE_CRDS_STEPPARS
  readonly_cache: false      # CRDS_READONLY_CACHE (shared cluster caches)

parallel:
  backend: process           # serial | process | dask
  workers: 4
  scheduler: null            # dask: tcp://host:8786, else a LocalCluster
  threads_per_worker: 1      # OMP/MKL/OPENBLAS limit per worker
  allow_nested_multiprocessing: false

logging: {level: INFO, console: true, per_task_logs: true}
checkpoint: {fingerprint: fast, require_outputs: true}   # fast = size+mtime, content = sha1

download:                    # optional; astroquery.mast by default
  program: 1234
  observations: [3, 4]
  instrument: NIRSPEC        # NIRSPEC | MIRI
  modes: [IFU]               # MAST sub-modes: IFU, MSA, SLIT, IMAGE, SLITLESS, MRS...
  products: [UNCAL]          # MAST productSubGroupDescription
  filename_patterns: null    # e.g. ['*_nrs1_*']
  backend: astroquery        # or jwst_mast_query (shells out to jwst_download.py)
```

Environment files and custom-step code:

```yaml
env_file: [../secrets/crds.env]   # optional; a `.env` next to the YAML, in any parent
                                  # directory, or in the cwd is loaded automatically
plugins: [./steps.py]             # files become importable by their stem, directories
                                  # go on sys.path -- in the workers as well. Keep plugin
                                  # files single and self-contained (see the graduation
                                  # ladder); shared code belongs in a package.
```

dotenv files (`.env`, `.env.crds`, `.env.mast`, … in the project root) hold
`CRDS_PATH`, `CRDS_SERVER_URL` and `MAST_API_TOKEN`; they are loaded before
`${env:...}` interpolation and override the shell (set `env_file_override:
false` for the opposite). `env_file:` adds explicit files, `$JWSTFLOW_ENV_FILE`
one more. `jwstflow validate` prints which files were loaded and whether a
MAST token is set; the token is optional for public data. No `PYTHONPATH` juggling is needed: point `plugins:` at
your code, or write the file path straight into the step
(`step: ./my_steps.py:MyStep`).

A stage:

```yaml
stages:
  - step: spec2              # alias | entry-point name | pkg.module:Object  -> stage `calwebb_spec2`
    variant: null            # short slug when the same step is used twice
    inputs:
      - {stage: calwebb_detector1, pattern: "*_rate.fits", filters: {EXP_TYPE: [NRS_IFU]}}
      - {run: miri_mrs, stage: mrs_extract, pattern: "*_s1d.fits"}   # a sibling run of the target
      - {path: /elsewhere, pattern: "*.fits", exclude: ["*_bad_*"], recursive: true}
    association:             # optional; turns the inputs into *_asn.json tasks
      mode: per_exposure     # per_exposure | group | official
      level: 2
      science_filters: {BKGDTARG: false, IS_IMPRT: false}     # default
      members:
        - exptype: imprint
          filters: {IS_IMPRT: true}
          match_on: [OBSERVTN, DETECTOR, GRATING, FILTER]
          differ_on: []      # e.g. [PATT_NUM] for nod backgrounds
      group_by: [PROGRAM, OBSERVTN, GRATING, FILTER]           # group mode
      product_name: "jw{PROGRAM}-o{OBSERVTN}_{TARGID}_nirspec_{GRATING}"   # DMS convention
    batch: per_file          # per_file | all; default: the step's own `batch` attribute
    parameters:              # stpipe: kwargs of Step.call(); nested steps: {...} for pipelines
      steps: {cube_build: {skip: true}}
    save_results: true
    parallel: true           # set false for stages using maximum_cores
    workers: null            # per-stage override
    force: false
    on_error: continue       # continue | fail
    depends_on: []           # extra ordering (inputs already imply ordering)
    tags: [qa]
```

`name`, `level` and `output_dir` are not settable: they follow from the step.

Header filters accept a scalar (equality, case-insensitive for strings), a
list (membership), `regex:...`, or `null` (keyword absent). Missing boolean
keywords count as `false`, so `IS_IMPRT: false` matches files without the
keyword.

Run `jwstflow schema -o jwstflow.schema.json` and keep the
`# yaml-language-server: $schema=jwstflow.schema.json` comment at the top of
your YAML for autocompletion and inline validation in VS Code and friends.

## Associations

* `per_exposure` (level 2): one association per science exposure, with
  background/imprint members attached by the `members` rules. Background
  exposures also get their own product so that level 3 can use them.
* `group` (level 3): one association per unique combination of `group_by`
  header values; `product_name` is a template over header keywords
  (`{OPTELEM}` becomes `g395h-f290lp` or `ch1-short` etc.). For IFU cubes
  follow the DMS convention and leave the band out of the product name
  (`..._nirspec_{GRATING}` for NIRSpec, `..._miri` for MRS): `cube_build`
  appends `_<grating>-<filter>` / `_ch1-short` itself. Association file
  names stay unique per group regardless of the product name.
* `official`: the jwst association generator with STScI's own DMS rule sets
  (`rules_level2b.py` / `rules_level3.py`) on a pool built from the headers.
  This reproduces MAST behaviour (verified against jwst 3.0.0 for NIRSpec IFU
  science + imprint + background exposures).

`jwstflow asn my_run.yaml spec3` builds and prints the associations of one
stage without running anything. Files are only rewritten when their content
changes, so checkpoints stay valid across reruns.

## Checkpointing

Every task's identity is the hash of its stage name, step, parameters, input
fingerprints and environment (jwst version, CRDS context, `save_results`,
parameter-reference flag). On a rerun a task is skipped when a success record
with that id exists and its outputs are still present. Consequently:

* changing a parameter reruns exactly the stages whose parameters changed and
  everything downstream of their new outputs;
* a re-downloaded or edited raw file reruns only the tasks touching it;
* upgrading `jwst` or pinning a different CRDS context reruns everything;
* editing the Python file that defines a custom step reruns that step's tasks
  (the file's hash is part of the task id; official jwst steps are covered by
  the jwst version instead), and editing *any* `plugins:` file reruns every
  plugin-defined step's tasks (plugin files may import each other). The hash
  is of the file's *content*: moving or renaming a module reruns nothing, any
  edit to it -- a comment included -- reruns the step. Helper modules a step
  imports are not hashed; bump the step's `version` when they change results;
* `--force`, `force: true` on a stage, or `jwstflow clean` override this.

stpipe stages write into a private `<stage>/.tmp-<task>/` directory that is
moved into place on success, so concurrent tasks never mix up each other's
products and a crash leaves no half-written files.

## Parallelism

`parallel.backend: process` runs tasks in a `spawn`-started pool and imports
the pipeline inside the worker, following the jwst multiprocessing notes.
Because the jwst docs state that step-level multiprocessing
(`maximum_cores`) must not be combined with exposure-level parallelism, a
stage using `maximum_cores` other than `none` is rejected at validation time
unless the stage has `parallel: false` or you set
`parallel.allow_nested_multiprocessing: true`.

`backend: dask` runs the same tasks on a `LocalCluster` or an existing
scheduler (`scheduler: tcp://...`), which is the route to SLURM/PBS via
`dask-jobqueue`. `backend: serial` runs everything in-process (notebooks,
`pdb`).

## Python API

```python
from jwstflow import load_config, Runner, config_from_dict

cfg = load_config("my_run.yaml", overrides=["parallel.backend=serial"])
runner = Runner(cfg, only=["spec2", "spec3"])
summary = runner.run()        # RunSummary with per-stage counts and failures
print(summary.ok)
```

## Testing status

`python -m pytest` runs the suite in `tests/` plus the example-step tests in
`examples/`: the mask-product contract (`jwstflow.cubes.masks`), the generic
stitcher (`jwstflow.spectra.stitching`), x1d writing with extra columns, the packaged
reference data and its project override, the contributed-package generator
(`jwstflow new-package`, both flavours, generated steps audited against the
contract), and the three example custom-step flavours end to end on synthetic
data. Contributed packages carry their own suites in their own repositories
and are run there.

Not exercised by tests in this repository: the configuration/association/
runner internals (an earlier private suite; re-adding it is on the roadmap),
end-to-end runs of the official pipelines on real data, the MAST download
backends, `crds.prefetch`, and the dask backend. Current development is
verified against jwst 3.0.0 / stpipe 1.1.0.

## Roadmap

* streaming DAG (start spec2 on an exposure as soon as its detector1 finished)
  instead of stage-level barriers
* `dask-jobqueue` presets for SLURM/PBS
* NIRCam / NIRISS presets (the machinery is instrument-agnostic)
* HTML run report from the checkpoint records
