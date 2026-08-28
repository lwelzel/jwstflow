# jwstflow guide

Everything beyond the [README](../README.md): the run model, naming, data
products, scheduling and the full YAML reference.

## Why this design

We wanted to build a light-weight, maintainable, compartmentalised, `jwst` orchestrator,
with a clean YAML interface, and built on well-maintained libraries.

| concern | choice | why |
|---|---|---|
| configuration | pydantic v2 models are the single source of truth; YAML via PyYAML | validation with readable errors, defaults, docstrings, and a JSON Schema for editor autocompletion all come from one class hierarchy |
| composition | ~150 lines of own code: `extends:` (files or `preset:<name>`), deep merge with stages merged by name, `${a.b}` / `${env:VAR,default}` interpolation, CLI dot-list overrides | these are the four Hydra features people actually use; no framework needed |
| unit of work | `stpipe.Step.call()` | pulls CRDS parameter-reference files (`pars-*`), so results match STScI's defaults; any `Step` subclass works, including subclasses of the official pipelines |
| user steps | tiny `jwstflow.Step` ABC or a plain function `f(inputs, ctx, params) -> list[Path]`; discovered through the `jwstflow.steps` entry-point group or a dotted path | no plugin framework, just `importlib.metadata` |
| execution | stage DAG from `graphlib`, tasks in a `concurrent.futures` process pool (`spawn`) or `dask.distributed` | matches the jwst multiprocessing guidance; dask gives clusters/SLURM (via `dask-jobqueue`) without changing the model |
| checkpoints | one JSON record per task keyed by a hash of (stage, step, parameters, input fingerprints, jwst version, CRDS context) | inspectable and diffable with plain tools; no database |

## Projects, targets, runs

jwstflow works inside a project: the directory tree that holds your
workflows (a git checkout or uv project; the nearest ancestor with `.git`,
`pyproject.toml`, `uv.lock` or a `.jwstflow-root` marker). Two things come
from the project root: every `.env` / `.env.*` file there is loaded (they win
over the shell unless `env_file_override: false`), and the default
`workspace` is `<project root>/reductions`.

A workflow names a target and a run; a target directory holds the raw
data of all its observations and one run per reduction, and a run holds its
products by calibration level:

```
reductions/eso-ha-569/                 target: ESO-Ha 569  (slugified)
  nirspec_ifu.yaml  miri_mrs.yaml  combine.yaml  steps.py     the workflows and custom steps
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

Stage names are derived from the step, never chosen in the YAML: the stpipe
`class_alias` (`calwebb_spec3`, `extract_1d`; a subclass inherits it unless it
sets its own), or a jwstflow step's `name` / entry-point name. Use
`variant: pass1` when the same step appears twice (`calwebb_spec3-pass1`). The
level directory comes from the step as well (1/2/3 for the jwst stages, 4 for
derived products, `qa` for plots). This keeps runs comparable across targets,
users and reductions; `--only`, `depends_on` and `inputs: [{stage: ...}]` use
the derived names, and `--set stages.<name>.…` accepts the name or the step.

## File names

Official products keep the names the pipeline gives them. Level-3 product
names follow the DMS convention `jw<PPPPP>-o<OOO>_<tTTT>_<instrument>` with the
target id taken from MAST at download time, so cubes come out named exactly as
in the archive. Custom steps derive names with `jwstflow.naming.derived_name()`
and may not use a suffix the pipeline reserves (`_x1d`, `_s3d`, `_cal`, …);
jwstflow's own product types are `_s1d` (extracted spectra), `_s1dcomb`
(stitched), `_bkgspec` (background spectra) and `_lsr`. Edited copies of an
official product (a DQ-flagged `_cal`) keep the official name and record the
edit in the header.

## Data products shipped with jwstflow

Reference data lives outside the code, in `data/` at the project root
(e.g. `data/spectral_features/*.ecsv`). It is found through
`$JWSTFLOW_DATA_DIR`, then `<project root>/data`, then the `data/` directory
of the jwstflow checkout; a wheel-only install needs one of the first two.

## Prefetching and what runs in parallel

A run overlaps its network work with compute where that is safe:

* MAST reference products download in a background thread while the
  stages run; the run only waits for them when a `mast_compare` stage needs
  them (or at the very end).
* CRDS references are synced once, up front, for the raw files
  (`crds bestrefs --sync-references`); reference selection depends only on
  header keys the raw files already carry, so this one sync covers every
  later stage -- there is nothing left to look ahead for during the run.
  It runs before the first stage on purpose: several worker processes
  populating a shared (often NFS-mounted) CRDS cache concurrently is the
  classic way to corrupt it, so jwstflow never races the cache against
  running tasks.
* Raw data must exist before stage 1 by definition, and is verified for
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
`jwstflow/data/spectral_features/` (`gas_lines`, `pah_bands`, `ice_bands`;
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
plugins: [./my_steps.py, ./lib]   # files become importable by their stem, directories
                                  # go on sys.path -- in the workers as well
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
  the jwst version instead);
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

#### WIP
`jwstflow` aims to be covered by automated tests. Without `jwst` installed they cover
configuration composition and validation, header filters, association
building (imprints, nod backgrounds, grouping, product names), the runner on
the serial and process backends (checkpoint reuse, invalidation, `--force`,
`on_error`, stage selection) and the CLI. With `jwst` installed two more run:
a real `JwstStep` through the adapter in a spawned process pool (outputs,
per-task logs, cache hits), and the official DMS association generator at
levels 2 and 3. These were run against jwst 3.0.0 / stpipe 1.1.0.

Not exercised by tests (due to required network/CRDS access):
end-to-end runs of the official pipelines on real data, the MAST download
backends, `crds.prefetch`, and the dask backend.

## Roadmap

* streaming DAG (start spec2 on an exposure as soon as its detector1 finished)
  instead of stage-level barriers
* `dask-jobqueue` presets for SLURM/PBS
* NIRCam / NIRISS presets (the machinery is instrument-agnostic)
* HTML run report from the checkpoint records
