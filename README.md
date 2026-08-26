# jwstflow

A lightweight, YAML-driven orchestrator for the official [JWST calibration
pipeline](https://jwst-pipeline.readthedocs.io/) (`jwst`/`stpipe`), initially
targeting **NIRSpec** and **MIRI**.

```
download (MAST) -> CRDS setup -> detector1 -> spec2/image2 -> spec3/image3 -> your own steps
     |                              |______ associations, checkpoints, parallel workers ______|
```

jwstflow does *not* reimplement any calibration. Every stage is an ordinary
`stpipe.Step`/`Pipeline` class (the official ones, or your subclass of them)
or a small Python function. jwstflow's job is everything around them: which
files go in, how they are grouped into associations, where outputs go, what
has already been done, and how many run at once.

* one YAML file describes a run, validated with readable errors and editor
  autocompletion (JSON Schema)
* presets for NIRSpec IFU / MOS / FS and MIRI MRS / LRS / imaging, which you
  `extend` rather than copy
* automatic checkpointing: rerun the same file and only what changed
  (inputs, parameters, jwst version, CRDS context) is recomputed
* parallel execution with a `spawn` process pool (as the jwst docs
  recommend) or `dask.distributed` for clusters; serial mode for notebooks
* one log file per task capturing the full stpipe/jwst/CRDS output
* association files built from FITS headers (per exposure, grouped, or via
  the official DMS rules), rewritten only when their content changes

## Installation

```bash
pip install -e .[all]          # jwst, astroquery, dask, matplotlib
pip install -e .[jwst,mast]    # or pick what you need
pip install -e .[dev]          # + pytest
```

`jwstflow` itself only depends on `pydantic`, `pyyaml`, `typer`, `rich`
and `astropy`, so validating and planning a workflow works on a laptop
without the pipeline installed. Python >= 3.11.

## Quick start

```bash
jwstflow init nirspec_ifu --extend -o my_run.yaml   # a YAML that extends the preset
$EDITOR my_run.yaml                                 # program id, root dir, CRDS path
jwstflow validate my_run.yaml --resolve             # schema + stage graph + step imports
jwstflow plan my_run.yaml                           # what would run (nothing executes)
jwstflow run my_run.yaml                            # download, calibrate, plot
jwstflow status my_run.yaml --failed                # what went wrong, with log paths
```

`my_run.yaml` after `init --extend`:

```yaml
extends: preset:nirspec_ifu
name: my_run
root: /data/jwst/my_run
crds: {path: ${env:CRDS_PATH,~/crds_cache}, context: latest}
download: {program: 1234, observations: [3, 4]}
parallel: {workers: 6}
stages:
  - name: spec3            # merged with the preset's spec3 by name
    parameters: {steps: {cube_build: {coord_system: ifualign}}}
```

Useful `run` options: `--set key.path=value` (any number, e.g.
`--set stages.spec2.parameters.steps.cube_build.skip=true`), `--only spec2`,
`--from spec2`, `--until spec2`, `--tag qa`, `--force`, `--dry-run`,
`--workers 8`, `--skip-download`.

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

## How a run is laid out

```
<root>/
  raw/                       downloaded (or your own) *_uncal.fits
  detector1/  spec2/  spec3/ one directory per stage, named after the stage
  associations/<stage>/      *_asn.json files built for association stages
  logs/jwstflow.log          run log
  logs/<stage>/<task>.log    everything the pipeline printed for that task
  .jwstflow/
    state/<stage>/<task>.json  checkpoint records (status, outputs, timings, error)
    manifest.json              per run: config hash, jwst version, pinned CRDS context
    config.resolved.yaml       the fully resolved configuration that was executed
    headers.json               FITS header cache used for filtering/grouping
```

## YAML reference

Top level:

```yaml
version: 1
name: my_run                 # required
root: /data/jwst/my_run      # required; relative paths are relative to the YAML file
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

A stage:

```yaml
stages:
  - name: spec2              # unique; also the output directory name
    step: spec2              # alias | entry-point name | pkg.module:Object
    inputs:
      - {stage: detector1, pattern: "*_rate.fits", filters: {EXP_TYPE: [NRS_IFU]}}
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
      product_name: "jw{PROGRAM}-o{OBSERVTN}_{TARGPROP}_nirspec_{OPTELEM}"
    batch: per_file          # per_file | all (user steps that combine inputs)
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
  (`{OPTELEM}` becomes `g395h-f290lp` or `ch1-short` etc.).
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

## Custom steps

Anything that subclasses `stpipe.Step` is run through `Step.call()` with your
`parameters:`; subclassing an official pipeline is the intended way to extend
it:

```python
from jwst.pipeline import Detector1Pipeline

class MyDet1(Detector1Pipeline):
    spec = """
    qa_summary = boolean(default=True)
    """
    def process(self, input):
        self.jump.expand_large_events = True
        result = super().process(input)
        ...
        return result
```

```yaml
- name: detector1
  step: my_steps:MyDet1           # PYTHONPATH must contain my_steps.py
  parameters: {qa_summary: true, steps: {ramp_fit: {maximum_cores: none}}}
```

Non-pipeline work (plots, tables, custom fixes) uses `jwstflow.Step` or a
function; both receive the input files and a `RunContext` (run name, root,
stage, output_dir, raw_dir, CRDS context, task id) and return the files they
produced, which become the inputs of downstream stages:

```python
from jwstflow import Step, RunContext

class SpectrumReport(Step):
    batch = "all"                      # one task with every input
    def run(self, inputs, ctx: RunContext, *, min_snr=3.0, **params):
        out = ctx.output_dir / "report.json"
        ...
        return [out]

def tag_header(inputs, ctx, *, keyword="X", value="1"):
    ...
    return outputs
```

Register names for your package in `pyproject.toml`:

```toml
[project.entry-points."jwstflow.steps"]
spectrum_report = "my_pkg.steps:SpectrumReport"
```

`examples/my_steps.py` and `examples/custom_steps.yaml` show all three
flavours; `jwstflow.contrib` contains ready-made QA steps (`plot_spectrum`,
`quicklook_image`, `header_summary`) and `jwstflow.contrib.nirspec.fix_msa_metafile`
for MOS runs whose `MSAMETFL` paths need fixing.

## Python API

```python
from jwstflow import load_config, Runner, config_from_dict

cfg = load_config("my_run.yaml", overrides=["parallel.backend=serial"])
runner = Runner(cfg, only=["spec2", "spec3"])
summary = runner.run()        # RunSummary with per-stage counts and failures
print(summary.ok)
```

## Testing status

`python -m pytest` runs 24 tests. Without `jwst` installed they cover
configuration composition and validation, header filters, association
building (imprints, nod backgrounds, grouping, product names), the runner on
the serial and process backends (checkpoint reuse, invalidation, `--force`,
`on_error`, stage selection) and the CLI. With `jwst` installed two more run:
a real `JwstStep` through the adapter in a spawned process pool (outputs,
per-task logs, cache hits), and the official DMS association generator at
levels 2 and 3. These were run against jwst 3.0.0 / stpipe 1.1.0.

Not exercised by tests (no network/CRDS in the development environment):
end-to-end runs of the official pipelines on real data, the MAST download
backends, `crds.prefetch`, and the dask backend.

## Roadmap

* streaming DAG (start spec2 on an exposure as soon as its detector1 finished)
  instead of stage-level barriers
* `dask-jobqueue` presets for SLURM/PBS
* NIRCam / NIRISS presets (the machinery is instrument-agnostic)
* HTML run report from the checkpoint records
