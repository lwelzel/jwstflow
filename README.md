<p align="center">
  <img src="docs/jwstflow_logo.png" alt="jwstflow" width="480">
  <!-- replace docs/jwstflow_logo.png with the real logo -->
</p>

<p align="center">
  <a href="https://github.com/spacetelescope/jwst"><img alt="jwst" src="https://img.shields.io/badge/jwst-3.x-blue"></a>
  <img alt="python" src="https://img.shields.io/badge/python-%E2%89%A53.12-blue">
  <img alt="tests" src="https://img.shields.io/badge/tests-80%20passing-brightgreen">
  <img alt="status" src="https://img.shields.io/badge/status-alpha-orange">
</p>

**jwstflow** (pronounced justflow) turns a JWST reduction into one declarative YAML file and one
command. It orchestrates the official STScI [`jwst`](https://github.com/spacetelescope/jwst)
pipeline and adds the parts the pipeline leaves to you: MAST download, 
CRDS pinning and prefetch, association building, DMS-compliant 
product naming, checkpointed parallel execution that resumes 
where it stopped, structured per-target/per-run output directories, 
quality assurance plots, and automatic comparison against the archive's own
products. Reductions become reproducible, restartable and identical across
machines and collaborators, while every calibration decision stays a visible,
version-controlled line of YAML. Custom science steps (extractions,
background models, spectral post-processing) plug in through a small
contract and run as first-class stages next to the official ones.

## Installation

```bash
git clone https://github.com/lwelzel/jwstflow.git && cd jwstflow
uv sync                    # core: jwstflow + the official jwst pipeline
uv sync --all-extras       # + proprietary collaboration contributed steps if available
```

Put your credentials in the project root (they are found automatically):
`.env.crds` with `CRDS_PATH=...`, and `CRDS_SERVER_URL=...`, as well as
`.env.mast` with `MAST_API_TOKEN=...` (optional for public data). 
`uv run jwstflow --help` shows the available commands.

## Usage

### One command per reduction

```bash
uv run jwstflow run PATH/TO/OUTPUT/.../workflow.yaml # path pointing at the workflow.yaml
```

A run downloads what it needs, syncs CRDS, executes the stages in parallel
with checkpointing, and can be interrupted and rerun at any time. Finished
work is not unnecessarily repeated. A workflow YAML names a `target` and a `run`,
optionally a `download:` block (program/observations/instrument), and a list
of `stages:`, each an official pipeline (`detector1`, `spec2`, `spec3`, ...)
or a custom step, with its inputs and parameters:

```yaml
target: ESO-Ha 569
run: nirspec_ifu
download: {program: 1751, observations: [6], instrument: NIRSPEC, modes: [IFU], products: [UNCAL]}
stages:
  - step: detector1
    inputs: [{stage: raw, pattern: "*_uncal.fits"}]
  - step: spec2
    inputs: [{stage: calwebb_detector1, pattern: "*_rate.fits"}]
    parameters: {steps: {clean_flicker_noise: {skip: false, fit_method: fft}}}
  - step: spec3
    inputs: [{stage: calwebb_spec2, pattern: "*_cal.fits"}]
    association: {mode: group, level: 3, group_by: [PROGRAM, OBSERVTN, GRATING, FILTER]}
```

Outputs land in `PATH/TO/OUTPUT/<target>/<run>/` split by calibration level
(`stage1/ ... stage4/`, `qa/`), raw data is shared per target, and stage
names come from the steps themselves so every run looks the same everywhere.
Start from a preset (`extends: preset:nirspec_ifu`; also `nirspec_mos`,
`miri_mrs`, `miri_imaging`) and override only what your data needs.

### The rest of the interface

| command | purpose |
|---|---|
| `validate <wf> [--resolve]` | check the YAML, env files, CRDS/MAST setup and (with `--resolve`) that every step imports |
| `plan <wf>` | show every task that would run, and what is cached, without running |
| `run <wf> [--only/--from/--until STAGE] [--task GLOB] [--force] [--set key=value]` | run all or part of the workflow |
| `status <wf>` | per-stage table of done/failed/pending tasks |
| `prefetch <wf>` | do all network work (data, CRDS, reference products) without running stages |
| `clean <wf> [--stage S] [--orphans]` | delete outputs / stray files so stages rerun |
| `debug-task <wf> <task-id>` | rerun one task in-process (drop into `pdb` on failure) |
| `steps [--describe NAME]` | list every available step: official, contributed, plugins |
| `check-step SPEC` / `new-step Name` | audit a custom step against the contract / scaffold one with a test |
| `init [preset]` | write a starter workflow |

Everything is also a Python API (`load_config`, `Runner`); see
[docs/guide.md](docs/guide.md) for the run model, naming rules, the full YAML
reference, associations, checkpointing and parallelism.

## Custom steps

Any part of a reduction can be replaced or extended: subclass an official
pipeline to change its behaviour (it keeps its stage identity), or write a
new step, a small class declaring what it consumes and produces, with typed
parameters and a `run()` method:

```python
class ExtractExtended(Step):
    """Sum an extended source over an aperture mask."""
    level, inputs, outputs = 4, ("*_s3d.fits",), ("s1d",)
    class Params(StepParams):
        threshold: float = Field(0.05, gt=0)
    def run(self, inputs, ctx, *, threshold=0.05, **params):
        out = ctx.derived_path(inputs[0], "s1d", descriptor="extended")
        ...
        return [out]
```

jwstflow validates the declaration at import, the parameters at config load,
and the outputs after every run. `jwstflow new-step` scaffolds a step with a passing test
and `jwstflow.testing` runs steps on synthetic data without CRDS or real
files. Details in [docs/custom_steps.md](docs/custom_steps.md).

## Citing

There is no jwstflow publication yet. If jwstflow contributed to your
research, please acknowledge it, e.g.:

> This work made use of jwstflow (L. Welzel), an orchestrator for the JWST calibration pipeline.

and cite the [`jwst` pipeline](https://github.com/spacetelescope/jwst)
version and CRDS context recorded in your run manifest
(`<run>/.jwstflow/manifest.json`). If a jwstflow paper appears, this section
will change to the reference, please check back before submitting.

## Author & license

Lukas Welzel (<welzel@strw.leidenuniv.nl>), Leiden Observatory.
Issues and contributions are welcome; run `python -m pytest` (core) and
`python -m pytest contrib_packages/jwstflow-joys` before a PR.
License: to be decided before public release.
