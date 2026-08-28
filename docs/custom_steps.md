# Custom steps

How to replace official pipeline steps or add your own, and how jwstflow
validates and tests them. See also `examples/my_steps.py` and
`examples/test_my_steps.py`.

## Custom steps

A custom step is a Python class (or, for one-liners, a function) that
jwstflow can validate, document and test without running it. The contract:

```python
from pydantic import Field
from jwstflow import RunContext, Step, StepParams

class ExtractExtended(Step):
    """Sum an extended source over an aperture mask."""   # first line = description

    level = 4                       # 1/2/3 jwst stages, 4 derived products, "qa" plots -> directory
    batch = "per_file"              # or "all": one task receives every input
    inputs = ("*_s3d.fits",)        # accepted files (documentation + planning-time check)
    outputs = ("s1d",)              # suffixes written; a reserved jwst suffix is refused
    version = "1"                   # bump when results change: existing tasks rerun

    class Params(StepParams):       # the `parameters:` block, checked by `jwstflow validate`
        threshold: float = Field(0.05, gt=0, description="aperture threshold (fraction of peak)")

    def run(self, inputs, ctx: RunContext, *, threshold: float = 0.05, params):
        (cube,) = inputs
        out = ctx.derived_path(cube, "s1d", descriptor="extended")   # naming rules applied
        ...
        return [out]                # every file written
```

What jwstflow enforces, and when:

| check | when | on failure |
|---|---|---|
| `level`/`batch`/`name` valid, `outputs` not a jwst suffix, `Params` forbids unknown keys | class definition (import) | `TypeError` with the reason |
| `parameters:` match `Params` (types, ranges, no typos) | `jwstflow validate`, config load | `ConfigError` naming the stage and field |
| inputs match the declared `inputs` patterns | planning | warning |
| returned files exist, lie in the stage directory, use no reserved suffix | after `run` | task fails with the reason |
| files written but not returned | after the stage | orphan warning |

The context gives a step everything it may need: `ctx.output_dir`,
`ctx.derived_path()`, `ctx.log`, `ctx.raw_dir`, `ctx.target` /
`ctx.target_coords` / `ctx.target_dir` (for `jwstflow.targets`), `ctx.reference_dir`,
`ctx.crds_context`. Shared building blocks live in `jwstflow.spectra` (x1d
I/O, cube wavelengths), `jwstflow.features` (line/band datasets),
`jwstflow.targets` (positions) and `jwstflow.naming`.

Steps that deliberately write *edited copies of official products* under the
official name (a DQ-flagged `_cal`) set `writes_official_products = True` and
record the edit in the header; everything else uses its own suffix.

Tooling:

```bash
jwstflow new-step ExtractExtended --dir reductions/my-target --level 4 --inputs "*_s3d.fits" --suffix s1d
                                   # writes extract_extended.py + test_extract_extended.py (both pass)
jwstflow check-step ./steps.py:ExtractExtended mrs_extract spec3
                                   # audits declaration, run() signature, Params vs keywords
python -m pytest reductions/my-target
```

`jwstflow.testing` backs the tests: `run_step(step, inputs, tmp_path, params=...)`
runs a step exactly as a worker would (validated parameters, output checks),
and `synthetic_image` / `synthetic_cube` / `synthetic_x1d` write small
JWST-like files so tests need neither real data nor CRDS. `examples/my_steps.py`
and `examples/test_my_steps.py` show all three flavours with their tests.

Subclassing an official pipeline is the way to extend jwst itself:

```python
from jwst.pipeline import Detector1Pipeline

class MyDet1(Detector1Pipeline):      # inherits class_alias calwebb_detector1 -> same stage
    spec = """
    qa_summary = boolean(default=True)
    """
    def process(self, input):
        self.jump.expand_large_events = True
        return super().process(input)
```

Register names for a package in `pyproject.toml` so workflows can use them by name:

```toml
[project.entry-points."jwstflow.steps"]
extract_extended = "my_pkg.steps:ExtractExtended"
```

## Developing custom steps (the debug loop)

1. Run everything upstream once, then iterate on one stage:
   `jwstflow run my_run.yaml --only extract`. Upstream stages are cached, the
   stage reruns whenever its code, parameters or inputs changed.
2. Narrow further with `--task "*g395h*"` (glob on the task label) and use
   `--set parallel.backend=serial` so exceptions and `pdb` behave normally.
3. Every task's full output is in `logs/<stage>/<task>.log`;
   `jwstflow status my_run.yaml --failed` lists failures with the log path and
   the traceback is stored in `.jwstflow/state/<stage>/<task>.json`.
4. In a notebook, run a task by hand with the same inputs and context the
   worker would get:

   ```python
   from jwstflow import Runner, load_config
   runner = Runner(load_config("my_run.yaml"))
   step, inputs, ctx, params = runner.debug_task("extract", "*g395h*")
   outputs = step.run(inputs, ctx, params)      # set breakpoints inside your step
   ```

While workers are busy the console is quiet; `tail -f logs/<stage>/<task>.log`
shows what a task is doing. The run announces each slow step (CRDS context
resolution, MAST query, reference prefetch, input selection) before starting
it; CRDS and astroquery print their own lines in between.

If a stage's directory contains files that no current task produced (left
over from a run with other parameters or product names), the run warns and
`jwstflow clean my_run.yaml --stage <name> --orphans` deletes just those files.

`reductions/eso-ha-569/` is a complete real case: `nirspec_ifu.yaml` (MAST
level-2 inputs, custom frame/gap rejection, in-field background fed to the
official `master_background` step through a `Spec3Pipeline` subclass,
extended-source extraction, stitching), `miri_mrs.yaml` (background
observation, `selfcal` members, jwstflow-joys extraction) and `combine.yaml`
(a level-4 run stitching both instruments).

## Contributed step packages

Steps that belong to a collaboration or project live in their own package
and register through the `jwstflow.steps` entry-point group; installing the
package is all it takes for `step: <name>` to work. `contrib_packages/jwstflow-joys`
is the first: MIRI MRS multi-aperture extraction, band stitching, spike
cleaning, LSR correction, region masking and Gaia astrometry, re-implemented
from the JOYS+ `icebear` scripts (its `REVIEW.md` documents what was kept
and why the rest was not).

`jwstflow.spectra` offers the shared building blocks for such steps: x1d
reading/writing in the pipeline's format, cube wavelength axes, the MRS PSF
FWHM relation.

