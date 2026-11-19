"""Run one step the way a worker would, on files of your choice."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from ..steps.base import FunctionStep, RunContext, Step, check_outputs, make_step


def make_context(tmp_path: Path, *, stage: str = "test_step", target: str = "Test Target",
                 target_coords: dict[str, Any] | None = None) -> RunContext:
    """A RunContext rooted at ``tmp_path/<target>/<run>`` with the usual directories created."""
    target_dir = Path(tmp_path) / "test-target"
    run_dir = target_dir / "test_run"
    out = run_dir / "stage4" / stage
    for d in (target_dir / "raw", out, run_dir / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    return RunContext(run_name="test-target/test_run", root=run_dir, stage=stage, output_dir=out, log_dir=run_dir / "logs",
                      raw_dir=target_dir / "raw", stage_dirs={}, crds_context=None, dry_run=False, task_id="test",
                      target=target, target_coords=target_coords, target_dir=target_dir,
                      reference_dir=target_dir / "mast_reference" / "test_run")


def run_step(step: str | type | Step | Any, inputs: list[Path] | Path, tmp_path: Path, *,
             params: dict[str, Any] | None = None, ctx: RunContext | None = None, **ctx_kwargs: Any) -> list[Path]:
    """Run ``step`` (spec string, class, instance or function) on ``inputs`` like a worker would:
    parameters validated against ``Params``, outputs checked (exist, in the output directory,
    non-reserved suffix). Returns the outputs."""
    obj = make_step(step) if isinstance(step, str) else step
    if inspect.isclass(obj):
        obj = obj()
    elif not isinstance(obj, Step) and callable(obj):
        obj = FunctionStep(obj)
    ctx = ctx or make_context(tmp_path, **ctx_kwargs)
    files = [Path(p) for p in ([inputs] if isinstance(inputs, (str, Path)) else inputs)]
    validated = obj.validate_params(dict(params or {}))
    outputs = obj.run(files, ctx, **validated)
    return check_outputs(outputs, ctx, obj)
