"""Step abstraction and registry.

There are two kinds of steps, and one interface:

* **stpipe steps** -- any subclass of ``stpipe.Step`` (the official pipelines,
  individual steps, or your own subclass such as ``class MyDet1(Detector1Pipeline)``).
  They are wrapped by :class:`JwstStepAdapter`, which calls ``Step.call(...)``
  (the preferred API: it pulls CRDS parameter-reference files) and reports the
  files it produced.
* **user steps** -- anything else: a subclass of :class:`Step` or a plain
  function ``f(inputs, ctx, **params) -> list[Path]``. Plotting, QA, custom
  destriping ... whatever you like.

Both are addressed in YAML by ``step:``, using an alias, an entry-point name,
or a dotted path. Resolution happens lazily inside the worker process, so the
parent never has to import ``jwst``.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import logging
import os
import sys
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, ClassVar

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "jwstflow.steps"

# Built-in aliases -> dotted path. Anything importable from jwst works; these
# are just the names people type most.
BUILTIN_ALIASES: dict[str, str] = {
    # pipelines
    "detector1": "jwst.pipeline:Detector1Pipeline",
    "image2": "jwst.pipeline:Image2Pipeline",
    "spec2": "jwst.pipeline:Spec2Pipeline",
    "image3": "jwst.pipeline:Image3Pipeline",
    "spec3": "jwst.pipeline:Spec3Pipeline",
    "tso3": "jwst.pipeline:Tso3Pipeline",
    "coron3": "jwst.pipeline:Coron3Pipeline",
    "dark": "jwst.pipeline:DarkPipeline",
    # a few individual steps (the full set is importable from jwst.step)
    "assign_wcs": "jwst.step:AssignWcsStep",
    "extract_1d": "jwst.step:Extract1dStep",
    "cube_build": "jwst.step:CubeBuildStep",
    "resample_spec": "jwst.step:ResampleSpecStep",
    "resample": "jwst.step:ResampleStep",
    "outlier_detection": "jwst.step:OutlierDetectionStep",
    "master_background": "jwst.step:MasterBackgroundStep",
    "photom": "jwst.step:PhotomStep",
    "jump": "jwst.step:JumpStep",
    "ramp_fit": "jwst.step:RampFitStep",
    "clean_flicker_noise": "jwst.step:CleanFlickerNoiseStep",
}

# Default asn_type for association files fed to these aliases.
# jwstflow's own QA steps; also exposed as `jwstflow.steps` entry points, this table
# is the fallback when a package was installed without them.
CONTRIB_ALIASES: dict[str, str] = {
    "plot_spectrum": "jwstflow.contrib.qa:PlotSpectrum",
    "quicklook_image": "jwstflow.contrib.qa:QuicklookImage",
    "header_summary": "jwstflow.contrib.qa:HeaderSummary",
    "fix_msa_metafile": "jwstflow.contrib.nirspec:fix_msa_metafile",
}

ALIAS_ASN_TYPE: dict[str, str] = {
    "image2": "image2",
    "spec2": "spec2",
    "image3": "image3",
    "spec3": "spec3",
    "tso3": "tso3",
    "coron3": "coron3",
}

_REGISTRY: dict[str, Any] = {}


def register_step(name: str, target: Any = None) -> Any:
    """Register a step under ``name`` (also usable as a decorator).

    In a plugin package you would normally use the ``jwstflow.steps`` entry
    point instead; this is handy inside notebooks/scripts.
    """

    def deco(obj: Any) -> Any:
        _REGISTRY[name] = obj
        return obj

    return deco(target) if target is not None else deco


# ---------------------------------------------------------------------------
# run context & result
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    """Everything a step may want to know about where it runs."""

    run_name: str
    root: Path
    stage: str
    output_dir: Path
    log_dir: Path
    raw_dir: Path
    stage_dirs: dict[str, Path] = field(default_factory=dict)
    crds_context: str | None = None
    dry_run: bool = False
    task_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def dir_of(self, stage: str) -> Path:
        """Output directory of another stage (e.g. to find companion files)."""
        return self.stage_dirs[stage]


@dataclass
class StepResult:
    outputs: list[Path]
    info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# step interface
# ---------------------------------------------------------------------------


class Step(ABC):
    """Base class for user-defined steps.

    Subclass, implement :meth:`run`, and reference the class in YAML::

        - name: qa
          step: my_pkg.qa:PlotSpectrum
          inputs: [{stage: spec3, pattern: "*_x1d.fits"}]
          parameters: {ylim: [0, 10]}
    """

    #: Default batching if the stage does not say otherwise ("per_file" | "all").
    batch: ClassVar[str] = "per_file"

    @abstractmethod
    def run(self, inputs: list[Path], ctx: RunContext, **params: Any) -> Iterable[Path] | None:
        """Process ``inputs`` and return the files written (used for checkpointing)."""


class FunctionStep(Step):
    """Wraps a plain callable ``f(inputs, ctx, **params)``."""

    def __init__(self, func: Callable[..., Any]):
        self.func = func

    def run(self, inputs: list[Path], ctx: RunContext, **params: Any) -> Iterable[Path] | None:
        return self.func(inputs, ctx, **params)


class JwstStepAdapter(Step):
    """Runs any ``stpipe.Step`` subclass through ``Step.call``.

    Notes on the call:

    * ``call`` (not ``run``) is used so CRDS parameter-reference files apply.
    * ``configure_log=False`` is passed when supported (jwst >= 1.20), because
      jwstflow configures logging itself (one file per task).
    * The step writes into a private ``<stage>/.tmp-<task>/`` directory which
      is moved into the stage directory on success. That makes output detection
      exact even when many tasks write to the same stage directory, and a
      crashed task leaves no half-written products behind.
    * The returned datamodel is closed immediately to keep memory flat.
    """

    def __init__(self, step_cls: type):
        self.step_cls = step_cls

    def run(self, inputs: list[Path], ctx: RunContext, **params: Any) -> Iterable[Path]:
        if len(inputs) != 1:
            raise ValueError(
                f"{self.step_cls.__name__} expects exactly one input (a file or an association); "
                f"got {len(inputs)}. Use `association:` or `batch: per_file`."
            )
        (inp,) = inputs
        out_dir = ctx.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = out_dir / f".tmp-{ctx.task_id or uuid.uuid4().hex[:8]}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()
        kwargs: dict[str, Any] = dict(params)
        kwargs["output_dir"] = str(tmp_dir)
        kwargs.setdefault("save_results", ctx.extra.get("save_results", True))

        t0 = time.time()
        try:
            result = self._call(str(inp), kwargs)
            _close(result)
            outputs: list[Path] = []
            for p in sorted(tmp_dir.iterdir()):
                dst = out_dir / p.name
                os.replace(p, dst)
                outputs.append(dst)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        tmp_dir.rmdir()
        log.info(
            "%s finished in %.1fs, %d file(s)", self.step_cls.__name__, time.time() - t0, len(outputs)
        )
        return outputs

    def _call(self, inp: str, kwargs: dict[str, Any]) -> Any:
        try:
            return self.step_cls.call(inp, configure_log=False, **kwargs)
        except Exception as exc:
            # Older stpipe (< jwst 1.20) rejects the unknown `configure_log` parameter
            # during config validation, i.e. before any processing happened.
            if "configure_log" in str(exc):
                log.debug("stpipe does not accept configure_log; retrying without it")
                return self.step_cls.call(inp, **kwargs)
            raise


def _close(result: Any) -> None:
    """Close whatever a pipeline returned (DataModel, ModelContainer, list, ...)."""
    if result is None:
        return
    items = result if isinstance(result, (list, tuple)) else [result]
    for item in items:
        close = getattr(item, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover
                pass


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def is_path_spec(spec: str) -> bool:
    """True for ``path/to/file.py:Object`` step specs."""
    head, sep, _ = spec.rpartition(":")
    return bool(sep) and head.endswith(".py")


def import_file(path: str | Path) -> Any:
    """Import a Python file as a module named after its stem (cached in sys.modules)."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ImportError(f"step file not found: {path}")
    name = path.stem
    existing = sys.modules.get(name)
    if existing is not None and Path(getattr(existing, "__file__", "") or "").resolve() == path:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ACTIVATED: set[str] = set()


def activate_plugins(paths: Iterable[str | Path]) -> list[str]:
    """Make user step code importable: directories go on sys.path, files by their parent.

    Idempotent; returns the sys.path entries that were (or already had been) added.
    Used by the runner in the parent process and by every worker.
    """
    added: list[str] = []
    for entry in paths:
        path = Path(entry).expanduser().resolve()
        directory = path if path.is_dir() else path.parent
        if not directory.is_dir():
            log.warning("plugin path does not exist: %s", path)
            continue
        d = str(directory)
        if d not in sys.path:
            sys.path.insert(0, d)
        _ACTIVATED.add(d)
        added.append(d)
    return added


def import_object(dotted: str) -> Any:
    """Import ``pkg.mod:attr``, ``pkg.mod.attr`` or ``path/to/file.py:attr``."""
    if is_path_spec(dotted):
        file, _, attr = dotted.rpartition(":")
        return getattr(import_file(file), attr)
    if ":" in dotted:
        mod_name, _, attr = dotted.partition(":")
        return getattr(importlib.import_module(mod_name), attr)
    parts = dotted.split(".")
    for i in range(len(parts) - 1, 0, -1):
        try:
            mod = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        obj: Any = mod
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"cannot import {dotted!r}")


def registered_steps() -> dict[str, str]:
    """All known step names -> description (built-ins, entry points, runtime registry)."""
    out = {k: v for k, v in BUILTIN_ALIASES.items()}
    out.update(CONTRIB_ALIASES)
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        out[ep.name] = ep.value
    for k, v in _REGISTRY.items():
        out[k] = getattr(v, "__qualname__", repr(v))
    return out


def resolve_target(spec: str) -> Any:
    """Turn a YAML ``step:`` string into a Python object (not yet a Step instance)."""
    if spec in _REGISTRY:
        return _REGISTRY[spec]
    if spec in BUILTIN_ALIASES:
        return import_object(BUILTIN_ALIASES[spec])
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == spec:
            return ep.load()
    if spec in CONTRIB_ALIASES:
        return import_object(CONTRIB_ALIASES[spec])
    return import_object(spec)


def source_fingerprint(spec: str) -> str | None:
    """Short hash of the source file that defines a user step, or None.

    Built-in jwst aliases return None (the jwst version already identifies
    them). For dotted paths, entry points and registered objects the module
    file is located with ``importlib.util.find_spec`` -- the module is *not*
    imported -- and hashed, so editing a custom step invalidates its tasks.
    """
    if spec in BUILTIN_ALIASES:
        return None
    if is_path_spec(spec):
        file = Path(spec.rpartition(":")[0]).expanduser()
        if not file.is_file():
            return None
        return hashlib.sha1(file.read_bytes()).hexdigest()[:12]
    if spec in _REGISTRY:
        mod_name = getattr(_REGISTRY[spec], "__module__", None)
    else:
        dotted = CONTRIB_ALIASES.get(spec, spec)
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            if ep.name == spec:
                dotted = ep.value
                break
        mod_name = dotted.partition(":")[0] if ":" in dotted else dotted.rpartition(".")[0]
    if not mod_name:
        return None
    try:
        found = importlib.util.find_spec(mod_name)
    except (ImportError, ValueError, AttributeError):
        return None
    origin = getattr(found, "origin", None)
    if not origin or not os.path.isfile(origin):
        return None
    with open(origin, "rb") as fh:
        return hashlib.sha1(fh.read()).hexdigest()[:12]


def is_stpipe_step(obj: Any) -> bool:
    """True if ``obj`` is a class deriving from stpipe.Step (checked by name so
    we never have to import stpipe in the parent process)."""
    if not inspect.isclass(obj):
        return False
    return any(
        base.__name__ == "Step" and base.__module__.split(".")[0] == "stpipe"
        for base in obj.__mro__
    )


def make_step(spec: str) -> Step:
    """Resolve ``spec`` and wrap it in the uniform :class:`Step` interface."""
    target = resolve_target(spec)
    if isinstance(target, Step):
        return target
    if is_stpipe_step(target):
        return JwstStepAdapter(target)
    if inspect.isclass(target) and issubclass(target, Step):
        return target()
    if callable(target):
        return FunctionStep(target)
    raise TypeError(f"{spec!r} resolved to {target!r}, which is not a step")


def describe_target(spec: str) -> dict[str, Any]:
    """Cheap description used by `jwstflow validate` (imports the object)."""
    target = resolve_target(spec)
    kind = (
        "stpipe"
        if is_stpipe_step(target)
        else "step"
        if inspect.isclass(target) and issubclass(target, Step)
        else "function"
        if callable(target)
        else "unknown"
    )
    return {
        "spec": spec,
        "kind": kind,
        "object": f"{getattr(target, '__module__', '?')}.{getattr(target, '__qualname__', '?')}",
    }


def default_thread_env(n: int) -> dict[str, str]:
    """Environment limiting BLAS/OpenMP threads (exported into workers)."""
    n = max(1, int(n))
    return {
        k: str(n)
        for k in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
        if k not in os.environ or os.environ.get("JWSTFLOW_FORCE_THREAD_ENV")
    }
