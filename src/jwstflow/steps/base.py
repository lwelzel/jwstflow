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
import re
import sys
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path

from pydantic import BaseModel, ConfigDict
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
# Calibration level of the official pipelines and steps, keyed by stpipe class_alias.
# 1: ramps -> rate, 2: exposure calibration, 3: combined products; unknown single steps -> 3.
STPIPE_LEVELS: dict[str, int] = {
    "calwebb_detector1": 1, "calwebb_dark": 1, "calwebb_guider": 1,
    **{a: 1 for a in ("group_scale", "dq_init", "emicorr", "saturation", "ipc", "superbias", "refpix", "rscd",
                      "firstframe", "lastframe", "linearity", "dark_current", "reset", "persistence",
                      "charge_migration", "jump", "clean_flicker_noise", "ramp_fit", "gain_scale", "undersampling_correction")},
    "calwebb_spec2": 2, "calwebb_image2": 2, "calwebb_wfs-image2": 2,
    **{a: 2 for a in ("assign_wcs", "badpix_selfcal", "msa_flagging", "bkg_subtract", "imprint_subtract", "extract_2d",
                      "master_background_mos", "targ_centroid", "wavecorr", "flat_field", "srctype", "straylight", "fringe",
                      "residual_fringe", "pathloss", "barshadow", "wfss_contam", "photom", "picture_frame", "resample_spec",
                      "nsclean")},
    "calwebb_spec3": 3, "calwebb_image3": 3, "calwebb_tso3": 3, "calwebb_coron3": 3, "calwebb_ami3": 3, "calwebb_wfs-image3": 3,
    **{a: 3 for a in ("assign_mtwcs", "master_background", "outlier_detection", "cube_build", "pixel_replace", "extract_1d",
                      "combine_1d", "spectral_leak", "adaptive_trace_model", "resample", "tweakreg", "skymatch",
                      "source_catalog", "klip", "stack_refs", "align_refs", "white_light", "tso_photometry")},
}

LEVEL_DIRS: dict[int | str, str] = {1: "stage1", 2: "stage2", 3: "stage3", 4: "stage4", "qa": "qa"}


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def identity_of(target: Any, spec: str = "") -> tuple[str, int | str]:
    """Canonical ``(name, level)`` of a resolved step object.

    stpipe classes: their ``class_alias`` (``calwebb_spec3``, ``extract_1d``; a
    subclass inherits its parent's alias unless it sets its own) and the level
    of that alias. jwstflow steps: the class/function ``name`` attribute or the
    snake-cased class name, and their ``level`` attribute (default 4, "qa" for
    plot-type steps).
    """
    if is_stpipe_step(target):
        alias = getattr(target, "class_alias", None) or _snake(target.__name__)
        return str(alias), STPIPE_LEVELS.get(str(alias), 3)
    obj = target if isinstance(target, type) or callable(target) else type(target)
    name = getattr(obj, "name", None) or getattr(obj, "__name__", None) or spec.rpartition(":")[2] or spec
    level = getattr(obj, "level", 4)
    if isinstance(name, str) and name and name[0].isupper():
        name = _snake(name)
    return str(name), level


def step_identity(spec: str) -> tuple[str, int | str]:
    """``(canonical name, level)`` for a step spec without running it.

    Built-in aliases are answered from a table; everything else is imported.
    If the import fails the name is derived from the spec (validation reports
    the import error later) and the level defaults to 4.
    """
    if spec in BUILTIN_ALIASES:
        alias = BUILTIN_ALIASES[spec].rpartition(":")[2]
        canonical = {"Detector1Pipeline": "calwebb_detector1", "Image2Pipeline": "calwebb_image2", "Spec2Pipeline": "calwebb_spec2",
                     "Image3Pipeline": "calwebb_image3", "Spec3Pipeline": "calwebb_spec3", "Tso3Pipeline": "calwebb_tso3",
                     "Coron3Pipeline": "calwebb_coron3", "DarkPipeline": "calwebb_dark"}.get(alias)
        if canonical is None:
            canonical = spec  # single-step aliases are the jwst class_alias already (extract_1d, cube_build, ...)
        return canonical, STPIPE_LEVELS.get(canonical, 3)
    try:
        target = resolve_target(spec)
    except Exception as exc:
        log.debug("cannot import %s for its identity: %s", spec, exc)
        tail = spec.rpartition(":")[2] or spec.rpartition(".")[2]
        return _snake(tail), 4
    name, level = identity_of(target, spec)
    # steps registered by name (entry points, contrib table, register_step) are known by that
    # name to users and in `jwstflow steps`; keep it unless the class declares its own `name`
    registered = spec in _REGISTRY or spec in CONTRIB_ALIASES or any(ep.name == spec for ep in entry_points(group=ENTRY_POINT_GROUP))
    if registered and not is_stpipe_step(target) and getattr(target, "name", None) in (None, ""):
        name = spec
    return name, level


# jwstflow's own QA steps; also exposed as `jwstflow.steps` entry points, this table
# is the fallback when a package was installed without them.
CONTRIB_ALIASES: dict[str, str] = {
    "plot_spectrum": "jwstflow.contrib.qa:PlotSpectrum",
    "quicklook_image": "jwstflow.contrib.qa:QuicklookImage",
    "header_summary": "jwstflow.contrib.qa:HeaderSummary",
    "fix_msa_metafile": "jwstflow.contrib.nirspec:fix_msa_metafile",
    "mast_compare": "jwstflow.contrib.mast_compare:MastCompare",
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
    #: target label of the run, explicit coordinates if the workflow gave any, and the
    #: target directory (shared by all runs of the target; holds the name-resolution cache)
    target: str = ""
    target_coords: dict[str, Any] | None = None
    target_dir: Path | None = None
    #: <target>/mast_reference/<run>: MAST's own products mirroring this run's layout (opt-in download)
    reference_dir: Path | None = None

    def dir_of(self, stage: str) -> Path:
        """Output directory of another stage (e.g. to find companion files)."""
        return self.stage_dirs[stage]

    def derived_path(self, source: str | Path, suffix: str, *, descriptor: str | None = None, ext: str = ".fits") -> Path:
        """Output path for a product derived from ``source``, following jwstflow's naming rules
        (official suffix stripped, descriptor inserted, custom suffix appended) inside ``output_dir``."""
        from ..naming import derived_name

        return self.output_dir / derived_name(source, suffix, descriptor=descriptor, ext=ext)

    @property
    def log(self) -> logging.Logger:
        """Logger named after the stage; its output lands in the task log."""
        return logging.getLogger(f"jwstflow.step.{self.stage}")


@dataclass
class StepResult:
    outputs: list[Path]
    info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# step interface
# ---------------------------------------------------------------------------


class StepParams(BaseModel):
    """Base class for a step's ``Params`` model: typed, documented, unknown keys rejected."""

    model_config = ConfigDict(extra="forbid", validate_default=True)


class Step(ABC):
    """Base class for user-defined steps: the contract between a step and jwstflow.

    A step is a class with one method, :meth:`run`, plus a small *declaration*
    that lets jwstflow validate, document and test it without running it::

        class ExtractExtended(Step):
            \"\"\"One-line summary (shown by `jwstflow steps --describe`).\"\"\"

            level = 4                       # 1/2/3 jwst stages, 4 derived products, "qa" plots
            batch = "per_file"              # or "all": one task with every input
            inputs = ("*_s3d.fits",)        # what it accepts (glob patterns)
            outputs = ("s1d",)              # product suffixes it writes (never a jwst one)
            version = "1"                   # bump when results change -> tasks rerun

            class Params(StepParams):       # the `parameters:` block, validated at config time
                threshold: float = Field(0.05, gt=0, description="aperture threshold")

            def run(self, inputs, ctx, *, threshold: float = 0.05, **params):
                (cube,) = inputs
                out = ctx.derived_path(cube, "s1d", descriptor="extended")
                ...
                return [out]

    Rules jwstflow enforces (at import, config load, or after `run`):

    * ``name``/``level``/``batch`` are valid; ``outputs`` never use a reserved
      jwst suffix (``_cal``, ``_s3d``, ``_x1d``, ...);
    * ``parameters:`` in the YAML match ``Params`` (unknown keys are errors);
    * every path returned by ``run`` exists, lies in ``ctx.output_dir`` and,
      for FITS files, carries a non-reserved suffix; files written but not
      returned are reported as orphans.

    ``jwstflow check-step my_module:MyStep`` runs these checks, and
    ``jwstflow.testing.run_step`` executes a step on synthetic data in a test.
    """

    #: Default batching if the stage does not say otherwise ("per_file" | "all").
    batch: ClassVar[str] = "per_file"
    #: Canonical stage name (default: snake_case of the class name). Fixed by the step,
    #: not by the workflow, so runs stay comparable across users; `variant:` disambiguates.
    name: ClassVar[str | None] = None
    #: Calibration level of the products: 1, 2, 3 (jwst stages), 4 (derived products) or "qa".
    level: ClassVar[int | str] = 4
    #: Glob patterns of the files the step accepts (documentation + a planning-time check).
    inputs: ClassVar[tuple[str, ...]] = ()
    #: Product suffixes the step writes (``"s1d"`` for ``..._s1d.fits``); checked against
    #: the reserved jwst suffixes when the class is defined.
    outputs: ClassVar[tuple[str, ...]] = ()
    #: Bump when the algorithm changes in a way that should rerun existing tasks.
    version: ClassVar[str] = "1"
    #: Only for steps that deliberately write *edited copies of official products* under their
    #: official names (a DQ-flagged ``_cal``, a WCS-shifted ``_rate``); everything else must use
    #: its own suffix. Record the edit in the header when you set this.
    writes_official_products: ClassVar[bool] = False
    #: Optional pydantic model describing ``parameters:`` (subclass of :class:`StepParams`).
    Params: ClassVar[type[Any] | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for problem in check_declaration(cls):
            raise TypeError(f"{cls.__module__}.{cls.__qualname__}: {problem}")

    @abstractmethod
    def run(self, inputs: list[Path], ctx: RunContext, **params: Any) -> Iterable[Path] | None:
        """Process ``inputs`` and return every file written (used for checkpointing)."""

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        """Validate a ``parameters:`` block against ``Params`` (returns it unchanged if there is none)."""
        if cls.Params is None:
            return dict(params)
        try:
            return cls.Params.model_validate(params).model_dump(exclude_unset=True)
        except Exception as exc:  # pydantic.ValidationError
            raise ValueError(f"parameters of {cls.__name__} are invalid: {exc}") from exc

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """Machine-readable description of the declaration (for the CLI and docs)."""
        params: list[dict[str, Any]] = []
        if cls.Params is not None:
            for fname, field in cls.Params.model_fields.items():
                params.append({"name": fname, "type": _type_name(field.annotation), "default": field.default
                               if field.default is not _PydanticUndefined() else "(required)",
                               "description": field.description or ""})
        name, level = identity_of(cls)
        return {"name": name, "level": level, "batch": cls.batch, "inputs": list(cls.inputs), "outputs": list(cls.outputs),
                "version": cls.version, "doc": (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else "", "params": params}


def _PydanticUndefined() -> Any:
    from pydantic_core import PydanticUndefined

    return PydanticUndefined


def _type_name(annotation: Any) -> str:
    return getattr(annotation, "__name__", None) or str(annotation).replace("typing.", "")


def check_declaration(cls: type) -> list[str]:
    """Problems with a step class's declaration (empty list when it is fine)."""
    problems: list[str] = []
    if cls.batch not in ("per_file", "all"):
        problems.append(f"batch must be 'per_file' or 'all', not {cls.batch!r}")
    if cls.level not in (1, 2, 3, 4, "qa"):
        problems.append(f"level must be 1, 2, 3, 4 or 'qa', not {cls.level!r}")
    if cls.name is not None and not re.fullmatch(r"[a-z0-9][a-z0-9_]*", str(cls.name)):
        problems.append(f"name {cls.name!r} must be a lowercase slug")
    if not isinstance(cls.inputs, (tuple, list)) or not all(isinstance(p, str) for p in cls.inputs):
        problems.append("inputs must be a tuple of glob patterns")
    if not isinstance(cls.outputs, (tuple, list)) or not all(isinstance(p, str) for p in cls.outputs):
        problems.append("outputs must be a tuple of suffix strings")
    else:
        from ..naming import JWST_PRODUCT_SUFFIXES

        for suffix in cls.outputs:
            if suffix in JWST_PRODUCT_SUFFIXES:
                problems.append(f"output suffix {suffix!r} is reserved for jwst pipeline products")
            elif not re.fullmatch(r"[a-z0-9]+", suffix):
                problems.append(f"output suffix {suffix!r} must be lowercase alphanumeric")
    if cls.Params is not None:
        model_fields = getattr(cls.Params, "model_fields", None)
        if model_fields is None:
            problems.append("Params must be a pydantic model (subclass jwstflow.StepParams)")
        elif getattr(cls.Params, "model_config", {}).get("extra") != "forbid":
            problems.append("Params must forbid unknown keys (subclass jwstflow.StepParams)")
    return problems


class FunctionStep(Step):
    """Wraps a plain callable ``f(inputs, ctx, **params)``.

    A function may carry the same declaration as a class through attributes
    (``f.level = 2``, ``f.outputs = ("s1d",)``, ``f.writes_official_products = True``, ...).
    """

    DECLARATION = ("batch", "level", "name", "inputs", "outputs", "version", "writes_official_products", "Params")

    def __init__(self, func: Callable[..., Any]):
        self.func = func
        for attr in self.DECLARATION:
            if hasattr(func, attr):
                setattr(self, attr, getattr(func, attr))

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
        if path.is_file() and path.suffix == ".py":
            # `steps.py` next to every workflow is a common name: make sure the module
            # cached under that stem is the one from *this* file, not an earlier plugin's
            cached = sys.modules.get(path.stem)
            cached_file = Path(getattr(cached, "__file__", "") or "")
            if cached is not None and cached_file.resolve() != path:
                del sys.modules[path.stem]
        _ACTIVATED.add(d)
        added.append(d)
    return added


def check_outputs(outputs: Iterable[Path] | None, ctx: RunContext, step: Any) -> list[Path]:
    """Verify what a step returned: paths exist, live in the output directory and (for custom
    steps) use no reserved jwst suffix. Raises ValueError on a violation."""
    from ..naming import JWST_PRODUCT_SUFFIXES, split_suffix

    paths = [Path(o) for o in (outputs or [])]
    out_dir = ctx.output_dir.resolve()
    declared = tuple(getattr(step, "outputs", ()) or ())

    def violation(message: str) -> ValueError:
        # the task fails: take its files with it, so downstream stages never consume
        # the products of a failed task and a fixed step reruns from a clean directory
        removed = 0
        for q in paths:
            if q.exists() and out_dir in q.resolve().parents:
                q.unlink()
                removed += 1
        if removed:
            message += f" (the task's {removed} output file(s) were removed)"
        return ValueError(message)

    for p in paths:
        if not p.exists():
            raise violation(f"step returned a file that does not exist: {p}")
        if out_dir not in p.resolve().parents and p.resolve() != out_dir:
            raise violation(f"step returned a file outside its output directory {out_dir}: {p}")
        if p.suffix.lower() == ".fits" and not isinstance(step, JwstStepAdapter):
            _, suffix = split_suffix(p.stem)
            if suffix in JWST_PRODUCT_SUFFIXES and not getattr(step, "writes_official_products", False):
                raise violation(
                    f"custom step wrote an official jwst product name ({p.name}); use a jwstflow "
                    "suffix via ctx.derived_path(), or set `writes_official_products = True` on the step "
                    "if it deliberately produces edited copies of official products"
                )
            if declared and suffix is not None and suffix not in declared:
                log.warning("%s: output %s has suffix %r, not among the declared outputs %s", type(step).__name__, p.name, suffix, declared)
    return paths


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
