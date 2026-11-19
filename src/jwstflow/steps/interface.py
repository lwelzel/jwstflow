"""The step interface: what a jwstflow step is, and what it may count on.

* :class:`Step` -- the base class of user-defined steps: one method, ``run``, plus a
  small *declaration* (level, batch, inputs, outputs, version, ``Params``) that lets
  jwstflow validate, document and test a step without running it;
* :class:`StepParams` -- base of a step's typed ``Params`` model;
* :class:`FunctionStep` -- a plain function ``f(inputs, ctx, **params)`` behind the same interface;
* :class:`RunContext` -- everything a step may want to know about where it runs;
* :func:`check_declaration` / :func:`check_outputs` -- the contract checks, at class
  definition and after every ``run``.

The official pipelines enter through :mod:`jwstflow.steps.adapters`; names are turned
into step objects by :mod:`jwstflow.steps.registry`.
"""

from __future__ import annotations

import inspect
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

log = logging.getLogger(__name__)

#: Directory of each calibration level inside a run (``stage1/`` ... ``stage4/``, ``qa/``).
LEVEL_DIRS: dict[int | str, str] = {1: "stage1", 2: "stage2", 3: "stage3", 4: "stage4", "qa": "qa"}


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
        from .naming import derived_name

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
            \"\"\"One-line summary (the `jwstflow steps` table).

            Detailed explanation of what run() computes, shown by
            `jwstflow steps <name>` / `--describe`.
            \"\"\"

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
    #: Only for steps that read MAST's own products of the run (``ctx.reference_dir``, fetched in the
    #: background when ``download.reference_products`` is on): the run waits for that download to
    #: finish before the stage starts.
    needs_reference_products: ClassVar[bool] = False
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
        from .adapters import identity_of  # the adapters build on this module

        name, level = identity_of(cls)
        short, detailed = split_description(inspect.getdoc(cls))
        return {"name": name, "level": level, "batch": cls.batch, "inputs": list(cls.inputs), "outputs": list(cls.outputs),
                "version": cls.version, "doc": short, "description": detailed, "params": params}



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
        from .naming import JWST_PRODUCT_SUFFIXES

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

    DECLARATION = ("batch", "level", "name", "inputs", "outputs", "version", "writes_official_products",
                   "needs_reference_products", "Params")

    def __init__(self, func: Callable[..., Any]):
        self.func = func
        for attr in self.DECLARATION:
            if hasattr(func, attr):
                setattr(self, attr, getattr(func, attr))

    def run(self, inputs: list[Path], ctx: RunContext, **params: Any) -> Iterable[Path] | None:
        return self.func(inputs, ctx, **params)


def check_outputs(outputs: Iterable[Path] | None, ctx: RunContext, step: Any) -> list[Path]:
    """Verify what a step returned: paths exist, live in the output directory and (for custom
    steps) use no reserved jwst suffix. Raises ValueError on a violation."""
    from .adapters import JwstStepAdapter
    from .naming import JWST_PRODUCT_SUFFIXES, split_suffix

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


def split_description(doc: str | None) -> tuple[str, str]:
    """``(short, detailed)`` from a docstring: the first paragraph collapsed to
    one line, and everything after it."""
    if not doc:
        return "", ""
    text = inspect.cleandoc(doc)
    first, _, rest = text.partition("\n\n")
    return " ".join(first.split()), rest.strip()
