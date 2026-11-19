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

This module is the one import path for all of it (``from jwstflow.steps.base
import Step, RunContext, ...``); the code lives next to it, by concern:

* :mod:`.interface` -- :class:`Step`, :class:`StepParams`, :class:`FunctionStep`,
  :class:`RunContext` and the contract checks;
* :mod:`.adapters` -- :class:`JwstStepAdapter` and the identity (stage name,
  level) of both kinds of step;
* :mod:`.registry` -- names to step objects: aliases, entry points,
  :func:`register_step`, :func:`resolve_target`, :func:`make_step`, and the
  source fingerprint that enters every task id;
* :mod:`.plugins` -- importing user code by path, the ``plugins:`` of a workflow;
* :mod:`.descriptions` -- the texts ``jwstflow steps`` prints.
"""

from __future__ import annotations

from .adapters import STPIPE_LEVELS, JwstStepAdapter, identity_of, is_stpipe_step
from .descriptions import BUILTIN_DESCRIPTIONS, step_description
from .interface import (
    LEVEL_DIRS,
    FunctionStep,
    RunContext,
    Step,
    StepParams,
    StepResult,
    check_declaration,
    check_outputs,
    split_description,
)
from .plugins import (
    activate_plugins,
    import_file,
    import_object,
    is_path_spec,
    plugins_fingerprint,
)
from .registry import (
    _REGISTRY,  # noqa: F401  (the in-process registry: tests and the mock toolkit shadow names in it)
    ALIAS_ASN_TYPE,
    BUILTIN_ALIASES,
    BUNDLED_STEPS,
    ENTRY_POINT_GROUP,
    MOVED_STEPS,
    describe_target,
    make_step,
    register_step,
    registered_steps,
    resolve_target,
    source_fingerprint,
    source_origin,
    step_identity,
    step_package_versions,
)

__all__ = [
    "ALIAS_ASN_TYPE",
    "BUILTIN_ALIASES",
    "BUILTIN_DESCRIPTIONS",
    "BUNDLED_STEPS",
    "ENTRY_POINT_GROUP",
    "LEVEL_DIRS",
    "MOVED_STEPS",
    "STPIPE_LEVELS",
    "FunctionStep",
    "JwstStepAdapter",
    "RunContext",
    "Step",
    "StepParams",
    "StepResult",
    "activate_plugins",
    "check_declaration",
    "check_outputs",
    "describe_target",
    "identity_of",
    "import_file",
    "import_object",
    "is_path_spec",
    "is_stpipe_step",
    "make_step",
    "plugins_fingerprint",
    "register_step",
    "registered_steps",
    "resolve_target",
    "source_fingerprint",
    "source_origin",
    "split_description",
    "step_description",
    "step_identity",
    "step_package_versions",
]
