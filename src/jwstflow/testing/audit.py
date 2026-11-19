"""Audit a step against the step contract without running it (``jwstflow check-step``)."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from ..steps import (
    FunctionStep,
    Step,
    check_declaration,
    is_stpipe_step,
    resolve_target,
    source_origin,
)


def check_step(spec: str | type | Any) -> tuple[dict[str, Any], list[str]]:
    """Audit a step without running it: ``(description, problems)``.

    Checks: the spec imports; it is a jwstflow Step, an stpipe Step, or a
    callable; a Step's declaration (name, level, batch, inputs, outputs, Params)
    is valid; ``run`` has the ``(inputs, ctx, **params)`` shape; keyword
    parameters of ``run`` are consistent with ``Params`` when both exist. For
    a step addressed by name or dotted path, also that the address leads to
    the module that *defines* the step, not to one that re-exports it.
    """
    problems: list[str] = []
    try:
        target = resolve_target(spec) if isinstance(spec, str) else spec
    except Exception as exc:
        return {"spec": str(spec)}, [f"cannot import {spec!r}: {exc}"]
    if isinstance(spec, str):
        problems += _check_address(spec, target)
    if is_stpipe_step(target):
        return {"spec": str(spec), "kind": "stpipe", "object": f"{target.__module__}.{target.__name__}",
                "class_alias": getattr(target, "class_alias", None)}, problems
    if isinstance(target, Step):
        cls: Any = type(target)
    elif inspect.isclass(target) and issubclass(target, Step):
        cls = target
    elif callable(target):
        return _check_function(target, spec, problems)
    else:
        return {"spec": str(spec)}, [f"{spec!r} is neither a jwstflow Step, an stpipe Step nor a function"]
    problems += check_declaration(cls)
    sig = inspect.signature(cls.run)
    names = [p.name for p in sig.parameters.values()]
    if names[:3] != ["self", "inputs", "ctx"]:
        problems.append(f"run() must start with (self, inputs, ctx), got ({', '.join(names)})")
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        problems.append("run() should accept **params so that extra parameters never break it")
    keyword_only = [p.name for p in sig.parameters.values() if p.kind is inspect.Parameter.KEYWORD_ONLY]
    if cls.Params is not None:
        declared = set(cls.Params.model_fields)
        missing = [k for k in keyword_only if k not in declared]
        if missing:
            problems.append(f"run() keyword parameters {missing} are not declared in Params")
    if not (cls.__doc__ or "").strip():
        problems.append("add a docstring: first paragraph = short description, rest = detailed (see `jwstflow steps`)")
    desc = cls.describe()
    desc["spec"] = str(spec)
    desc["kind"] = "step"
    return desc, problems


def _check_address(spec: str, target: Any) -> list[str]:
    """A name or dotted path must lead to the file that defines the step: that file is what
    jwstflow hashes into the task ids (so an edit reruns the step) and reads the description
    from. An entry point naming a package that merely re-exports the class defeats both."""
    obj = target if inspect.isclass(target) or inspect.isfunction(target) else type(target)
    try:
        defined = inspect.getsourcefile(obj)
    except (TypeError, OSError):  # builtins, classes made in a session: nothing to compare
        return []
    origin = source_origin(spec)
    if origin is None or defined is None or Path(defined).resolve() == Path(origin).resolve():
        return []
    return [f"{spec!r} is addressed through {origin}, but {obj.__name__} is defined in {defined}: name the "
            f"defining module (its file is hashed into the task ids and read for the description)"]


def _check_function(func: Any, spec: Any, problems: list[str] | None = None) -> tuple[dict[str, Any], list[str]]:
    problems = list(problems or [])
    names = [p.name for p in inspect.signature(func).parameters.values()]
    if names[:2] != ["inputs", "ctx"]:
        problems.append(f"a step function must be f(inputs, ctx, **params), got ({', '.join(names)})")
    wrapped = FunctionStep(func)
    problems += check_declaration(type("_Decl", (Step,), {"run": lambda self, i, c, **p: None,
                                                                 **{a: getattr(wrapped, a) for a in FunctionStep.DECLARATION if hasattr(wrapped, a)}}))
    return {"spec": str(spec), "kind": "function", "object": f"{func.__module__}.{func.__qualname__}",
            "level": wrapped.level, "batch": wrapped.batch, "outputs": list(wrapped.outputs)}, problems
