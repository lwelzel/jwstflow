"""The step registry: from a YAML ``step:`` string to a step object.

A step is addressed by a built-in alias of an official pipeline or step
(:data:`BUILTIN_ALIASES`), by a name registered in the ``jwstflow.steps`` entry-point
group (jwstflow registers its own steps there like any step package: pyproject.toml is
the one list), by a name registered at run time (:func:`register_step`), or by a dotted
path / file path. Resolution happens
lazily inside the worker process, so the parent never has to import ``jwst``.

The module also answers what the engine needs to know about a step *before* running
it: its identity (:func:`step_identity` -- stage name and level) and the file that
defines it (:func:`source_origin`, :func:`source_fingerprint` -- part of every task id,
so editing a custom step reruns its tasks).
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import logging
import os
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from .adapters import (
    STPIPE_LEVELS,
    JwstStepAdapter,
    _snake,
    identity_of,
    is_stpipe_step,
)
from .interface import FunctionStep, Step
from .plugins import import_object, is_path_spec

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


def _registered_target(name: str) -> str | None:
    """Dotted target behind a step name registered by an installed package (jwstflow itself
    included: its own steps are entry points like any other), or None."""
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == name:
            return ep.value
    return None


# Default asn_type for association files fed to these aliases.
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
    # steps registered by name (entry points, register_step) are known by that name to users
    # and in `jwstflow steps`; keep it unless the class declares its own `name`
    registered = spec in _REGISTRY or _registered_target(spec) is not None
    if registered and not is_stpipe_step(target) and getattr(target, "name", None) in (None, ""):
        name = spec
    return name, level


def resolve_target(spec: str) -> Any:
    """Turn a YAML ``step:`` string into a Python object (not yet a Step instance)."""
    if spec in _REGISTRY:
        return _REGISTRY[spec]
    if spec in BUILTIN_ALIASES:
        return import_object(BUILTIN_ALIASES[spec])
    matches = [ep for ep in entry_points(group=ENTRY_POINT_GROUP) if ep.name == spec]
    if matches:
        if len({ep.value for ep in matches}) > 1:
            log.warning("step name %r is registered by several installed packages (%s); using %s -- "
                        "uninstall one, or address the step by dotted path",
                        spec, ", ".join(sorted({ep.value for ep in matches})), matches[0].value)
        return matches[0].load()
    return import_object(spec)


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


def registered_steps() -> dict[str, str]:
    """All known step names -> target (built-ins, entry points, runtime registry)."""
    out = {k: v for k, v in BUILTIN_ALIASES.items()}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        out[ep.name] = ep.value
    for k, v in _REGISTRY.items():
        out[k] = getattr(v, "__qualname__", repr(v))
    return out


def step_package_versions() -> dict[str, str]:
    """Installed distributions providing `jwstflow.steps` entry points -> version.

    Recorded in every run manifest so a reduction's provenance names the
    contributed-package versions that produced it, next to jwst and CRDS.
    """
    from importlib.metadata import distributions

    out: dict[str, str] = {}
    for dist in distributions():
        try:
            if any(ep.group == ENTRY_POINT_GROUP for ep in dist.entry_points):
                out[str(dist.metadata["Name"])] = str(dist.version)
        except Exception:  # a broken .dist-info must never take the run down
            continue
    return dict(sorted(out.items()))


def source_origin(spec: str) -> Path | None:
    """Path of the source file that defines a user step, or None.

    Built-in jwst aliases return None (the jwst version already identifies
    them). For dotted paths, entry points and registered objects the module
    file is located with ``importlib.util.find_spec`` -- the module is *not*
    imported.
    """
    if spec in BUILTIN_ALIASES:
        return None
    if is_path_spec(spec):
        file = Path(spec.rpartition(":")[0]).expanduser()
        return file if file.is_file() else None
    if spec in _REGISTRY:
        mod_name = getattr(_REGISTRY[spec], "__module__", None)
    else:
        dotted = _registered_target(spec) or spec
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
    return Path(origin)


def source_fingerprint(spec: str) -> str | None:
    """Short hash of the file that defines a user step (None for jwst built-ins),
    so editing a custom step invalidates its tasks."""
    origin = source_origin(spec)
    if origin is None:
        return None
    return hashlib.sha1(origin.read_bytes()).hexdigest()[:12]
