"""Load YAML into a validated :class:`Config`.

Deliberately small. It implements the four composition features people
actually use from Hydra/OmegaConf, without adopting an application framework:

1. ``extends``: inherit from another YAML file (relative path) or a bundled
   preset (``preset:nirspec_ifu``). Chains are allowed. Dicts merge deeply,
   lists are replaced, *except* ``stages`` which merge by stage ``name`` so a
   user file can tweak one parameter of a preset stage without re-listing it.
2. ``${...}`` interpolation: ``${root}``, ``${name}``, any dotted key
   (``${crds.path}``), ``${env:VAR}`` and ``${env:VAR,default}``.
3. dot-list overrides from the CLI: ``--set parallel.workers=8`` or
   ``--set stages.detector1.parameters.steps.jump.rejection_threshold=5``
   (stages can be addressed by name or by index). Values are parsed as YAML.
4. Path resolution relative to the YAML file's directory for ``root``.

If you prefer OmegaConf/Hydra for composition, do that first and hand the
resolved dict to :func:`config_from_dict`; nothing here is mandatory.
"""

from __future__ import annotations

import copy
import logging
import os
import re
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ..steps.base import activate_plugins, step_identity
from .schema import RAW_STAGE, Config, slugify

EXTENDS_KEY = "extends"
PRESET_PREFIX = "preset:"
_INTERP = re.compile(r"\$\{([^${}]+)\}")


log = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    """Raised for composition/validation problems, with a user-facing message."""


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------


def preset_path(name: str) -> Path:
    """Path of a bundled preset YAML (``jwstflow init`` lists them)."""
    pkg = resources.files("jwstflow.config") / "presets" / f"{name}.yaml"
    p = Path(str(pkg))
    if not p.exists():
        raise ConfigError(f"unknown preset {name!r}; available: {', '.join(list_presets())}")
    return p


def list_presets() -> list[str]:
    d = Path(str(resources.files("jwstflow.config") / "presets"))
    return sorted(p.stem for p in d.glob("*.yaml"))


# ---------------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------------


def deep_merge(base: Any, override: Any) -> Any:
    """Merge ``override`` into ``base`` (copy). Dicts merge, lists replace,
    ``stages`` lists merge by name."""
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            if k == "stages" and isinstance(out.get(k), list) and isinstance(v, list):
                out[k] = _merge_stages(out[k], v)
            elif k in out:
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
        return out
    return copy.deepcopy(override)


def stage_key(stage: Any) -> str | None:
    """Identity used to merge stages across `extends`: canonical step name + variant.

    A preset's `step: spec3` and a user's `step: my_steps:Spec3WithBackground`
    (a subclass keeping class_alias calwebb_spec3) are the same stage.
    """
    if not isinstance(stage, dict) or not isinstance(stage.get("step"), str):
        return None
    name, _level = step_identity(stage["step"])
    return f"{name}-{stage['variant']}" if stage.get("variant") else name


def _merge_stages(base: list[Any], override: list[Any]) -> list[Any]:
    merged: list[Any] = [copy.deepcopy(s) for s in base]
    index = {stage_key(s): i for i, s in enumerate(merged)}
    for st in override:
        key = stage_key(st)
        if key is not None and key in index:
            merged[index[key]] = deep_merge(merged[index[key]], st)
        else:
            merged.append(copy.deepcopy(st))
            index[key] = len(merged) - 1
    return merged


# ---------------------------------------------------------------------------
# raw loading with `extends`
# ---------------------------------------------------------------------------


def load_raw(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML file and resolve its ``extends`` chain into one dict."""
    path = Path(path).expanduser().resolve()
    if path in _seen:
        raise ConfigError(f"circular `extends` chain: {' -> '.join(map(str, _seen + (path,)))}")
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if isinstance(data, dict) and data.get("plugins"):
        # stage identities (used to merge stages across `extends`) may live in the user's code
        activate_plugins([_absolute(p, path.parent) for p in _as_list(data["plugins"])])
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parent_ref = data.pop(EXTENDS_KEY, None)
    if parent_ref is None:
        return data
    parent_path = (
        preset_path(parent_ref[len(PRESET_PREFIX) :])
        if str(parent_ref).startswith(PRESET_PREFIX)
        else (path.parent / str(parent_ref))
    )
    parent = load_raw(parent_path, _seen + (path,))
    return deep_merge(parent, data)


# ---------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``key.path=value`` overrides (values parsed as YAML)."""
    data = copy.deepcopy(data)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must look like key.path=value")
        key, _, raw = item.partition("=")
        try:
            value = yaml.safe_load(raw) if raw != "" else None
        except yaml.YAMLError as exc:
            raise ConfigError(f"cannot parse value of override {item!r}: {exc}") from exc
        _set_path(data, key.strip().split("."), value)
    return data


def _set_path(node: Any, parts: list[str], value: Any) -> None:
    key, rest = parts[0], parts[1:]
    if isinstance(node, list):
        idx = _list_index(node, key)
        if not rest:
            node[idx] = value
        else:
            if node[idx] is None:
                node[idx] = {}
            _set_path(node[idx], rest, value)
        return
    if not isinstance(node, dict):
        raise ConfigError(f"cannot descend into non-mapping at {key!r}")
    if not rest:
        node[key] = value
        return
    if key not in node or node[key] is None:
        node[key] = [] if rest and rest[0].isdigit() else {}
    _set_path(node[key], rest, value)


def _list_index(node: list[Any], key: str) -> int:
    if key.isdigit():
        idx = int(key)
        if idx >= len(node):
            raise ConfigError(f"list index {idx} out of range (len {len(node)})")
        return idx
    for i, item in enumerate(node):
        if not isinstance(item, dict):
            continue
        # stages: canonical name (calwebb_spec3, calwebb_spec3-pass1) or the step string itself (spec3)
        if item.get("name") == key or (item.get("step") == key and not item.get("variant")) or stage_key(item) == key:
            return i
    raise ConfigError(f"no list element named {key!r} (stages are addressed by their derived name or step)")


# ---------------------------------------------------------------------------
# interpolation
# ---------------------------------------------------------------------------


def interpolate(data: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve ``${...}`` references everywhere in ``data``.

    ``extra`` adds read-only names (``run_dir``, ``target_dir``, ``raw_dir``,
    ``workspace``, ``project_root``) that are not part of the document.
    """
    extra = extra or {}

    def lookup(ref: str, stack: tuple[str, ...]) -> Any:
        if ref in extra:
            return extra[ref]
        if ref.startswith("env:"):
            name, _, default = ref[4:].partition(",")
            if name in os.environ:
                return os.environ[name]
            if default != "" or "," in ref:
                return yaml.safe_load(default) if default else None
            raise ConfigError(f"environment variable {name!r} is not set (used as ${{{ref}}})")
        if ref in stack:
            raise ConfigError(f"circular interpolation: {' -> '.join(stack + (ref,))}")
        node: Any = data
        for part in ref.split("."):
            if isinstance(node, list):
                node = node[_list_index(node, part)]
            elif isinstance(node, dict) and part in node:
                node = node[part]
            else:
                raise ConfigError(f"cannot resolve ${{{ref}}}")
        return resolve(node, stack + (ref,))

    def resolve(obj: Any, stack: tuple[str, ...] = ()) -> Any:
        if isinstance(obj, str):
            m = _INTERP.fullmatch(obj)
            if m:  # whole string is one reference: keep the referenced type
                return lookup(m.group(1), stack)
            return _INTERP.sub(lambda mm: str(lookup(mm.group(1), stack)), obj)
        if isinstance(obj, dict):
            return {k: resolve(v, stack) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(v, stack) for v in obj]
        return obj

    return resolve(data)


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _absolute(value: Any, base_dir: Path | None) -> str:
    path = Path(str(value)).expanduser()
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def is_path_step(spec: str) -> bool:
    """True for step specs of the form ``path/to/file.py:Object``."""
    head, sep, _ = spec.rpartition(":")
    return bool(sep) and head.endswith(".py")


def load_env_files(
    explicit: list[Any] | None = None, *, base_dir: Path | None = None, override: bool = True
) -> list[Path]:
    """Load dotenv files into ``os.environ``.

    Files, in increasing priority (later files win when ``override`` is true):
    ``.env`` and ``.env.*`` of the project root of the working directory, the
    same for the project root of the YAML (usually identical), the config's
    explicit ``env_file`` entries, and ``$JWSTFLOW_ENV_FILE``. With
    ``override`` false, variables already set in the shell win instead.
    Returns the files that were loaded.
    """
    from dotenv import dotenv_values

    from ..project import candidate_env_files

    candidates: list[tuple[Path, str]] = candidate_env_files(base_dir)
    explicit_paths: list[tuple[Path, str]] = []
    for entry in explicit or []:
        path = Path(_absolute(entry, base_dir))
        if not path.is_file():
            raise ConfigError(f"env_file not found: {path}")
        explicit_paths.append((path, "env_file"))
    # $JWSTFLOW_ENV_FILE stays last (highest priority)
    tail = [c for c in candidates if c[1] == "$JWSTFLOW_ENV_FILE"]
    candidates = [c for c in candidates if c[1] != "$JWSTFLOW_ENV_FILE"] + explicit_paths + tail
    loaded: list[Path] = []
    for path, origin in candidates:
        path = path.resolve()
        if path in loaded or not path.is_file():
            continue
        for key, value in dotenv_values(path).items():
            if value is None:
                continue
            current = os.environ.get(key)
            if current is None or override:
                if current is not None and current != value:
                    log.info("%s: overriding the shell value with %s", key, path.name)
                os.environ[key] = value
            elif current != value:
                log.warning("%s from %s ignored: already set in the environment with a different value "
                            "(env_file_override is false)", key, path)
        loaded.append(path)
        log.info("loaded environment from %s (%s)", path, origin)
    return loaded


def _derive_stage_identity(stages: list[Any]) -> None:
    """Fill in `name` and `level` from the step; reject user-set values."""
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        for key in ("name", "level", "output_dir"):
            if key in stage:
                raise ConfigError(
                    f"stage {key!r} is derived from the step and cannot be set in the YAML; "
                    "use `variant:` to distinguish two stages with the same step"
                )
        if not isinstance(stage.get("step"), str):
            continue
        name, level = step_identity(stage["step"])
        stage["name"] = f"{name}-{stage['variant']}" if stage.get("variant") else name
        stage["level"] = level


def config_from_dict(data: dict[str, Any], *, base_dir: Path | None = None) -> Config:
    """Validate an already-composed dict (e.g. from OmegaConf/Hydra)."""
    from ..project import find_project_root

    data = dict(data)
    project_root = find_project_root(base_dir or Path.cwd())
    env_files = _as_list(data.get("env_file"))
    loaded = load_env_files(env_files, base_dir=base_dir, override=bool(data.get("env_file_override", True)))
    data["env_file"] = [str(p) for p in loaded]
    # run directory: explicit root, or <workspace>/<target>/<run>; known before interpolation
    workspace = _absolute(data["workspace"], base_dir) if data.get("workspace") else str(project_root / "reductions")
    data["workspace"] = workspace
    target_slug = slugify(str(data.get("target", "")))
    run = str(data.get("run", ""))
    if data.get("root"):
        run_dir = Path(_absolute(data["root"], base_dir))
        data["root"] = str(run_dir)
    else:
        run_dir = Path(workspace) / target_slug / run
    raw_dir = _absolute(data["download"]["dest"], base_dir) if isinstance(data.get("download"), dict) and data["download"].get("dest") else str(run_dir.parent / RAW_STAGE)
    data = interpolate(data, extra={"run_dir": str(run_dir), "target_dir": str(run_dir.parent), "raw_dir": raw_dir,
                                    "workspace": workspace, "project_root": str(project_root)})
    data["plugins"] = [_absolute(p, base_dir) for p in _as_list(data.get("plugins"))]
    activate_plugins(data["plugins"])  # stage identities below may need the user's step code
    if isinstance(data.get("download"), dict) and data["download"].get("dest"):
        data["download"]["dest"] = _absolute(data["download"]["dest"], base_dir)
    for stage in data.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        if isinstance(stage.get("step"), str) and is_path_step(stage["step"]):
            head, _, attr = stage["step"].rpartition(":")
            stage["step"] = f"{_absolute(head, base_dir)}:{attr}"
        for spec in stage.get("inputs") or []:
            if isinstance(spec, dict) and spec.get("path"):
                spec["path"] = _absolute(spec["path"], base_dir)  # explicit input dirs: relative to the YAML too
    _derive_stage_identity(data.get("stages") or [])
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc)) from exc


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """YAML file -> :class:`Config`. This is what the CLI uses."""
    path = Path(path).expanduser().resolve()
    data = load_raw(path)
    data = apply_overrides(data, overrides)
    return config_from_dict(data, base_dir=path.parent)


def format_validation_error(exc: ValidationError) -> str:
    lines = ["configuration is invalid:"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


def dump_config(cfg: Config) -> str:
    """YAML rendering of a validated config (used for the run manifest)."""
    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
