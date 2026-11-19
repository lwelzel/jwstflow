"""User step code: importing objects by path, and the ``plugins:`` of a workflow.

A ``step:`` may name a Python file (``./steps.py:MyStep``) and a workflow may list
``plugins:`` -- files or directories with step code. This module makes that code
importable, in the parent process and in every worker, and fingerprints it so that
editing a plugin file invalidates the checkpoints of the steps defined there.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def is_path_spec(spec: str) -> bool:
    """True for ``path/to/file.py:Object`` step specs."""
    head, sep, _ = spec.rpartition(":")
    return bool(sep) and head.endswith(".py")


def import_file(path: str | Path) -> Any:
    """Import a Python file as a module named after its stem (cached in sys.modules).

    The file's directory is put on ``sys.path`` first -- exactly what
    :func:`activate_plugins` does for a workflow's ``plugins:`` -- so a plugin
    file behaves the same under ``jwstflow check-step ./steps.py:X`` as inside
    a run, including imports of sibling modules.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ImportError(f"step file not found: {path}")
    directory = str(path.parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
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


def plugins_fingerprint(paths: Iterable[str | Path]) -> str | None:
    """One short hash over every plugin source file of a workflow.

    ``plugins:`` files may import each other (helpers next to a workflow's
    ``steps.py``), so a step's own file hash is not enough to invalidate its
    checkpoints -- this combined hash is folded into the task identity of
    every step that is defined inside a plugin location. Files are hashed
    directly; for directories, their top-level ``*.py`` files.
    """
    entries: list[tuple[str, str]] = []
    for entry in paths:
        path = Path(entry).expanduser().resolve()
        files = [path] if path.is_file() else sorted(path.glob("*.py")) if path.is_dir() else []
        for f in files:
            entries.append((f.name, hashlib.sha1(f.read_bytes()).hexdigest()))
    if not entries:
        return None
    combined = hashlib.sha1(repr(sorted(entries)).encode())
    return combined.hexdigest()[:12]
