"""Project root discovery and environment files.

A *project* is the directory tree that uses jwstflow: a git checkout, a uv
project, or any directory holding a ``.jwstflow-root`` marker. Its root is
where ``.env`` files live and where the default ``reductions/`` workspace is
created, so a workflow YAML can sit anywhere below it without relative
``../../..`` paths.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

ROOT_MARKERS = (".jwstflow-root", ".git", "pyproject.toml", "uv.lock")


def find_project_root(start: Path | None = None) -> Path:
    """Nearest ancestor of ``start`` (default: cwd) that contains a root marker.

    Falls back to ``start`` itself when no marker is found.
    """
    here = (start or Path.cwd()).resolve()
    if here.is_file():
        here = here.parent
    for directory in (here, *here.parents):
        if any((directory / m).exists() for m in ROOT_MARKERS):
            return directory
    return here


def env_files_in(directory: Path) -> list[Path]:
    """``.env`` and ``.env.*`` files of a directory (``.env.example`` excluded), sorted."""
    files = [p for p in directory.glob(".env*") if p.is_file() and not p.name.endswith(".example")]
    return sorted(files, key=lambda p: (p.name != ".env", p.name))


def candidate_env_files(yaml_dir: Path | None) -> list[tuple[Path, str]]:
    """Environment files in loading order (later wins): project root of the
    working directory, then the project root of the YAML, then ``$JWSTFLOW_ENV_FILE``."""
    out: list[tuple[Path, str]] = []
    roots: list[Path] = [find_project_root(Path.cwd())]
    if yaml_dir is not None:
        yaml_root = find_project_root(yaml_dir)
        if yaml_root not in roots:
            roots.append(yaml_root)
    for root in roots:
        out += [(p, f"project root {root}") for p in env_files_in(root)]
    if os.environ.get("JWSTFLOW_ENV_FILE"):
        out.append((Path(os.environ["JWSTFLOW_ENV_FILE"]).expanduser(), "$JWSTFLOW_ENV_FILE"))
    return out
