"""CRDS management.

The jwst pipeline talks to CRDS through environment variables that must be
set *before* ``jwst``/``crds`` are imported. jwstflow therefore

1. computes the environment from :class:`CRDSConfig` (:func:`crds_environment`),
2. exports it in the parent and in every worker (workers are spawned, so the
   environment is re-applied by the executor initializer),
3. optionally resolves ``context: latest`` to a concrete ``.pmap`` and pins it
   in the run manifest,
4. optionally prefetches references so cluster nodes can run offline.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from .config.schema import CRDSConfig

log = logging.getLogger(__name__)


def crds_environment(cfg: CRDSConfig, context: str | None = None) -> dict[str, str]:
    """Environment variables implementing ``cfg``."""
    env = {
        "CRDS_PATH": str(Path(cfg.path).expanduser()),
        "CRDS_SERVER_URL": cfg.server_url,
    }
    ctx = context or cfg.context
    if ctx and ctx != "latest":
        env["CRDS_CONTEXT"] = ctx
    if cfg.disable_steppars:
        env["STPIPE_DISABLE_CRDS_STEPPARS"] = "true"
    if cfg.readonly_cache:
        env["CRDS_READONLY_CACHE"] = "1"
    return env


def apply_environment(env: dict[str, str]) -> None:
    for k, v in env.items():
        os.environ[k] = v
    Path(os.environ.get("CRDS_PATH", "")).expanduser().mkdir(parents=True, exist_ok=True)


def resolve_context(cfg: CRDSConfig) -> str | None:
    """Return the context to pin: explicit value, or the server default for 'latest'."""
    if cfg.context is None:
        return None
    if cfg.context != "latest":
        return cfg.context
    apply_environment(crds_environment(cfg))
    try:
        import crds

        ctx = crds.get_default_context("jwst")
    except ImportError as exc:
        raise RuntimeError("crds.context: latest needs the `crds` package (pip install crds)") from exc
    except Exception as exc:  # network problems etc.
        raise RuntimeError(f"could not resolve the latest CRDS context: {exc}") from exc
    log.info("pinned CRDS context to %s", ctx)
    return str(ctx)


def prefetch_references(files: Iterable[Path], cfg: CRDSConfig, context: str | None = None) -> None:
    """Download all references needed by ``files`` (``crds bestrefs --sync-references``)."""
    files = [str(f) for f in files]
    if not files:
        return
    env = dict(os.environ)
    env.update(crds_environment(cfg, context))
    cmd = [sys.executable, "-m", "crds.bestrefs", "--files", *files, "--sync-references=1"]
    if context or (cfg.context and cfg.context != "latest"):
        cmd += ["--new-context", context or cfg.context]  # type: ignore[list-item]
    log.info("prefetching CRDS references for %d file(s)", len(files))
    subprocess.run(cmd, env=env, check=True)


def describe() -> dict[str, str | None]:
    """Current CRDS-related environment (for manifests / `jwstflow status`)."""
    keys = ("CRDS_PATH", "CRDS_SERVER_URL", "CRDS_CONTEXT", "STPIPE_DISABLE_CRDS_STEPPARS")
    return {k: os.environ.get(k) for k in keys}
