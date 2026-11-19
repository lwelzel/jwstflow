"""Run a whole workflow YAML offline: the overrides that cut the network, and the one-line runner."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .stubs import stub_pipelines


def offline_overrides(work: Path, *, backend: str = "serial", extra: Iterable[str] = ()) -> list[str]:
    """Config overrides that make any workflow run offline inside ``work``.

    The CRDS context is pinned to a fixed name (never contacted -- the stubs
    replace every CRDS-using stage), prefetch and the workflow graph are off,
    the run tree goes to ``<work>/reductions``, and tasks run serially so the
    in-process stub registry applies.
    """
    work = Path(work)
    return [
        f"workspace={work / 'reductions'}",
        f"crds.path={work / 'crds_cache'}",
        "crds.context=jwst_1364.pmap",
        "crds.prefetch=false",
        f"parallel.backend={backend}",
        "workflow_graph=false",
        *extra,
    ]


def run_mock_workflow(
    config_path: str | Path,
    work: Path,
    *,
    overrides: Iterable[str] = (),
    stubs: dict[str, Any] | None = None,
    only: Iterable[str] | None = None,
    start: str | None = None,
    until: str | None = None,
    force: bool = False,
) -> tuple[Any, Any]:
    """Load ``config_path`` with :func:`offline_overrides`, run it with stubbed
    pipelines, and return ``(cfg, RunSummary)``.

    Raw mock data is expected in the config's raw directory (see
    :func:`write_observation`); MAST download is skipped. ``overrides`` are
    appended after the offline set, so tests can tweak any stage parameter
    (``--set`` syntax, e.g. ``"stages.extract_extended.parameters.apcorr=false"``).
    """
    from ...config.loader import load_config
    from ...engine.runner import Runner

    cfg = load_config(config_path, overrides=offline_overrides(Path(work), extra=overrides))
    with stub_pipelines(stubs):
        summary = Runner(cfg, skip_download=True, only=only, start=start, until=until, force=force).run()
    return cfg, summary
