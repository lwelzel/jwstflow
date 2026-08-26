"""Stage ordering and selection.

Dependencies are implied by ``inputs[].stage`` plus explicit ``depends_on``.
Stages run in topological order with a barrier between them (all tasks of a
stage finish before the next stage plans its tasks), which keeps the model
simple and lets each stage re-scan its input directory.
"""

from __future__ import annotations

from collections.abc import Iterable
from graphlib import CycleError, TopologicalSorter

from ..config.schema import RAW_STAGE, Config, StageConfig


def dependencies(stage: StageConfig) -> set[str]:
    deps = {i.stage for i in stage.inputs if i.stage and i.stage != RAW_STAGE}
    deps.update(d for d in stage.depends_on if d != RAW_STAGE)
    return deps


def ordered_stages(cfg: Config) -> list[StageConfig]:
    """All stages in a valid execution order (input order breaks ties)."""
    by_name = {s.name: s for s in cfg.stages}
    ts: TopologicalSorter[str] = TopologicalSorter()
    for s in cfg.stages:
        ts.add(s.name, *dependencies(s))
    try:
        ts.prepare()
    except CycleError as exc:
        raise ValueError(f"stages form a cycle: {exc.args[1]}") from exc
    order: list[str] = []
    while ts.is_active():
        ready = sorted(ts.get_ready(), key=lambda n: list(by_name).index(n))
        order.extend(ready)
        ts.done(*ready)
    return [by_name[n] for n in order]


def select_stages(
    cfg: Config,
    *,
    only: Iterable[str] | None = None,
    start: str | None = None,
    until: str | None = None,
    tags: Iterable[str] | None = None,
) -> list[StageConfig]:
    """Apply ``--only/--from/--until/--tag`` to the ordered stage list."""
    stages = [s for s in ordered_stages(cfg) if s.enabled]
    names = [s.name for s in stages]
    for n in list(only or []) + [x for x in (start, until) if x]:
        if n not in {s.name for s in cfg.stages}:
            raise ValueError(f"unknown stage {n!r}; available: {', '.join(names)}")
    if only:
        wanted = set(only)
        stages = [s for s in stages if s.name in wanted]
    if start:
        idx = names.index(start)
        stages = [s for s in stages if names.index(s.name) >= idx]
    if until:
        idx = names.index(until)
        stages = [s for s in stages if names.index(s.name) <= idx]
    if tags:
        wanted = set(tags)
        stages = [s for s in stages if wanted & set(s.tags)]
    return stages
