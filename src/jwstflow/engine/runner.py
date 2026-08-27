"""The orchestrator.

``Runner`` turns a validated :class:`Config` into work:

1. prepare directories, pin the CRDS context, write the run manifest;
2. download data (optional);
3. for each selected stage, in dependency order:
   a. discover inputs (upstream stage directory + header filters),
   b. optionally build association files,
   c. turn them into tasks with deterministic ids,
   d. skip the ones already checkpointed, run the rest on the executor,
   e. record every result.

Everything the runner needs from a stage is in ``StageConfig``; everything a
step needs is in the task payload. That separation is what makes the pieces
individually testable and replaceable.
"""

from __future__ import annotations

import fnmatch
import re
import os
import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .. import crds as crds_mod
from ..associations import Association, build_associations
from ..config.loader import dump_config
from ..config.schema import RAW_STAGE, Config, StageConfig
from ..data.discovery import FileRecord, HeaderCache, discover, fingerprint
from ..steps.base import (
    ALIAS_ASN_TYPE,
    BUILTIN_ALIASES,
    activate_plugins,
    default_thread_env,
    is_stpipe_step,
    resolve_target,
    source_fingerprint,
)
from .executor import Executor, Payload, Result, make_executor
from .graph import select_stages
from .state import StateStore, TaskRecord, make_task_id, now_iso, stable_hash

log = logging.getLogger(__name__)


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@dataclass
class Task:
    task_id: str
    stage: StageConfig
    label: str
    inputs: list[Path]
    fingerprints: dict[str, str]
    parameters: dict[str, Any]
    log_file: Path | None
    cached: TaskRecord | None = None
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "cached" if self.cached is not None else "pending"


@dataclass
class StageSummary:
    stage: str
    total: int = 0
    cached: int = 0
    success: int = 0
    failed: int = 0
    seconds: float = 0.0
    failures: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class RunSummary:
    stages: list[StageSummary] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return sum(s.failed for s in self.stages)

    @property
    def ok(self) -> bool:
        return self.failed == 0


class Runner:
    def __init__(
        self,
        cfg: Config,
        *,
        force: bool = False,
        dry_run: bool = False,
        only: Iterable[str] | None = None,
        start: str | None = None,
        until: str | None = None,
        tags: Iterable[str] | None = None,
        workers: int | None = None,
        skip_download: bool = False,
        tasks: Iterable[str] | None = None,
    ):
        self.cfg = cfg
        self.force = force
        self.dry_run = dry_run
        self.selection = dict(only=list(only or []) or None, start=start, until=until, tags=list(tags or []) or None)
        self.workers_override = workers
        self.skip_download = skip_download
        self.task_filter = list(tasks or [])  # glob patterns on task labels
        self._source_hashes: dict[str, str | None] = {}
        self._batch_modes: dict[str, str] = {}
        self.sys_path = activate_plugins(cfg.plugins)  # user step code, also handed to workers
        self.state = StateStore(cfg.state_dir / "state")
        self.headers = HeaderCache(cfg.state_dir / "headers.json")
        self.crds_context: str | None = None
        self._env: dict[str, str] = {}

    # ------------------------------------------------------------------ setup
    def prepare(self) -> None:
        cfg = self.cfg
        for d in (cfg.run_dir, cfg.state_dir, cfg.log_dir, cfg.asn_dir):
            d.mkdir(parents=True, exist_ok=True)
        log.info("preparing run directory %s", cfg.run_dir)
        try:
            self.crds_context = crds_mod.resolve_context(cfg.crds)
        except RuntimeError:
            if not self.dry_run:
                raise
            # planning must work offline; a real run insists on a pinned context
            log.warning("CRDS context could not be resolved; planning with an unpinned context")
            self.crds_context = None
        self._env = {
            **default_thread_env(cfg.parallel.threads_per_worker),
            **crds_mod.crds_environment(cfg.crds, self.crds_context),
            **cfg.env,
        }
        crds_mod.apply_environment(self._env)
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = self.cfg.state_dir / "manifest.json"
        entry = {
            "time": now_iso(),
            "name": self.cfg.name,
            "jwstflow": _pkg_version("jwstflow"),
            "jwst": _pkg_version("jwst"),
            "stpipe": _pkg_version("stpipe"),
            "crds": {**crds_mod.describe(), "pinned_context": self.crds_context},
            "selection": self.selection,
            "force": self.force,
            "dry_run": self.dry_run,
        }
        history: list[dict[str, Any]] = []
        if manifest.exists():
            try:
                history = json.loads(manifest.read_text()).get("runs", [])
            except json.JSONDecodeError:
                history = []
        history.append(entry)
        manifest.write_text(json.dumps({"runs": history[-50:]}, indent=2))
        (self.cfg.state_dir / "config.resolved.yaml").write_text(dump_config(self.cfg))

    @property
    def env_signature(self) -> dict[str, Any]:
        return {
            "jwst": _pkg_version("jwst"),
            "crds_context": self.crds_context or self.cfg.crds.context,
            "steppars_disabled": self.cfg.crds.disable_steppars,
        }

    # --------------------------------------------------------------- download
    def download(self) -> list[Path]:
        cfg = self.cfg
        if cfg.download is None or not cfg.download.enabled or self.skip_download:
            return []
        from ..data.download import download

        dest = cfg.raw_dir
        log.info("querying MAST for program %s, observations %s, %s %s -> %s", cfg.download.program,
                 cfg.download.observations or "all", cfg.download.instrument, cfg.download.modes or "", dest)
        files = download(cfg.download, dest, dry_run=self.dry_run)
        log.info("%d file(s) available in %s", len(files), dest)
        if cfg.download.reference_products and not self.dry_run:
            self._start_reference_download()
        if cfg.crds.prefetch and files and not self.dry_run:
            crds_mod.prefetch_references(files, cfg.crds, self.crds_context)
        return files

    def _start_reference_download(self) -> None:
        """Fetch MAST's own products in a background thread (network-bound; stages keep running)."""
        import threading

        from ..data.download import download_reference_products

        cfg = self.cfg
        assert cfg.download is not None

        def work() -> None:
            try:
                files = download_reference_products(cfg.download, cfg.reference_dir)
                log.info("MAST reference products: %d file(s) in %s", len(files), cfg.reference_dir)
            except Exception as exc:  # never take the run down for a QA extra
                log.error("MAST reference products download failed: %s", exc)

        log.info("fetching MAST reference products in the background -> %s", cfg.reference_dir)
        self._reference_thread = threading.Thread(target=work, name="mast-archive", daemon=True)
        self._reference_thread.start()

    def wait_for_reference(self) -> None:
        thread = getattr(self, "_reference_thread", None)
        if thread is not None and thread.is_alive():
            log.info("waiting for the MAST reference download to finish")
            thread.join()

    # --------------------------------------------------------------- planning
    def stage_inputs(self, stage: StageConfig) -> list[FileRecord]:
        records: dict[Path, FileRecord] = {}
        for spec in stage.inputs:
            if spec.path:
                directory = spec.path.expanduser()
            elif spec.run:
                directory = self.cfg.sibling_stage_dir(spec.run, spec.stage or RAW_STAGE)
            else:
                directory = self.cfg.stage_dir(spec.stage or RAW_STAGE)
            found = discover(
                directory,
                spec.pattern,
                recursive=spec.recursive,
                exclude=spec.exclude,
                filters=spec.filters,
                cache=self.headers,
            )
            for r in found:
                records.setdefault(r.path.resolve(), r)
            log.debug("stage %s: %d file(s) from %s/%s", stage.name, len(found), directory, spec.pattern)
        self.headers.save()
        targids = self.mast_target_ids()
        for r in records.values():
            key = (str(r.get("PROGRAM") or "").zfill(5), str(r.get("OBSERVTN") or "").zfill(3))
            r.meta.setdefault("TARGID", targids.get(key) or _fallback_targid(r))
        return list(records.values())

    def mast_target_ids(self) -> dict[tuple[str, str], str]:
        """(program, observation) -> DMS target id ('t010') recorded by the download step."""
        from ..data.download import target_ids_from_log

        if not hasattr(self, "_targids"):
            self._targids = target_ids_from_log(self.cfg.raw_dir)
        return self._targids

    def plan_stage(self, stage: StageConfig) -> list[Task]:
        log.info("stage %s: selecting inputs and building tasks", stage.name)
        records = self.stage_inputs(stage)
        out_dir = self.cfg.stage_dir(stage.name)
        log_dir = self.cfg.log_dir / stage.name
        tasks: list[Task] = []
        if stage.association is not None:
            asn_type = stage.association.asn_type or _infer_asn_type(stage, records)
            asns = build_associations(records, stage.association, asn_type=asn_type, stage=stage.name)
            asn_dir = self.cfg.asn_dir / stage.name
            for asn in asns:
                path = _write_if_changed(asn, asn_dir, relative=stage.association.relative_paths)
                fps = fingerprint(asn.members, self.cfg.checkpoint.fingerprint)
                params = dict(stage.parameters)
                extra = {"asn": _normalized(asn.data)}
                tasks.append(self._task(stage, asn.name, [path], fps, params, extra, out_dir, log_dir))
            if records and not asns:
                log.warning("stage %s: %d input file(s) but no association could be built", stage.name, len(records))
        elif self.batch_mode(stage) == "all":
            if records:
                paths = [r.path for r in records]
                fps = fingerprint(paths, self.cfg.checkpoint.fingerprint)
                tasks.append(self._task(stage, "all", paths, fps, dict(stage.parameters), {}, out_dir, log_dir))
        else:
            for r in records:
                fps = fingerprint([r.path], self.cfg.checkpoint.fingerprint)
                tasks.append(self._task(stage, r.stem, [r.path], fps, dict(stage.parameters), {}, out_dir, log_dir))
        if not records:
            log.warning("stage %s: no input files found", stage.name)
        return tasks

    def batch_mode(self, stage: StageConfig) -> str:
        """'per_file' or 'all': the YAML value if given, else the step's own `batch`
        attribute (jwst steps and pipelines are always per_file)."""
        if stage.batch is not None:
            return stage.batch
        if stage.name not in self._batch_modes:
            mode = "per_file"
            if stage.step not in BUILTIN_ALIASES:
                try:
                    target = resolve_target(stage.step)
                except Exception as exc:  # unresolvable steps fail later with a proper message
                    log.debug("cannot resolve %s to read its batch mode: %s", stage.step, exc)
                else:
                    if not is_stpipe_step(target):
                        mode = str(getattr(target, "batch", "per_file"))
            self._batch_modes[stage.name] = mode
        return self._batch_modes[stage.name]

    def _task(
        self,
        stage: StageConfig,
        label: str,
        inputs: list[Path],
        fps: dict[str, str],
        params: dict[str, Any],
        extra: dict[str, Any],
        out_dir: Path,
        log_dir: Path,
    ) -> Task:
        signature = {**self.env_signature, **extra, "save_results": stage.save_results}
        if stage.step not in self._source_hashes:
            self._source_hashes[stage.step] = source_fingerprint(stage.step)
        if self._source_hashes[stage.step]:
            signature["step_source"] = self._source_hashes[stage.step]
        tid = make_task_id(stage.name, stage.step, [str(p) for p in inputs], params, fps, signature)
        cached = None
        if not (self.force or stage.force):
            rec = self.state.get(stage.name, tid)
            if rec is not None and rec.status == "success" and (
                not self.cfg.checkpoint.require_outputs or rec.outputs_exist()
            ):
                cached = rec
        log_file = (log_dir / f"{label}.log") if self.cfg.logging.per_task_logs else None
        return Task(tid, stage, label, inputs, fps, params, log_file, cached, info={"output_dir": str(out_dir)})

    def payload(self, task: Task) -> Payload:
        stage = task.stage
        params = dict(task.parameters)
        return {
            "task_id": task.task_id,
            "stage": stage.name,
            "label": task.label,
            "step": stage.step,
            "inputs": [str(p) for p in task.inputs],
            "parameters": params,
            "output_dir": str(self.cfg.stage_dir(stage.name)),
            "log_dir": str(self.cfg.log_dir / stage.name),
            "log_file": str(task.log_file) if task.log_file else None,
            "log_level": self.cfg.logging.level,
            "run_name": self.cfg.name,
            "root": str(self.cfg.run_dir),
            "raw_dir": str(self.cfg.stage_dir(RAW_STAGE)),
            "stage_dirs": {s.name: str(self.cfg.stage_dir(s.name)) for s in self.cfg.stages},
            "crds_context": self.crds_context,
            "dry_run": self.dry_run,
            "env": self._env,
            "sys_path": list(self.sys_path),
            "target": self.cfg.target,
            "target_coords": self.cfg.target_coords,
            "target_dir": str(self.cfg.target_dir),
            "reference_dir": str(self.cfg.reference_dir),
            "extra": {"save_results": stage.save_results},
        }

    # -------------------------------------------------------------- execution
    def run_stage(self, stage: StageConfig, tasks: list[Task] | None = None) -> StageSummary:
        if stage.name.startswith("mast_compare"):
            self.wait_for_reference()
        tasks = self.plan_stage(stage) if tasks is None else tasks
        if self.task_filter:
            tasks = [t for t in tasks if any(fnmatch.fnmatch(t.label, pat) for pat in self.task_filter)]
        summary = StageSummary(stage.name, total=len(tasks))
        pending = [t for t in tasks if t.cached is None]
        summary.cached = len(tasks) - len(pending)
        log.info(
            "stage %-14s %3d task(s): %d cached, %d to run", stage.name, len(tasks), summary.cached, len(pending)
        )
        if self.dry_run or not pending:
            return summary
        workers = self.workers_override or stage.workers or self.cfg.parallel.workers
        if not stage.parallel:
            workers = 1
        executor = make_executor(self.cfg.parallel.backend, workers, self._env, self.cfg.parallel.scheduler)
        t0 = time.time()
        by_id = {t.task_id: t for t in pending}
        try:
            for payload, result in executor.run([self.payload(t) for t in pending]):
                task = by_id[payload["task_id"]]
                self._record(task, payload, result, summary)
                if result["status"] != "success" and stage.on_error == "fail":
                    raise RuntimeError(f"stage {stage.name}: task {task.label} failed: {result.get('error')}")
        finally:
            executor.close()
        summary.seconds = time.time() - t0
        self._report_orphans(stage, tasks)
        log.info(
            "stage %-14s done in %.1fs: %d ok, %d failed", stage.name, summary.seconds, summary.success, summary.failed
        )
        return summary

    def orphans(self, stage: StageConfig, tasks: list[Task] | None = None) -> list[Path]:
        """Files in the stage directory that no current task produced (e.g. left over from an
        earlier run with different parameters or product names). They would otherwise be
        picked up by downstream stages; `jwstflow clean --orphans` removes them."""
        tasks = self.plan_stage(stage) if tasks is None else tasks
        out_dir = self.cfg.stage_dir(stage.name)
        if not out_dir.is_dir():
            return []
        current = {t.task_id for t in tasks}
        produced: set[Path] = set()
        for rec in self.state.records(stage.name):
            if rec.task_id in current and rec.status == "success":
                produced.update(Path(o).resolve() for o in rec.outputs)
        return sorted(
            p for p in out_dir.rglob("*")
            if p.is_file() and not any(part.startswith(".tmp-") for part in p.parts) and p.resolve() not in produced
        )

    def _report_orphans(self, stage: StageConfig, tasks: list[Task]) -> None:
        orphans = self.orphans(stage, tasks)
        if orphans:
            log.warning(
                "stage %s: %d file(s) in %s were not produced by the current tasks (e.g. %s); "
                "downstream stages will see them too -- `jwstflow clean --stage %s --orphans` removes them",
                stage.name, len(orphans), self.cfg.stage_dir(stage.name), orphans[0].name, stage.name,
            )

    def _record(self, task: Task, payload: Payload, result: Result, summary: StageSummary) -> None:
        rec = TaskRecord(
            task_id=task.task_id,
            stage=task.stage.name,
            step=task.stage.step,
            label=task.label,
            inputs=payload["inputs"],
            fingerprints=task.fingerprints,
            parameters=task.parameters,
            outputs=result.get("outputs", []),
            status=result["status"],
            finished=now_iso(),
            duration_s=result.get("duration_s"),
            error=result.get("error"),
            traceback=result.get("traceback"),
            log_file=payload.get("log_file"),
            env=self.env_signature,
        )
        rec.started = (
            datetime.fromtimestamp(time.time() - (rec.duration_s or 0), tz=timezone.utc).isoformat(timespec="seconds")
        )
        self.state.put(rec)
        if rec.status == "success":
            summary.success += 1
            log.info("  ok      %s (%.1fs, %d file(s))", task.label, rec.duration_s or 0, len(rec.outputs))
        else:
            summary.failed += 1
            summary.failures.append((task.label, rec.error or "?"))
            log.error("  FAILED  %s: %s  [log: %s]", task.label, rec.error, rec.log_file)

    def debug_task(self, stage: str, label: str | None = None) -> tuple[Any, list[Path], Any, dict[str, Any]]:
        """Materialise one task for running by hand: ``(step, inputs, ctx, params)``.

        Meant for notebooks and ``pdb``::

            step, inputs, ctx, params = Runner(cfg).debug_task("extract")
            outputs = step.run(inputs, ctx, **params)

        ``label`` selects the task (glob allowed); default: the first one. The
        step is resolved in *this* process and nothing is recorded in the
        checkpoint store.
        """
        from ..steps.base import RunContext, make_step

        self.prepare()
        st = self.cfg.stage(stage)
        tasks = self.plan_stage(st)
        if label is not None:
            tasks = [t for t in tasks if fnmatch.fnmatch(t.label, label)]
        if not tasks:
            raise ValueError(f"stage {stage!r}: no task matches {label!r}")
        task = tasks[0]
        payload = self.payload(task)
        ctx = RunContext(
            run_name=payload["run_name"],
            root=Path(payload["root"]),
            stage=payload["stage"],
            output_dir=Path(payload["output_dir"]),
            log_dir=Path(payload["log_dir"]),
            raw_dir=Path(payload["raw_dir"]),
            stage_dirs={k: Path(v) for k, v in payload["stage_dirs"].items()},
            crds_context=payload["crds_context"],
            dry_run=False,
            task_id=payload["task_id"],
            extra=payload["extra"],
            target=payload.get("target", ""),
            target_coords=payload.get("target_coords"),
            target_dir=Path(payload["target_dir"]) if payload.get("target_dir") else None,
            reference_dir=Path(payload["reference_dir"]) if payload.get("reference_dir") else None,
        )
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        return make_step(st.step), list(task.inputs), ctx, dict(task.parameters)

    def run(self) -> RunSummary:
        self.prepare()
        stages = select_stages(self.cfg, **self.selection)
        log.info("run %r in %s  (%s)", self.cfg.name, self.cfg.run_dir, " -> ".join(s.name for s in stages))
        if self.selection["start"] is None and self.selection["only"] is None:
            self.download()
        summary = RunSummary()
        for stage in stages:
            summary.stages.append(self.run_stage(stage))
        self.wait_for_reference()
        return summary

    def plan(self) -> dict[str, list[Task]]:
        """All tasks of the selected stages without running anything (see `jwstflow plan`)."""
        self.prepare()
        return {s.name: self.plan_stage(s) for s in select_stages(self.cfg, **self.selection)}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


_IMAGING_EXPTYPES = ("NRC_IMAGE", "MIR_IMAGE", "NIS_IMAGE", "FGS_IMAGE", "NRC_CORON", "MIR_LYOT", "MIR_4QPM")


def _infer_asn_type(stage: StageConfig, records: list[FileRecord]) -> str:
    """asn_type for the association file: alias table, then the step's name
    (``Spec3WithBackground`` -> spec3), then the level + EXP_TYPE of the inputs."""
    if stage.step in ALIAS_ASN_TYPE:
        return ALIAS_ASN_TYPE[stage.step]
    lowered = stage.step.rpartition(":")[2].lower() or stage.step.lower()
    for key in ("spec3", "image3", "tso3", "coron3", "spec2", "image2"):
        if key in lowered:
            return key
    assert stage.association is not None
    imaging = any(str(r.get("EXP_TYPE", "")).upper() in _IMAGING_EXPTYPES for r in records)
    return f"{'image' if imaging else 'spec'}{stage.association.level}"


def _fallback_targid(rec: FileRecord) -> str:
    """When MAST's target id is unknown: the DMS-style slug of TARGPROP (never a bare 'tNNN')."""
    from ..config.schema import slugify

    return slugify(str(rec.get("TARGPROP") or rec.get("TARGNAME") or "target"))


def _normalized(asn_data: dict[str, Any]) -> str:
    data = dict(asn_data)
    data.pop("version_id", None)
    return stable_hash(data)


def _write_if_changed(asn: Association, directory: Path, *, relative: bool) -> Path:
    """Write the association file only when the file *as it would be written*
    differs (ignoring version_id) from what is on disk, so mtimes stay stable
    across runs -- but a change of member-path form (relative vs absolute) is
    a real change and is written out."""
    path = directory / asn.filename
    if path.exists():
        try:
            old = json.loads(path.read_text())
            if _normalized(old) == _normalized(asn.serialize(directory, relative=relative)):
                return path
        except (json.JSONDecodeError, KeyError):
            pass
    return asn.write(directory, relative=relative)


def _abs_members(data: dict[str, Any], base: Path) -> dict[str, Any]:
    data = json.loads(json.dumps(data))
    for product in data.get("products", []):
        for m in product.get("members", []):
            p = Path(m["expname"])
            m["expname"] = str((base / p).resolve() if not p.is_absolute() else p.resolve())
    return data


def run_config(cfg: Config, **kwargs: Any) -> RunSummary:
    """Convenience: ``run_config(load_config('run.yaml'), workers=8)``."""
    return Runner(cfg, **kwargs).run()
