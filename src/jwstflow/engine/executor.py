"""Executors: run task payloads serially, in a spawn-based process pool, or on dask.

Design rules that keep this robust with the jwst pipeline:

* A task payload is a plain dict (JSON-serialisable). Nothing heavy crosses
  the process boundary; the worker resolves the step, imports ``jwst`` (only
  there, to avoid a known memory leak when importing it in the parent) and
  returns another plain dict.
* Workers are started with the ``spawn`` method, as the jwst docs recommend,
  and receive the CRDS/thread environment through an initializer.
* Every task gets its own log file; stdout/stderr of the step are captured
  into it as well so the terminal stays readable.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
import warnings
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

Payload = dict[str, Any]
Result = dict[str, Any]


# ---------------------------------------------------------------------------
# worker side
# ---------------------------------------------------------------------------


def init_worker(env: dict[str, str], sys_path: list[str] | None = None) -> None:
    """Process-pool initializer: export environment before anything imports jwst,
    and make the user's step code importable (`plugins:` / file-based steps)."""
    for k, v in env.items():
        os.environ[k] = v
    for d in reversed(sys_path or []):
        if d not in sys.path:
            sys.path.insert(0, d)


def execute_task(payload: Payload) -> Result:
    """Run one task. Never raises; failures are reported in the result dict."""
    from ..steps.base import RunContext, check_outputs, make_step

    init_worker(payload.get("env", {}), payload.get("sys_path"))
    t0 = time.time()
    log_file = payload.get("log_file")
    result: Result = {"task_id": payload["task_id"], "status": "failed", "outputs": []}
    warnings.filterwarnings("ignore", message="Input association file contains path information")
    with _task_logging(log_file, payload.get("log_level", "INFO")):
        logger = logging.getLogger("jwstflow.task")
        logger.info("=== %s / %s  (%s)", payload["stage"], payload.get("label", ""), payload["task_id"])
        logger.info("inputs: %s", ", ".join(payload["inputs"]))
        logger.info("parameters: %s", payload["parameters"])
        try:
            step = make_step(payload["step"])
            ctx = RunContext(
                run_name=payload["run_name"],
                root=Path(payload["root"]),
                stage=payload["stage"],
                output_dir=Path(payload["output_dir"]),
                log_dir=Path(payload["log_dir"]),
                raw_dir=Path(payload["raw_dir"]),
                stage_dirs={k: Path(v) for k, v in payload.get("stage_dirs", {}).items()},
                crds_context=payload.get("crds_context"),
                dry_run=payload.get("dry_run", False),
                task_id=payload["task_id"],
                extra=payload.get("extra", {}),
                target=payload.get("target", ""),
                target_coords=payload.get("target_coords"),
                target_dir=Path(payload["target_dir"]) if payload.get("target_dir") else None,
                reference_dir=Path(payload["reference_dir"]) if payload.get("reference_dir") else None,
            )
            ctx.output_dir.mkdir(parents=True, exist_ok=True)
            outputs = step.run([Path(p) for p in payload["inputs"]], ctx, **payload["parameters"])
            outputs = check_outputs(outputs, ctx, step)
            result["outputs"] = sorted({str(Path(o)) for o in outputs})
            result["status"] = "success"
            logger.info("done: %d output(s)", len(result["outputs"]))
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc()
            logger.error("task failed: %s", result["error"])
            logger.debug(result["traceback"])
    result["duration_s"] = round(time.time() - t0, 2)
    return result


@contextlib.contextmanager
def _task_logging(log_file: str | None, level: str) -> Iterator[None]:
    """Route all logging (jwstflow, stpipe, jwst, CRDS, warnings) + stdout/stderr
    into ``log_file`` for the duration of a task."""
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.captureWarnings(True)
    handler: logging.Handler | None = None
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, mode="w")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
        )
        root.addHandler(handler)
    stream = handler.stream if handler is not None else None  # type: ignore[union-attr]
    try:
        if stream is not None and os.environ.get("JWSTFLOW_CAPTURE_STDOUT", "1") == "1":
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                yield
        else:
            yield
    finally:
        if handler is not None:
            root.removeHandler(handler)
            handler.close()
        root.setLevel(old_level)


# ---------------------------------------------------------------------------
# executors
# ---------------------------------------------------------------------------


class Executor:
    """Runs ``execute_task`` over payloads and yields ``(payload, result)`` as they finish."""

    def __init__(self, workers: int = 1, env: dict[str, str] | None = None):
        self.workers = max(1, workers)
        self.env = env or {}

    def run(self, payloads: Iterable[Payload]) -> Iterator[tuple[Payload, Result]]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SerialExecutor(Executor):
    """In-process execution (the task's own logging reaches the console directly)."""

    def run(self, payloads: Iterable[Payload]) -> Iterator[tuple[Payload, Result]]:
        init_worker(self.env)
        for p in payloads:
            yield p, execute_task(p)


class ProcessExecutor(Executor):
    """``concurrent.futures.ProcessPoolExecutor`` with the spawn start method."""

    def __init__(self, workers: int = 1, env: dict[str, str] | None = None, start_method: str = "spawn"):
        super().__init__(workers, env)
        self.start_method = start_method

    def run(self, payloads: Iterable[Payload]) -> Iterator[tuple[Payload, Result]]:
        payloads = list(payloads)
        ctx = mp.get_context(self.start_method)
        with ProcessPoolExecutor(
            max_workers=min(self.workers, len(payloads)) or 1,
            mp_context=ctx,
            initializer=init_worker,
            initargs=(self.env,),
        ) as pool:
            futures: dict[Future[Result], Payload] = {pool.submit(execute_task, p): p for p in payloads}
            pending = set(futures)
            while pending:
                try:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                except BrokenProcessPool:  # pragma: no cover - e.g. OOM-killed worker
                    for f in pending:
                        yield futures[f], _broken(futures[f], "worker process died (out of memory?)")
                    return
                for f in done:
                    payload = futures[f]
                    try:
                        yield payload, f.result()
                    except BrokenProcessPool:  # pragma: no cover
                        yield payload, _broken(payload, "worker process died (out of memory?)")
                        for g in pending:
                            yield futures[g], _broken(futures[g], "pool broken")
                        return
                    except Exception as exc:  # pragma: no cover
                        yield payload, _broken(payload, f"{type(exc).__name__}: {exc}")


class DaskExecutor(Executor):
    """dask.distributed backend (LocalCluster or an existing scheduler)."""

    def __init__(self, workers: int = 1, env: dict[str, str] | None = None, scheduler: str | None = None):
        super().__init__(workers, env)
        self.scheduler = scheduler
        self._client: Any = None
        self._cluster: Any = None

    def _connect(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from dask.distributed import Client, LocalCluster
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("parallel.backend=dask needs `pip install dask[distributed]`") from exc
        if self.scheduler:
            self._client = Client(self.scheduler)
        else:
            self._cluster = LocalCluster(n_workers=self.workers, threads_per_worker=1, processes=True)
            self._client = Client(self._cluster)
        self._client.run(init_worker, self.env)
        return self._client

    def run(self, payloads: Iterable[Payload]) -> Iterator[tuple[Payload, Result]]:
        from dask.distributed import as_completed

        client = self._connect()
        futures = {client.submit(execute_task, p, pure=False): p for p in payloads}
        for f in as_completed(list(futures)):
            payload = futures[f]
            try:
                yield payload, f.result()
            except Exception as exc:  # pragma: no cover
                yield payload, _broken(payload, f"{type(exc).__name__}: {exc}")

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
        if self._cluster is not None:
            self._cluster.close()


def _broken(payload: Payload, msg: str) -> Result:
    return {"task_id": payload["task_id"], "status": "failed", "outputs": [], "error": msg}


def make_executor(backend: str, workers: int, env: dict[str, str], scheduler: str | None = None) -> Executor:
    if backend == "serial" or workers <= 1 and backend != "dask":
        return SerialExecutor(1, env)
    if backend == "process":
        return ProcessExecutor(workers, env)
    if backend == "dask":
        return DaskExecutor(workers, env, scheduler)
    raise ValueError(f"unknown parallel backend {backend!r}")
