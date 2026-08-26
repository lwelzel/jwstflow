"""Checkpointing.

Every task (one file or one association through one step) has a deterministic
``task_id`` = hash of (stage, step, resolved parameters, input fingerprints,
software/CRDS versions). After a task succeeds its record is written to
``<root>/.jwstflow/state/<stage>/<task_id>.json``. On the next run a task is
skipped when its record exists, says ``success``, and all recorded outputs are
still on disk. Changing a parameter, an input file, the pipeline version or
the CRDS context changes the id, so the task reruns automatically.

Plain JSON files, one per task: inspectable, diffable, trivially deletable
(``jwstflow clean``) and safe under concurrent writers.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha1(payload).hexdigest()


def make_task_id(
    stage: str,
    step: str,
    inputs: Iterable[str],
    parameters: Mapping[str, Any],
    fingerprints: Mapping[str, str],
    env_signature: Mapping[str, Any],
) -> str:
    return stable_hash(
        {
            "stage": stage,
            "step": step,
            "inputs": sorted(str(i) for i in inputs),
            "parameters": parameters,
            "fingerprints": dict(fingerprints),
            "env": dict(env_signature),
        }
    )[:16]


class TaskRecord(BaseModel):
    task_id: str
    stage: str
    step: str
    label: str = ""
    inputs: list[str] = Field(default_factory=list)
    fingerprints: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)
    status: str = "pending"  # pending | running | success | failed | skipped
    started: str | None = None
    finished: str | None = None
    duration_s: float | None = None
    error: str | None = None
    traceback: str | None = None
    log_file: str | None = None
    env: dict[str, Any] = Field(default_factory=dict)
    info: dict[str, Any] = Field(default_factory=dict)

    def outputs_exist(self) -> bool:
        return all(Path(o).exists() for o in self.outputs)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StateStore:
    def __init__(self, directory: Path):
        self.dir = Path(directory)

    def _path(self, stage: str, task_id: str) -> Path:
        return self.dir / stage / f"{task_id}.json"

    def get(self, stage: str, task_id: str) -> TaskRecord | None:
        p = self._path(stage, task_id)
        if not p.exists():
            return None
        try:
            return TaskRecord.model_validate_json(p.read_text())
        except Exception:
            return None

    def put(self, rec: TaskRecord) -> Path:
        p = self._path(rec.stage, rec.task_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=p.parent)
        with os.fdopen(fd, "w") as fh:
            fh.write(rec.model_dump_json(indent=2))
        os.replace(tmp, p)
        return p

    def is_complete(self, stage: str, task_id: str, *, require_outputs: bool = True) -> bool:
        rec = self.get(stage, task_id)
        if rec is None or rec.status != "success":
            return False
        return rec.outputs_exist() if require_outputs else True

    def records(self, stage: str | None = None) -> list[TaskRecord]:
        out: list[TaskRecord] = []
        dirs = [self.dir / stage] if stage else sorted(d for d in self.dir.glob("*") if d.is_dir())
        for d in dirs:
            for p in sorted(d.glob("*.json")):
                try:
                    out.append(TaskRecord.model_validate_json(p.read_text()))
                except Exception:
                    continue
        return out

    def clear(self, stage: str | None = None) -> int:
        n = 0
        dirs = [self.dir / stage] if stage else list(self.dir.glob("*"))
        for d in dirs:
            if d.is_dir():
                for p in d.glob("*.json"):
                    p.unlink()
                    n += 1
        return n
