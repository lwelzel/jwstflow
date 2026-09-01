"""Engine behaviour: planning, checkpoints, failure policy, orphans, selection, state.

Everything runs tiny registered function steps on text files -- no FITS, no
astropy -- so these tests pin down the orchestration semantics (what reruns
when, what a failure does, what lands in the state store) at unit-test speed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jwstflow.config.loader import config_from_dict
from jwstflow.engine.graph import ordered_stages, select_stages
from jwstflow.engine.runner import Runner
from jwstflow.engine.state import StateStore, TaskRecord, make_task_id
from jwstflow.steps.base import _REGISTRY, register_step


@pytest.fixture()
def steps_registry():
    """Register the toy steps for one test, restoring the registry afterwards."""
    calls: dict[str, int] = {"boom": 0}

    @register_step("make_copies")
    def make_copies(inputs, ctx, *, tag: str = "copy", **params):
        out = []
        for src in inputs:
            dst = ctx.output_dir / f"{src.stem}.copy.txt"   # stable name: a tag change rewrites in place
            dst.write_text(src.read_text() + f"+{tag}")
            out.append(dst)
        return out

    make_copies.level = 4

    @register_step("combine_all")
    def combine_all(inputs, ctx, **params):
        dst = ctx.output_dir / "combined.txt"
        dst.write_text("|".join(sorted(p.name for p in inputs)))
        return [dst]

    combine_all.batch = "all"
    combine_all.level = 4

    @register_step("boom")
    def boom(inputs, ctx, **params):
        calls["boom"] += 1
        raise RuntimeError("kaboom")

    boom.level = 4

    yield calls
    for name in ("make_copies", "combine_all", "boom"):
        _REGISTRY.pop(name, None)


def base_config(tmp_path: Path, stages: list[dict[str, Any]]) -> Any:
    (tmp_path / "target" / "raw").mkdir(parents=True, exist_ok=True)
    return config_from_dict(
        {
            "target": "T",
            "run": "r1",
            "root": str(tmp_path / "target" / "r1"),
            "crds": {"context": "jwst_1364.pmap", "prefetch": False},
            "parallel": {"backend": "serial", "workers": 1},
            "workflow_graph": False,
            "stages": stages,
        },
        base_dir=tmp_path,
    )


def seed_raw(cfg: Any, names: list[str]) -> list[Path]:
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for n in names:
        p = cfg.raw_dir / n
        p.write_text(f"data:{n}")
        out.append(p)
    return out


def test_checkpoints_skip_and_invalidate(tmp_path: Path, steps_registry):
    cfg = base_config(tmp_path, [
        {"step": "make_copies", "inputs": [{"stage": "raw", "pattern": "*.txt"}]},
        {"step": "combine_all", "inputs": [{"stage": "make_copies", "pattern": "*.txt"}]},
    ])
    seed_raw(cfg, ["a.txt", "b.txt"])
    summary = Runner(cfg, skip_download=True).run()
    assert summary.ok and [(s.stage, s.success) for s in summary.stages] == [("make_copies", 2), ("combine_all", 1)]
    assert (cfg.stage_dir("combine_all") / "combined.txt").read_text() == "a.copy.txt|b.copy.txt"
    assert (cfg.stage_dir("make_copies") / "a.copy.txt").read_text() == "data:a.txt+copy"

    # unchanged rerun: everything cached
    summary2 = Runner(cfg, skip_download=True).run()
    assert all(s.cached == s.total and s.success == 0 for s in summary2.stages)

    # a parameter change invalidates the stage it touches, and the rewritten outputs
    # (new mtimes) cascade into the downstream stage's fingerprints
    cfg.stage("make_copies").parameters["tag"] = "v2"
    summary3 = Runner(cfg, skip_download=True).run()
    by = {s.stage: s for s in summary3.stages}
    assert by["make_copies"].success == 2 and by["make_copies"].cached == 0
    assert by["combine_all"].success == 1 and by["combine_all"].cached == 0
    assert (cfg.stage_dir("make_copies") / "a.copy.txt").read_text() == "data:a.txt+v2"

    # force=True ignores checkpoints
    summary4 = Runner(cfg, skip_download=True, force=True).run()
    assert all(s.cached == 0 for s in summary4.stages)


def test_missing_outputs_rerun_the_task(tmp_path: Path, steps_registry):
    cfg = base_config(tmp_path, [{"step": "make_copies", "inputs": [{"stage": "raw", "pattern": "*.txt"}]}])
    seed_raw(cfg, ["a.txt"])
    Runner(cfg, skip_download=True).run()
    (cfg.stage_dir("make_copies") / "a.copy.txt").unlink()
    summary = Runner(cfg, skip_download=True).run()
    assert summary.stages[0].success == 1  # checkpoint.require_outputs: record alone is not enough


def test_on_error_continue_records_and_fail_aborts(tmp_path: Path, steps_registry):
    cfg = base_config(tmp_path, [
        {"step": "boom", "inputs": [{"stage": "raw", "pattern": "*.txt"}], "on_error": "continue"},
    ])
    seed_raw(cfg, ["a.txt", "b.txt"])
    summary = Runner(cfg, skip_download=True).run()
    assert not summary.ok and summary.failed == 2
    assert steps_registry["boom"] == 2  # both tasks attempted
    recs = Runner(cfg, skip_download=True).state.records("boom")
    assert {r.status for r in recs} == {"failed"}
    assert all("kaboom" in (r.error or "") for r in recs)

    cfg2 = base_config(tmp_path / "x", [
        {"step": "boom", "inputs": [{"stage": "raw", "pattern": "*.txt"}], "on_error": "fail"},
    ])
    seed_raw(cfg2, ["a.txt", "b.txt"])
    with pytest.raises(RuntimeError, match="failed"):
        Runner(cfg2, skip_download=True).run()


def test_failed_tasks_rerun_next_time(tmp_path: Path, steps_registry):
    cfg = base_config(tmp_path, [
        {"step": "boom", "inputs": [{"stage": "raw", "pattern": "*.txt"}], "on_error": "continue"},
    ])
    seed_raw(cfg, ["a.txt"])
    Runner(cfg, skip_download=True).run()
    Runner(cfg, skip_download=True).run()
    assert steps_registry["boom"] == 2  # a failed record never counts as a checkpoint


def test_orphans_are_detected(tmp_path: Path, steps_registry):
    cfg = base_config(tmp_path, [{"step": "make_copies", "inputs": [{"stage": "raw", "pattern": "*.txt"}]}])
    seed_raw(cfg, ["a.txt"])
    runner = Runner(cfg, skip_download=True)
    runner.run()
    stray = cfg.stage_dir("make_copies") / "stale_from_last_year.txt"
    stray.write_text("boo")
    assert [p.name for p in runner.orphans(cfg.stage("make_copies"))] == ["stale_from_last_year.txt"]


def test_task_filter_selects_by_label(tmp_path: Path, steps_registry):
    cfg = base_config(tmp_path, [{"step": "make_copies", "inputs": [{"stage": "raw", "pattern": "*.txt"}]}])
    seed_raw(cfg, ["a.txt", "b.txt"])
    summary = Runner(cfg, skip_download=True, tasks=["a*"]).run()
    assert summary.stages[0].total == 1 and summary.stages[0].success == 1


def test_stage_selection_and_ordering(tmp_path: Path, steps_registry):
    cfg = base_config(tmp_path, [
        {"step": "combine_all", "variant": "late", "inputs": [{"stage": "make_copies", "pattern": "*.txt"}],
         "tags": ["qa"]},
        {"step": "make_copies", "inputs": [{"stage": "raw", "pattern": "*.txt"}]},
    ])
    ordered = [s.name for s in ordered_stages(cfg)]
    assert ordered == ["make_copies", "combine_all-late"]  # inputs imply the order, YAML order does not
    assert [s.name for s in select_stages(cfg, tags=["qa"])] == ["combine_all-late"]
    assert [s.name for s in select_stages(cfg, until="make_copies")] == ["make_copies"]
    assert [s.name for s in select_stages(cfg, start="combine_all-late")] == ["combine_all-late"]
    with pytest.raises(ValueError, match="unknown stage"):
        select_stages(cfg, only=["nope"])


def test_disabled_stages_are_skipped(tmp_path: Path, steps_registry):
    cfg = base_config(tmp_path, [
        {"step": "make_copies", "inputs": [{"stage": "raw", "pattern": "*.txt"}], "enabled": False},
    ])
    seed_raw(cfg, ["a.txt"])
    summary = Runner(cfg, skip_download=True).run()
    assert summary.stages == []


def test_debug_task_materialises_a_runnable_task(tmp_path: Path, steps_registry):
    cfg = base_config(tmp_path, [{"step": "make_copies", "inputs": [{"stage": "raw", "pattern": "*.txt"}]}])
    seed_raw(cfg, ["a.txt"])
    step, inputs, ctx, params = Runner(cfg, skip_download=True).debug_task("make_copies")
    outputs = step.run(inputs, ctx, **params)
    assert [o.name for o in outputs] == ["a.copy.txt"]
    assert Runner(cfg, skip_download=True).state.records("make_copies") == []  # nothing recorded


def test_run_writes_manifest_and_resolved_config(tmp_path: Path, steps_registry):
    cfg = base_config(tmp_path, [{"step": "make_copies", "inputs": [{"stage": "raw", "pattern": "*.txt"}]}])
    seed_raw(cfg, ["a.txt"])
    Runner(cfg, skip_download=True).run()
    manifest = json.loads((cfg.state_dir / "manifest.json").read_text())
    assert manifest["runs"][-1]["crds"]["pinned_context"] == "jwst_1364.pmap"
    assert (cfg.state_dir / "config.resolved.yaml").is_file()


# ------------------------------------------------------------------ state store


def test_state_store_roundtrip_and_atomicity(tmp_path: Path):
    store = StateStore(tmp_path / "state")
    rec = TaskRecord(task_id="t1", stage="s", step="x", outputs=[str(tmp_path / "gone.txt")], status="success")
    store.put(rec)
    assert store.get("s", "t1") is not None
    assert not store.is_complete("s", "t1")  # outputs are gone
    assert store.is_complete("s", "t1", require_outputs=False)
    (tmp_path / "gone.txt").write_text("x")
    assert store.is_complete("s", "t1")
    assert store.clear("s") == 1 and store.get("s", "t1") is None


def test_task_ids_are_stable_and_sensitive():
    base = dict(stage="s", step="x", inputs=["/a", "/b"], parameters={"k": 1},
                fingerprints={"/a": "1:2", "/b": "3:4"}, env_signature={"jwst": "3.0"})
    tid = make_task_id(**base)
    assert tid == make_task_id(**{**base, "inputs": ["/b", "/a"]})  # order-insensitive
    assert tid != make_task_id(**{**base, "parameters": {"k": 2}})
    assert tid != make_task_id(**{**base, "env_signature": {"jwst": "3.1"}})
    assert tid != make_task_id(**{**base, "fingerprints": {"/a": "9:9", "/b": "3:4"}})
