"""Config composition and validation: extends, interpolation, overrides, schema rules."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from jwstflow.config.loader import ConfigError, apply_overrides, deep_merge, load_config, load_raw
from jwstflow.config.schema import slugify


def write(path: Path, text: str) -> Path:
    path.write_text(textwrap.dedent(text))
    return path


BASE = """
version: 1
target: ESO-Ha 569
run: base_run
crds: {context: jwst_1364.pmap, prefetch: false}
parallel: {backend: serial, workers: 1}
workflow_graph: false
stages:
  - step: detector1
    inputs:
      - {stage: raw, pattern: "*_uncal.fits"}
    parameters:
      steps:
        jump: {expand_large_events: true}
  - step: spec3
    variant: pass1
    inputs:
      - {stage: calwebb_detector1, pattern: "*_rate.fits"}
    association: {mode: group, level: 3}
"""


def test_stage_names_and_levels_are_derived(tmp_path: Path):
    cfg = load_config(write(tmp_path / "w.yaml", BASE))
    assert [s.name for s in cfg.stages] == ["calwebb_detector1", "calwebb_spec3-pass1"]
    assert [s.level for s in cfg.stages] == [1, 3]
    assert cfg.stage_dir("calwebb_detector1").name == "calwebb_detector1"
    assert cfg.stage_dir("calwebb_detector1").parent.name == "stage1"
    assert cfg.target_slug == "eso-ha-569" and cfg.name == "eso-ha-569/base_run"


def test_extends_merges_stages_by_identity(tmp_path: Path):
    write(tmp_path / "base.yaml", BASE)
    child = write(tmp_path / "child.yaml", """
    extends: base.yaml
    run: child_run
    stages:
      - step: detector1
        parameters:
          steps:
            jump: {rejection_threshold: 5}
      - step: quicklook_image
        inputs:
          - {stage: calwebb_detector1, pattern: "*_rate.fits"}
    """)
    cfg = load_config(child)
    assert cfg.run == "child_run"
    det1 = cfg.stage("calwebb_detector1")
    # deep merge: the child's jump tweak lands next to the parent's, inputs survive
    assert det1.parameters["steps"]["jump"] == {"expand_large_events": True, "rejection_threshold": 5}
    assert det1.inputs and det1.inputs[0].pattern == "*_uncal.fits"
    assert [s.name for s in cfg.stages] == ["calwebb_detector1", "calwebb_spec3-pass1", "quicklook_image"]


def test_extends_cycles_are_rejected(tmp_path: Path):
    write(tmp_path / "a.yaml", "extends: b.yaml\n")
    write(tmp_path / "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ConfigError, match="circular"):
        load_raw(tmp_path / "a.yaml")


def test_interpolation_env_defaults_and_run_dir(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("JWF_TEST_UNSET", raising=False)
    monkeypatch.setenv("JWF_TEST_SET", "hello")
    cfg = load_config(write(tmp_path / "w.yaml", BASE + """
env:
  A: ${env:JWF_TEST_SET}
  B: ${env:JWF_TEST_UNSET,fallback}
  C: ${run_dir}/stage4/in_field_background
  D: ${target}
"""))
    assert cfg.env["A"] == "hello"
    assert cfg.env["B"] == "fallback"
    assert cfg.env["C"] == f"{cfg.run_dir}/stage4/in_field_background"
    assert cfg.env["D"] == "ESO-Ha 569"


def test_interpolation_unset_env_without_default_fails(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("JWF_TEST_UNSET", raising=False)
    with pytest.raises(ConfigError, match="JWF_TEST_UNSET"):
        load_config(write(tmp_path / "w.yaml", BASE + "env: {A: '${env:JWF_TEST_UNSET}'}\n"))


def test_overrides_address_stages_by_name_step_or_index(tmp_path: Path):
    cfg = load_config(write(tmp_path / "w.yaml", BASE), overrides=[
        "parallel.workers=3",
        "stages.detector1.parameters.steps.jump.rejection_threshold=9",
        "stages.calwebb_spec3-pass1.enabled=false",
    ])
    assert cfg.parallel.workers == 3
    assert cfg.stage("calwebb_detector1").parameters["steps"]["jump"]["rejection_threshold"] == 9
    assert cfg.stage("calwebb_spec3-pass1").enabled is False
    with pytest.raises(ConfigError, match="no list element"):
        apply_overrides({"stages": [{"step": "detector1"}]}, ["stages.nope.enabled=false"])


def test_unknown_keys_and_stages_are_rejected(tmp_path: Path):
    with pytest.raises(ConfigError, match="workersz"):
        load_config(write(tmp_path / "a.yaml", BASE.replace("workers:", "workersz:")))
    bad = BASE + """
  - step: quicklook_image
    inputs:
      - {stage: no_such_stage, pattern: "*.fits"}
"""
    with pytest.raises(ConfigError, match="unknown stage"):
        load_config(write(tmp_path / "b.yaml", bad))


def test_stage_name_cannot_be_set_in_yaml(tmp_path: Path):
    bad = BASE.replace("  - step: detector1", "  - step: detector1\n    name: my_name")
    with pytest.raises(ConfigError, match="derived from the step"):
        load_config(write(tmp_path / "w.yaml", bad))


def test_duplicate_stages_need_a_variant(tmp_path: Path):
    dup = BASE + """
  - step: spec3
    variant: pass1
    inputs:
      - {stage: calwebb_detector1, pattern: "*_rate.fits"}
    association: {mode: group, level: 3}
"""
    with pytest.raises(ConfigError, match="duplicate stage name"):
        load_config(write(tmp_path / "w.yaml", dup))


def test_nested_multiprocessing_is_refused(tmp_path: Path):
    bad = BASE.replace("parallel: {backend: serial, workers: 1}",
                       "parallel: {backend: process, workers: 4}")
    bad = bad.replace("jump: {expand_large_events: true}",
                      "ramp_fit: {maximum_cores: all}")
    with pytest.raises(ConfigError, match="forbids nesting"):
        load_config(write(tmp_path / "w.yaml", bad))
    # serial backend, explicit opt-in, or workers 1 are all fine
    ok = bad.replace("parallel: {backend: process, workers: 4}",
                     "parallel: {backend: process, workers: 4, allow_nested_multiprocessing: true}")
    load_config(write(tmp_path / "ok.yaml", ok))


def test_input_spec_needs_exactly_one_source(tmp_path: Path):
    bad = BASE.replace('{stage: raw, pattern: "*_uncal.fits"}',
                       '{stage: raw, path: /tmp/x, pattern: "*_uncal.fits"}')
    with pytest.raises(ConfigError, match="exactly one"):
        load_config(write(tmp_path / "w.yaml", bad))


def test_deep_merge_semantics():
    base = {"a": {"x": 1, "y": 2}, "list": [1, 2], "stages": [{"step": "detector1", "parameters": {"p": 1}}]}
    override = {"a": {"y": 3}, "list": [9], "stages": [{"step": "detector1", "parameters": {"q": 2}}]}
    merged = deep_merge(base, override)
    assert merged["a"] == {"x": 1, "y": 3}          # dicts merge
    assert merged["list"] == [9]                    # plain lists replace
    assert merged["stages"] == [{"step": "detector1", "parameters": {"p": 1, "q": 2}}]  # stages merge by identity


def test_slugify():
    assert slugify("ESO-Ha 569") == "eso-ha-569"
    assert slugify("  weird__Name!! ") == "weird-name"
