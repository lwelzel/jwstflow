"""Figures come from QA steps only (docs/qa_figures.md, rule 3).

A step whose level is not "qa" that returns a figure, or leaves a new one in its
output directory without returning it, fails -- and its files are removed, as for
every other contract violation. QA steps write figures freely; a figure that was in
a data stage's directory before the step ran is left to the orphan report.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jwstflow import RunContext, Step
from jwstflow.config.loader import config_from_dict
from jwstflow.engine.runner import Runner
from jwstflow.steps import FIGURE_SUFFIXES, register_step
from jwstflow.steps.registry import _REGISTRY
from jwstflow.testing import make_context, run_step

RULE = "figures come from QA steps only"


def _product(ctx: RunContext, name: str = "product.ecsv") -> Path:
    path = ctx.output_dir / name
    path.write_text("# a product\n")
    return path


def _figure(ctx: RunContext, name: str) -> Path:
    path = ctx.output_dir / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return path


class ReturnsAFigure(Step):
    level = 4
    batch = "all"

    def run(self, inputs, ctx, **_):
        return [_product(ctx), _figure(ctx, "product_diagnostic.png")]


class WritesAFigureItDoesNotReturn(Step):
    level = 4
    batch = "all"

    def run(self, inputs, ctx, **_):
        _figure(ctx, "product_diagnostic.pdf")
        return [_product(ctx)]


class WritesAProduct(Step):
    level = 4
    batch = "all"

    def run(self, inputs, ctx, **_):
        return [_product(ctx)]


class DrawsAFigure(Step):
    level = "qa"
    batch = "all"

    def run(self, inputs, ctx, **_):
        return [_figure(ctx, "product.png")]


@pytest.fixture()
def spectrum(tmp_path: Path) -> Path:
    path = tmp_path / "input_s1dcomb.ecsv"
    path.write_text("# an input\n")
    return path


def test_the_figure_suffixes_cover_raster_and_vector_formats():
    assert {".png", ".jpg", ".pdf", ".svg", ".eps"} <= FIGURE_SUFFIXES
    assert not {".fits", ".ecsv", ".asdf", ".json", ".txt"} & FIGURE_SUFFIXES


def test_a_data_step_that_returns_a_figure_fails_and_its_files_are_removed(tmp_path, spectrum):
    ctx = make_context(tmp_path / "work")
    with pytest.raises(ValueError, match=RULE) as err:
        run_step(ReturnsAFigure, [spectrum], tmp_path, ctx=ctx)
    assert "ReturnsAFigure is a level-4 step and returned a figure (product_diagnostic.png)" in str(err.value)
    assert sorted(ctx.output_dir.iterdir()) == []


def test_a_data_step_that_writes_a_figure_without_returning_it_fails(tmp_path, spectrum):
    ctx = make_context(tmp_path / "work")
    with pytest.raises(ValueError, match=RULE) as err:
        run_step(WritesAFigureItDoesNotReturn, [spectrum], tmp_path, ctx=ctx)
    assert "without returning them (product_diagnostic.pdf)" in str(err.value)
    assert sorted(ctx.output_dir.iterdir()) == []


def test_a_function_step_is_a_data_step_unless_it_says_otherwise(tmp_path, spectrum):
    def plot_it(inputs, ctx, **_):
        return [_figure(ctx, "plot.svg")]

    with pytest.raises(ValueError, match=RULE):
        run_step(plot_it, [spectrum], tmp_path / "data")

    plot_it.level = "qa"
    (out,) = run_step(plot_it, [spectrum], tmp_path / "qa")
    assert out.name == "plot.svg" and out.is_file()


def test_a_qa_step_returns_its_figures(tmp_path, spectrum):
    (out,) = run_step(DrawsAFigure, [spectrum], tmp_path)
    assert out.name == "product.png" and out.is_file()


def test_a_figure_already_in_a_data_stage_is_left_to_the_orphan_report(tmp_path, spectrum):
    ctx = make_context(tmp_path / "work")
    old = _figure(ctx, "left_by_an_old_version.png")
    (out,) = run_step(WritesAProduct, [spectrum], tmp_path, ctx=ctx)
    assert out.name == "product.ecsv" and old.is_file()


def test_the_runner_fails_the_task_and_keeps_the_stage_directory_clean(tmp_path):
    @register_step("draws_in_its_product_dir")
    def draws_in_its_product_dir(inputs, ctx, **_):
        return [_product(ctx, f"{inputs[0].stem}.ecsv"), _figure(ctx, f"{inputs[0].stem}.png")]

    @register_step("draws_in_qa")
    def draws_in_qa(inputs, ctx, **_):
        return [_figure(ctx, f"{inputs[0].stem}.png")]

    draws_in_qa.level = "qa"
    try:
        cfg = config_from_dict({
            "target": "T", "run": "r1", "root": str(tmp_path / "target" / "r1"),
            "crds": {"context": "jwst_1364.pmap", "prefetch": False},
            "parallel": {"backend": "serial", "workers": 1}, "workflow_graph": False,
            "stages": [
                {"step": "draws_in_its_product_dir", "inputs": [{"stage": "raw", "pattern": "*.txt"}],
                 "on_error": "continue"},
                {"step": "draws_in_qa", "inputs": [{"stage": "raw", "pattern": "*.txt"}]},
            ],
        }, base_dir=tmp_path)
        cfg.raw_dir.mkdir(parents=True, exist_ok=True)
        (cfg.raw_dir / "a.txt").write_text("a")

        summary = Runner(cfg, skip_download=True).run()
        assert summary.failed == 1
        (record,) = Runner(cfg, skip_download=True).state.records("draws_in_its_product_dir")
        assert record.status == "failed" and RULE in (record.error or "")
        assert sorted(cfg.stage_dir("draws_in_its_product_dir").iterdir()) == []
        assert [p.name for p in cfg.stage_dir("draws_in_qa").iterdir()] == ["a.png"]
        assert cfg.stage_dir("draws_in_qa").parent.name == "qa"
    finally:
        for name in ("draws_in_its_product_dir", "draws_in_qa"):
            _REGISTRY.pop(name, None)
