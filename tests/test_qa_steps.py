"""The QA steps shipped with jwstflow, run on synthetic products."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from jwstflow.qa.steps import (
    PlotSnr,
    PlotSpectrum,
    PlotSpectrumStages,
    QaCubeSlices,
    QaCubeSummary,
    QuicklookImage,
)
from jwstflow.steps.base import resolve_target
from jwstflow.testing import make_context, run_step, synthetic_cube, synthetic_x1d


def _x1d(path, scale: float = 1.0):
    wave = np.linspace(2.9, 5.2, 1500)
    flux = scale * (1e-3 + 4e-3 * np.exp(-0.5 * ((wave - 4.0523) / 0.002) ** 2))
    return synthetic_x1d(path, wave=wave, flux=flux, instrument="NIRSPEC")


def test_stock_steps_resolve_from_both_addresses():
    assert resolve_target("plot_snr").__name__ == "PlotSnr"
    assert resolve_target("jwstflow.qa.steps.cubes:QaCubeSummary") is QaCubeSummary
    assert resolve_target("jwstflow.contrib.qa:PlotSpectrum") is PlotSpectrum        # the address of older workflows


def test_spectrum_steps_draw_annotated_spectrum_like_figures(tmp_path):
    spectrum = _x1d(tmp_path / "t_s1d.fits")
    (png,) = run_step(PlotSpectrum, [spectrum], tmp_path, stage="plot_spectrum", params={"features": "all"})
    assert png.name == "t_s1d.png" and png.stat().st_size > 20_000
    (snr,) = run_step(PlotSnr, [spectrum], tmp_path, stage="plot_snr")                  # features annotated by default
    (plain,) = run_step(PlotSnr, [spectrum], tmp_path, stage="plot_snr_plain", params={"features": None})
    assert snr.name == "t_s1d_snr.png" and snr.stat().st_size > plain.stat().st_size


def test_stage_comparison_finds_the_ancestor_by_name(tmp_path):
    ctx = make_context(tmp_path, stage="plot_spectrum_stages")
    early, late = ctx.root / "stage4" / "extract", ctx.root / "stage4" / "clean"
    for directory in (early, late):
        directory.mkdir(parents=True)
    _x1d(early / "t_s1d.fits", scale=1.05)
    final = _x1d(late / "t_clean_s1d.fits")
    ctx.stage_dirs.update({"extract": early, "clean": late})
    (png,) = run_step(PlotSpectrumStages, [final], tmp_path, ctx=ctx, params={"stages": ["extract"]})
    assert png.name == "t_clean_s1d_stages.png"


def test_cube_steps_draw_frames_with_both_sets_of_axes(tmp_path):
    cube = synthetic_cube(tmp_path / "t_s3d.fits", nwave=40, size=24)
    (summary,) = run_step(QaCubeSummary, [cube], tmp_path, stage="qa_cube_summary")
    (slices,) = run_step(QaCubeSlices, [cube], tmp_path, stage="qa_cube_slices", params={"n_slices": 6, "ncols": 3})
    (quick,) = run_step(QuicklookImage, [cube], tmp_path, stage="quicklook_image")
    assert (summary.name, slices.name, quick.name) == ("t_s3d_summary.png", "t_s3d_slices.png", "t_s3d.png")
    assert all(p.stat().st_size > 15_000 for p in (summary, slices, quick))
