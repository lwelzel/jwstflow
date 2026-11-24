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
from jwstflow.steps import resolve_target
from jwstflow.testing import make_context, run_step, synthetic_cube, synthetic_x1d


def _x1d(path, scale: float = 1.0):
    wave = np.linspace(2.9, 5.2, 1500)
    flux = scale * (1e-3 + 4e-3 * np.exp(-0.5 * ((wave - 4.0523) / 0.002) ** 2))
    return synthetic_x1d(path, wave=wave, flux=flux, instrument="NIRSPEC")


def test_stock_steps_resolve_by_name_and_by_path():
    assert resolve_target("plot_snr").__name__ == "PlotSnr"
    assert resolve_target("jwstflow.qa.steps.cubes:QaCubeSummary") is QaCubeSummary


def test_spectrum_steps_draw_annotated_spectrum_like_figures(tmp_path):
    spectrum = _x1d(tmp_path / "t_s1d.fits")
    png, detected = run_step(PlotSpectrum, [spectrum], tmp_path, stage="plot_spectrum", params={"features": "all"})
    assert png.name == "t_s1d.png" and png.stat().st_size > 20_000
    assert detected.name == "t_s1d_detected.png"                     # with features: only the lines found, too
    (alone,) = run_step(PlotSpectrum, [spectrum], tmp_path, stage="plot_spectrum_one",
                        params={"features": "all", "detected": False})
    (bare,) = run_step(PlotSpectrum, [spectrum], tmp_path, stage="plot_spectrum_bare",   # no features, one figure
                       params={"features": None})
    default = run_step(PlotSpectrum, [spectrum], tmp_path, stage="plot_spectrum_default")  # annotated by default
    assert [p.name for p in default] == ["t_s1d.png", "t_s1d_detected.png"]
    assert alone.name == bare.name == "t_s1d.png"
    (snr,) = run_step(PlotSnr, [spectrum], tmp_path, stage="plot_snr")                  # features annotated by default
    (plain,) = run_step(PlotSnr, [spectrum], tmp_path, stage="plot_snr_plain", params={"features": None})
    assert snr.name == "t_s1d_snr.png" and snr.stat().st_size > plain.stat().st_size


def _apertures(path):
    """An extraction with named apertures: the disk, a tighter contour and a faint outflow."""
    from astropy.io import fits

    _x1d(path)
    with fits.open(path) as hdul:
        disk = hdul["EXTRACT1D"].copy()
        disk.header["APERNAME"] = "disk"
        tight, outflow = disk.copy(), disk.copy()
        tight.header["APERNAME"] = "disk_eta0.5"
        outflow.header["APERNAME"] = "outflow1"
        outflow.data["FLUX"] = outflow.data["FLUX"] * 1e-3
        fits.HDUList([hdul[0].copy(), outflow, disk, tight]).writeto(path, overwrite=True)
    return path


def test_spectrum_figure_draws_the_target_aperture_and_the_apertures_figure_all(tmp_path):
    from astropy.io import fits

    from jwstflow import qa
    from jwstflow.qa.steps import PlotSpectrumApertures

    path = _apertures(tmp_path / "e_s1d.fits")
    with fits.open(path) as hdul:
        tables = [h for h in hdul if h.name == "EXTRACT1D"]
        assert [h.header["APERNAME"] for h in qa.select_spectra(tables)] == ["disk"]        # by name, not position
        assert [h.header["APERNAME"] for h in qa.select_spectra(tables, "nothere")] == ["outflow1"]   # else the first
        assert len(qa.select_spectra(tables, None)) == 3
        unnamed = [h.copy() for h in tables]
        for h in unnamed:
            del h.header["APERNAME"]
        assert len(qa.select_spectra(unnamed)) == 3                                          # slits / sources: all
        assert qa.spectrum_name(tables[1], 1) == "disk"
    figures = run_step(PlotSpectrum, [path], tmp_path, stage="plot_spectrum", params={"features": None})
    assert [p.name for p in figures] == ["e_s1d.png"]
    (apertures,) = run_step(PlotSpectrumApertures, [path], tmp_path, stage="plot_spectrum_apertures")
    assert apertures.name == "e_s1d_apertures.png" and apertures.stat().st_size > 20_000


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
