"""Tests of the generic segment stitcher (jwstflow.spectra.stitching.StitchSegments)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jwstflow.spectra.stitching import StitchSegments
from jwstflow.testing import check_step, make_context, run_step, synthetic_x1d


def two_segments(tmp_path: Path) -> list[Path]:
    blue = synthetic_x1d(tmp_path / "jw001_nirspec_g235h-f170lp_s1d.fits",
                         wave=np.linspace(1.0, 2.0, 200), flux=np.full(200, 1.0),
                         instrument="NIRSPEC", GRATING="G235H", FILTER="F170LP")
    red = synthetic_x1d(tmp_path / "jw001_nirspec_g395h-f290lp_s1d.fits",
                        wave=np.linspace(1.8, 3.0, 200), flux=np.full(200, 2.0),
                        instrument="NIRSPEC", GRATING="G395H", FILTER="F290LP")
    return [blue, red]


def test_declaration_is_valid():
    _, problems = check_step(StitchSegments)
    assert problems == []


def test_splice_at_overlap_midpoint(tmp_path: Path):
    outputs = run_step(StitchSegments, two_segments(tmp_path), tmp_path)
    ecsv = [o for o in outputs if o.suffix == ".ecsv"]
    assert len(ecsv) == 1
    assert ecsv[0].name == "jw001_nirspec_s1dcomb.ecsv"  # common prefix, partial token dropped
    from astropy.table import Table

    tab = Table.read(ecsv[0])
    w, f = np.asarray(tab["WAVELENGTH"]), np.asarray(tab["FLUX"])
    assert np.all(f[w < 1.9] == 1.0) and np.all(f[w > 1.9] == 2.0)  # midpoint of the 1.8-2.0 overlap
    assert tab.meta["segments"] == ["g235h-f170lp", "g395h-f290lp"]
    assert tab.meta["neighbour_ratios"][0] == pytest.approx(2.0)
    assert tab.meta["scales_applied"] == [1.0, 1.0]
    # the measurement region: the ratio_window_frac window around the 1.9 crossover
    assert tab.meta["ratio_window_frac"] == pytest.approx(0.05)
    lo, hi = tab.meta["ratio_windows_um"][0]
    assert lo == pytest.approx(1.9 * (1 - 0.025), abs=1e-6) and hi == pytest.approx(1.9 * (1 + 0.025), abs=1e-6)
    assert tab.meta["neighbour_ratio_errors"][0] > 0
    assert tab.meta["scale_errors"] == [0.0, 0.0]            # nothing rescaled, nothing propagated


def test_rescale_chains_onto_the_reference(tmp_path: Path):
    outputs = run_step(StitchSegments, two_segments(tmp_path), tmp_path,
                       params={"rescale": True, "crossovers": [1.9]})
    from astropy.table import Table

    tab = Table.read([o for o in outputs if o.suffix == ".ecsv"][0])
    # default reference is the reddest segment: the blue one is scaled up by the ratio 2
    assert tab.meta["scales_applied"] == [pytest.approx(2.0), 1.0]
    assert np.all(np.asarray(tab["FLUX"]) == pytest.approx(2.0))


def test_overlap_ratio_ignores_noise_crossing_zero(tmp_path: Path):
    # a noisy but significantly positive overlap: the ratio of windowed medians
    # must come out right even when single samples of `a` sit at or below zero
    # (ratio_window_frac chosen to span the whole 1.6-2.0 overlap, so the
    # median sees every overlap sample)
    rng = np.random.default_rng(5)
    n = 400
    blue = synthetic_x1d(tmp_path / "jw001_nirspec_g235h-f170lp_s1d.fits",
                         wave=np.linspace(1.0, 2.0, n), flux=0.10 + 0.08 * rng.standard_normal(n),
                         instrument="NIRSPEC", GRATING="G235H", FILTER="F170LP")
    red = synthetic_x1d(tmp_path / "jw001_nirspec_g395h-f290lp_s1d.fits",
                        wave=np.linspace(1.6, 3.0, n), flux=0.20 + 0.08 * rng.standard_normal(n),
                        instrument="NIRSPEC", GRATING="G395H", FILTER="F290LP")
    outputs = run_step(StitchSegments, [blue, red], tmp_path,
                       params={"rescale": True, "ratio_window_frac": 0.25})
    from astropy.table import Table

    tab = Table.read([o for o in outputs if o.suffix == ".ecsv"][0])
    assert tab.meta["neighbour_ratios"][0] == pytest.approx(2.0, rel=0.25)
    assert tab.meta["scales_applied"][0] == pytest.approx(2.0, rel=0.25)


def test_ratio_is_measured_around_the_crossover_not_the_whole_overlap(tmp_path: Path):
    # segment b carries a strong calibration slope across the wide 1.5-2.0
    # overlap: the multiply factor must come from the window around the
    # crossover (where the splice happens), not from the whole overlap,
    # whose median would land on the overlap centre instead
    n = 600
    wa, wb = np.linspace(1.0, 2.0, n), np.linspace(1.5, 3.0, n)
    blue = synthetic_x1d(tmp_path / "jw001_nirspec_g235h-f170lp_s1d.fits",
                         wave=wa, flux=np.full(n, 1.0),
                         instrument="NIRSPEC", GRATING="G235H", FILTER="F170LP")
    red = synthetic_x1d(tmp_path / "jw001_nirspec_g395h-f290lp_s1d.fits",
                        wave=wb, flux=2.0 + 3.0 * (wb - 1.6),
                        instrument="NIRSPEC", GRATING="G395H", FILTER="F290LP")
    outputs = run_step(StitchSegments, [blue, red], tmp_path,
                       params={"rescale": True, "crossovers": [1.6], "reference": "shortest"})
    from astropy.table import Table

    tab = Table.read([o for o in outputs if o.suffix == ".ecsv"][0])
    # around 1.6 um the true ratio is 2.0; the whole-overlap median would be
    # 2 + 3 * (1.75 - 1.6) = 2.45
    assert tab.meta["neighbour_ratios"][0] == pytest.approx(2.0, rel=0.05)
    lo, hi = tab.meta["ratio_windows_um"][0]
    assert lo == pytest.approx(1.6 * (1 - 0.025), abs=1e-6)
    assert hi == pytest.approx(1.6 * (1 + 0.025), abs=1e-6)


def test_sparse_ratio_window_widens_to_the_whole_overlap(tmp_path: Path, caplog):
    # 30 samples over 1.0-2.0: the default 5% window around 1.9 holds ~1-2
    # samples (< min_overlap_points), so the measurement falls back to the
    # whole overlap instead of failing
    n = 30
    blue = synthetic_x1d(tmp_path / "jw001_nirspec_g235h-f170lp_s1d.fits",
                         wave=np.linspace(1.0, 2.0, n), flux=np.full(n, 1.0),
                         instrument="NIRSPEC", GRATING="G235H", FILTER="F170LP")
    red = synthetic_x1d(tmp_path / "jw001_nirspec_g395h-f290lp_s1d.fits",
                        wave=np.linspace(1.8, 3.0, n), flux=np.full(n, 2.0),
                        instrument="NIRSPEC", GRATING="G395H", FILTER="F290LP")
    with caplog.at_level("WARNING"):
        outputs = run_step(StitchSegments, [blue, red], tmp_path, params={"rescale": True})
    from astropy.table import Table

    tab = Table.read([o for o in outputs if o.suffix == ".ecsv"][0])
    assert "widening to the whole overlap" in caplog.text
    assert tab.meta["neighbour_ratios"][0] == pytest.approx(2.0)
    lo, hi = tab.meta["ratio_windows_um"][0]
    assert (lo, hi) == (pytest.approx(1.8), pytest.approx(2.0))  # the whole overlap


def error_segments(tmp_path: Path, *, err: float = 0.05) -> list[Path]:
    """Two constant segments (fluxes 1 and 2) with a known per-sample flux error."""
    from jwstflow.spectra import Spectrum1D, write_x1d

    out = []
    for name, lo, hi, flux, grating, filt in (
            ("jw001_nirspec_g235h-f170lp_s1d.fits", 1.0, 2.0, 1.0, "G235H", "F170LP"),
            ("jw001_nirspec_g395h-f290lp_s1d.fits", 1.8, 3.0, 2.0, "G395H", "F290LP")):
        w = np.linspace(lo, hi, 200)
        out.append(write_x1d(tmp_path / name, Spectrum1D(w, np.full(200, flux), np.full(200, err)),
                             header={"INSTRUME": "NIRSPEC", "GRATING": grating, "FILTER": filt}))
    return out


def test_scaling_uncertainty_comes_from_the_window_flux_errors(tmp_path: Path):
    from astropy.table import Table

    outputs = run_step(StitchSegments, error_segments(tmp_path), tmp_path,
                       params={"rescale": True, "reference": "longest"})
    tab = Table.read([o for o in outputs if o.suffix == ".ecsv"][0])
    ratio = tab.meta["neighbour_ratios"][0]
    ratio_err = tab.meta["neighbour_ratio_errors"][0]
    assert ratio == pytest.approx(2.0)
    # the 5% window around the 1.9 crossover holds the samples in 1.8525-1.9475:
    # 200 samples over 1.0-2.0 -> 19 of them; each median carries
    # 1.2533 * 0.05 / sqrt(19), and the relative errors add in quadrature
    n_win = int(np.sum((np.linspace(1.0, 2.0, 200) >= 1.9 * 0.975)
                       & (np.linspace(1.0, 2.0, 200) <= 1.9 * 1.025)))
    med_err = 1.2533 * 0.05 / np.sqrt(n_win)
    assert ratio_err == pytest.approx(2.0 * np.hypot(med_err / 1.0, med_err / 2.0), rel=1e-6)
    # the blue segment was multiplied by 2 +- ratio_err: scale error recorded ...
    assert tab.meta["scales_applied"] == [pytest.approx(2.0), 1.0]
    assert tab.meta["scale_errors"] == [pytest.approx(ratio_err), 0.0]
    # ... and folded into the stitched FLUX_ERROR (in quadrature with the
    # rescaled per-sample error), so later stitches inherit it
    seg = np.asarray(tab["SEGMENT"], dtype=str)
    err = np.asarray(tab["FLUX_ERROR"], dtype=float)
    assert err[seg == "g235h-f170lp"] == pytest.approx(np.hypot(0.05 * 2.0, 1.0 * ratio_err))
    assert err[seg == "g395h-f290lp"] == pytest.approx(0.05)


def test_scale_errors_chain_in_quadrature(tmp_path: Path):
    # three constant segments rescaled onto the bluest: the last segment's
    # scale crosses both measured links, so its relative uncertainty is the
    # quadrature sum of both relative ratio errors
    from astropy.table import Table

    files = [synthetic_x1d(tmp_path / f"jw001_miri_ch1-{band}_s1d.fits",
                           wave=np.linspace(lo, hi, 300), flux=np.full(300, flux),
                           CHANNEL="1", BAND=band.upper())
             for band, lo, hi, flux in (("short", 4.9, 5.8, 1.0), ("medium", 5.65, 6.7, 2.0),
                                        ("long", 6.5, 7.7, 4.0))]
    outputs = run_step(StitchSegments, files, tmp_path,
                       params={"rescale": True, "reference": "shortest"})
    tab = Table.read([o for o in outputs if o.suffix == ".ecsv"][0])
    ratios = tab.meta["neighbour_ratios"]
    errors = tab.meta["neighbour_ratio_errors"]
    scales = tab.meta["scales_applied"]
    scale_errors = tab.meta["scale_errors"]
    assert scales == [1.0, pytest.approx(0.5), pytest.approx(0.25)]
    assert scale_errors[0] == 0.0
    assert scale_errors[1] == pytest.approx(scales[1] * errors[0] / ratios[0], rel=1e-6)
    expected_rel = np.hypot(errors[0] / ratios[0], errors[1] / ratios[1])
    assert scale_errors[2] == pytest.approx(scales[2] * expected_rel, rel=1e-6)


def test_near_zero_overlap_is_not_rescaled(tmp_path: Path, caplog):
    # the chain regression: a segment whose flux is consistent with zero (an
    # aperture that missed the source) cannot anchor a multiplicative
    # rescaling -- its links stay at 1 instead of amplifying everything redder
    # by huge (or negative) factors
    rng = np.random.default_rng(6)
    n = 300
    seg = [synthetic_x1d(tmp_path / "jw001_miri_ch1-short_s1d.fits",
                         wave=np.linspace(4.9, 5.8, n), flux=np.full(n, 1.0),
                         CHANNEL="1", BAND="SHORT"),
           synthetic_x1d(tmp_path / "jw001_miri_ch1-medium_s1d.fits",
                         wave=np.linspace(5.65, 6.7, n), flux=0.002 * rng.standard_normal(n),
                         CHANNEL="1", BAND="MEDIUM"),
           synthetic_x1d(tmp_path / "jw001_miri_ch1-long_s1d.fits",
                         wave=np.linspace(6.5, 7.7, n), flux=np.full(n, 4.0),
                         CHANNEL="1", BAND="LONG")]
    with caplog.at_level("WARNING"):
        outputs = run_step(StitchSegments, seg, tmp_path,
                           params={"rescale": True, "reference": "shortest"})
    from astropy.table import Table

    tab = Table.read([o for o in outputs if o.suffix == ".ecsv"][0])
    assert tab.meta["neighbour_ratios"] == [None, None]      # both unmeasurable, recorded as such
    assert tab.meta["scales_applied"] == [1.0, 1.0, 1.0]     # nothing blown up
    assert "consistent with zero" in caplog.text
    flux = np.asarray(tab["FLUX"])
    assert np.nanmax(np.abs(flux)) == pytest.approx(4.0)     # the red segment kept its own scale
    outputs = run_step(StitchSegments, seg, tmp_path, stage="unguarded",
                       params={"rescale": True, "reference": "shortest", "min_overlap_snr": 0.0})
    tab = Table.read([o for o in outputs if o.suffix == ".ecsv"][0])
    assert tab.meta["neighbour_ratios"][0] is not None       # 0 disables the guard


def test_single_segment_is_rejected(tmp_path: Path):
    (only,) = two_segments(tmp_path)[:1]
    with pytest.raises(ValueError, match="at least two"):
        run_step(StitchSegments, [only], tmp_path)


def test_stitching_never_plots_and_plot_stitch_does(tmp_path: Path):
    pytest.importorskip("matplotlib")
    from jwstflow.qa.steps import PlotStitch

    segments = two_segments(tmp_path)
    outputs = run_step(StitchSegments, segments, tmp_path)
    assert [o.suffix for o in outputs] == [".ecsv"]  # the data step writes no figure
    ctx = make_context(tmp_path, stage="plot_stitch")
    ctx.stage_dirs["extract"] = segments[0].parent   # where the segment files live
    pngs = run_step(PlotStitch, [o for o in outputs if o.suffix == ".ecsv"], tmp_path, ctx=ctx)
    # no rescaling happened -> the pre-rescale figure would repeat this one, so only one PNG
    assert [o.suffix for o in pngs] == [".png"]


def test_plot_stitch_also_draws_the_segments_before_rescaling(tmp_path: Path):
    pytest.importorskip("matplotlib")
    from jwstflow.qa.steps import PlotStitch

    segments = two_segments(tmp_path)
    outputs = run_step(StitchSegments, segments, tmp_path, params={"rescale": True})
    ctx = make_context(tmp_path, stage="plot_stitch")
    ctx.stage_dirs["extract"] = segments[0].parent
    pngs = run_step(PlotStitch, [o for o in outputs if o.suffix == ".ecsv"], tmp_path, ctx=ctx)
    assert sorted(p.name for p in pngs) == ["jw001_nirspec_s1dcomb.png",
                                            "jw001_nirspec_s1dcomb_unscaled.png"]
    pngs = run_step(PlotStitch, [o for o in outputs if o.suffix == ".ecsv"], tmp_path, ctx=ctx,
                    params={"unscaled": False})
    assert [p.name for p in pngs] == ["jw001_nirspec_s1dcomb.png"]
    # with features: every line labelled, only the lines found, and the series as combs
    pngs = run_step(PlotStitch, [o for o in outputs if o.suffix == ".ecsv"], tmp_path, ctx=ctx,
                    params={"unscaled": False, "features": "all"})
    assert [p.name for p in pngs] == ["jw001_nirspec_s1dcomb.png", "jw001_nirspec_s1dcomb_detected.png",
                                      "jw001_nirspec_s1dcomb_series.png"]


def test_plot_stitch_overlaps_draws_one_panel_per_pair(tmp_path: Path):
    pytest.importorskip("matplotlib")
    from jwstflow.qa.steps import PlotStitchOverlaps

    files = [synthetic_x1d(tmp_path / f"jw001_miri_ch1-{band}_s1d.fits",
                           wave=np.linspace(lo, hi, 300), flux=np.full(300, flux),
                           CHANNEL="1", BAND=band.upper())
             for band, lo, hi, flux in (("short", 4.9, 5.8, 1.0), ("medium", 5.65, 6.7, 2.0),
                                        ("long", 6.5, 7.7, 4.0))]
    outputs = run_step(StitchSegments, files, tmp_path, params={"rescale": True})
    ctx = make_context(tmp_path, stage="plot_stitch_overlaps")
    ctx.stage_dirs["extract"] = files[0].parent
    pngs = run_step(PlotStitchOverlaps, [o for o in outputs if o.suffix == ".ecsv"], tmp_path, ctx=ctx)
    assert [p.name for p in pngs] == ["jw001_miri_s1dcomb_overlaps.png"]


def test_plot_stitch_overlaps_tolerates_missing_segments_and_gaps(tmp_path: Path, caplog):
    pytest.importorskip("matplotlib")
    from jwstflow.qa.steps import PlotStitchOverlaps

    # a gap between the segments (no overlap: the ratio is unmeasurable) ...
    files = [synthetic_x1d(tmp_path / "jw001_miri_ch1-short_s1d.fits",
                           wave=np.linspace(4.9, 5.5, 200), flux=np.full(200, 1.0),
                           CHANNEL="1", BAND="SHORT"),
             synthetic_x1d(tmp_path / "jw001_miri_ch1-long_s1d.fits",
                           wave=np.linspace(6.5, 7.7, 200), flux=np.full(200, 4.0),
                           CHANNEL="1", BAND="LONG")]
    outputs = run_step(StitchSegments, files, tmp_path)
    # ... and one segment file gone by the time QA runs
    files[0].unlink()
    ctx = make_context(tmp_path, stage="plot_stitch_overlaps")
    ctx.stage_dirs["extract"] = tmp_path
    with caplog.at_level("WARNING"):
        pngs = run_step(PlotStitchOverlaps, [o for o in outputs if o.suffix == ".ecsv"], tmp_path, ctx=ctx)
    assert len(pngs) == 1 and pngs[0].name.endswith("_overlaps.png")
    assert "not found in any stage directory" in caplog.text


def segments_with_background(tmp_path: Path) -> list[Path]:
    """Two overlapping segments whose extraction recorded a background."""
    from jwstflow.spectra import Spectrum1D, write_x1d

    out = []
    for name, lo, hi, flux, bkg, grating, filt in (
            ("jw001_nirspec_g235h-f170lp_s1d.fits", 1.0, 2.0, 1.0, 0.5, "G235H", "F170LP"),
            ("jw001_nirspec_g395h-f290lp_s1d.fits", 1.8, 3.0, 1.0, 0.7, "G395H", "F290LP")):
        w = np.linspace(lo, hi, 200)
        out.append(write_x1d(
            tmp_path / name, Spectrum1D(w, np.full(200, flux), np.full(200, 0.01)),
            header={"INSTRUME": "NIRSPEC", "GRATING": grating, "FILTER": filt,
                    "SKYSUB": (True, "background subtracted"), "TARGPROP": "TEST"},
            columns={"background": np.full(200, bkg), "bkgd_error": np.full(200, 0.02)}))
    return out


def test_plot_stitch_background_reassembles_the_background(tmp_path: Path):
    pytest.importorskip("matplotlib")
    from jwstflow.qa.steps import PlotStitchBackground

    segments = segments_with_background(tmp_path)
    outputs = run_step(StitchSegments, segments, tmp_path, params={"rescale": True})
    ctx = make_context(tmp_path, stage="plot_stitch_background")
    ctx.stage_dirs["segments"] = tmp_path
    pngs = run_step(PlotStitchBackground, [o for o in outputs if o.suffix == ".ecsv"], tmp_path,
                    ctx=ctx, params={"segments_stage": "segments"})
    assert [p.name for p in pngs] == ["jw001_nirspec_s1dcomb_bkgcomp.png"]


def test_plot_stitch_background_tolerates_missing_records(tmp_path: Path, caplog):
    """A segment without a background record leaves a gap and a note, never a crash."""
    pytest.importorskip("matplotlib")
    from jwstflow.qa.steps import PlotStitchBackground

    segments = segments_with_background(tmp_path)
    # strip the record off the red segment: without SKYSUB the columns mean nothing
    from astropy.io import fits

    with fits.open(segments[1], mode="update") as hdul:
        del hdul[0].header["SKYSUB"]
    outputs = run_step(StitchSegments, segments, tmp_path)
    ctx = make_context(tmp_path, stage="plot_stitch_background")
    ctx.stage_dirs["segments"] = tmp_path
    pngs = run_step(PlotStitchBackground, [o for o in outputs if o.suffix == ".ecsv"], tmp_path,
                    ctx=ctx, params={"segments_stage": "segments"})
    assert len(pngs) == 1 and pngs[0].name.endswith("_bkgcomp.png")


def test_flagged_samples_are_not_spliced(tmp_path: Path):
    """Samples with DQ != 0 (e.g. FLUX_ESTIMATED planes at a detector gap) stay out of the product and the ratio."""
    from astropy.table import Table

    from jwstflow.spectra import Spectrum1D, write_x1d

    w = np.linspace(1.0, 2.0, 200)
    f = np.full(200, 1.0)
    dq = np.zeros(200, dtype=np.uint32)
    bad = ((w > 1.40) & (w < 1.45)) | ((w > 1.88) & (w < 1.92))   # one block inside, one in the ratio window
    f[bad], dq[bad] = 100.0, 2**28
    blue = write_x1d(tmp_path / "jw001_nirspec_g235h-f170lp_s1d.fits", Spectrum1D(w, f, np.full(200, 0.05), dq),
                     header={"INSTRUME": "NIRSPEC", "GRATING": "G235H", "FILTER": "F170LP"})
    red = synthetic_x1d(tmp_path / "jw001_nirspec_g395h-f290lp_s1d.fits", wave=np.linspace(1.8, 3.0, 200),
                        flux=np.full(200, 2.0), instrument="NIRSPEC", GRATING="G395H", FILTER="F290LP")
    out = run_step(StitchSegments, [blue, red], tmp_path, params={"crossovers": [1.9]})
    tab = Table.read([o for o in out if o.suffix == ".ecsv"][0])
    assert np.nanmax(np.asarray(tab["FLUX"])) == pytest.approx(2.0)          # no flagged value spliced
    assert not np.any((np.asarray(tab["WAVELENGTH"]) > 1.40) & (np.asarray(tab["WAVELENGTH"]) < 1.45))
    assert tab.meta["neighbour_ratios"][0] == pytest.approx(2.0)             # ratio from good samples only


def test_additive_combine_chains_offsets_onto_the_reference(tmp_path: Path):
    """combine: add brings each segment onto its neighbour's level by an offset, not a factor."""
    from astropy.table import Table

    w1, w2 = np.linspace(1.0, 2.0, 200), np.linspace(1.8, 3.0, 200)
    blue = synthetic_x1d(tmp_path / "jw001_nirspec_g235h-f170lp_s1d.fits", wave=w1, flux=1.0 + 0.5 * w1,
                         instrument="NIRSPEC", GRATING="G235H", FILTER="F170LP")
    red = synthetic_x1d(tmp_path / "jw001_nirspec_g395h-f290lp_s1d.fits", wave=w2, flux=1.0 + 0.5 * w2 + 0.3,
                        instrument="NIRSPEC", GRATING="G395H", FILTER="F290LP")
    out = run_step(StitchSegments, [blue, red], tmp_path,
                   params={"rescale": True, "combine": "add", "crossovers": [1.9], "reference": "g235h-f170lp"})
    tab = Table.read([o for o in out if o.suffix == ".ecsv"][0])
    assert tab.meta["neighbour_offsets"][0] == pytest.approx(0.3, abs=0.01)
    assert tab.meta["offsets_applied"] == [0.0, pytest.approx(-0.3, abs=0.01)]
    assert tab.meta["scales_applied"] == [1.0, 1.0]
    w, f = np.asarray(tab["WAVELENGTH"]), np.asarray(tab["FLUX"])
    assert np.allclose(f, 1.0 + 0.5 * w, atol=0.01)                     # the slope is kept, only the level moves


def test_fixed_scales_apply_a_measured_factor_with_its_error(tmp_path: Path):
    from astropy.table import Table

    out = run_step(StitchSegments, two_segments(tmp_path), tmp_path,
                   params={"crossovers": [1.9], "fixed_scales": {"g235h-f170lp": [0.5, 0.05]}})
    tab = Table.read([o for o in out if o.suffix == ".ecsv"][0])
    w, f, e = (np.asarray(tab[c]) for c in ("WAVELENGTH", "FLUX", "FLUX_ERROR"))
    assert np.allclose(f[w < 1.9], 0.5) and np.allclose(f[w > 1.9], 2.0)
    assert tab.meta["scales_applied"][0] == pytest.approx(0.5)
    assert tab.meta["fixed_scale_rel_errors"][0] == pytest.approx(0.1)
    # the factor's 10 % is a systematic of the segment: in FLUX_SYS, not in each sample's FLUX_ERROR
    assert np.allclose(np.asarray(tab["FLUX_SYS"])[w < 1.9], 0.05) and np.allclose(np.asarray(tab["FLUX_SYS"])[w > 1.9], 0.0)
    assert np.all(e[w < 1.9] < 0.05)


def test_stitched_ecsv_inputs_are_single_segments(tmp_path: Path):
    """A combine joins the stitched spectra of two runs (*_s1dcomb.ecsv) as two segments."""
    from astropy.table import Table

    for name, lo, hi, flux in (("jw001-o006_t001_nirspec_s1dcomb.ecsv", 1.7, 5.3, 1.0),
                               ("jw001-o010_t002_miri_target_disk_s1dcomb.ecsv", 4.9, 28.0, 2.0)):
        w = np.linspace(lo, hi, 300)
        t = Table({"WAVELENGTH": w, "FLUX": np.full(300, flux), "FLUX_ERROR": np.full(300, 0.01),
                   "SEGMENT": np.full(300, "x")})
        (tmp_path / "in").mkdir(exist_ok=True)
        t.write(tmp_path / "in" / name, format="ascii.ecsv", overwrite=True)
    files = sorted((tmp_path / "in").glob("*.ecsv"))
    out = run_step(StitchSegments, files, tmp_path / "out", params={"crossovers": [5.1], "rescale": True,
                                                                    "reference": "miri"})
    tab = Table.read([o for o in out if o.suffix == ".ecsv"][0])
    assert tab.meta["segments"] == ["nirspec", "miri"]
    assert tab.meta["scales_applied"] == [pytest.approx(2.0), 1.0]


def test_wavelength_limits_drop_samples(tmp_path: Path):
    from astropy.table import Table

    out = run_step(StitchSegments, two_segments(tmp_path), tmp_path,
                   params={"crossovers": [1.9], "min_wavelength": 1.2, "max_wavelength": 2.5})
    tab = Table.read([o for o in out if o.suffix == ".ecsv"][0])
    w = np.asarray(tab["WAVELENGTH"])
    assert w.min() >= 1.2 and w.max() <= 2.5


def test_crossfade_blends_across_the_ratio_window(tmp_path: Path):
    from astropy.table import Table

    out = run_step(StitchSegments, two_segments(tmp_path), tmp_path, params={"crossovers": [1.9], "crossfade": True})
    tab = Table.read([o for o in out if o.suffix == ".ecsv"][0])
    w, f = np.asarray(tab["WAVELENGTH"]), np.asarray(tab["FLUX"])
    lo, hi = tab.meta["crossfade_windows_um"][0]
    assert np.all(f[w < lo] == pytest.approx(1.0)) and np.all(f[w > hi] == pytest.approx(2.0))
    inside = (w > lo + 0.01) & (w < hi - 0.01)
    assert inside.any() and np.all((f[inside] > 1.0) & (f[inside] < 2.0)) and np.all(np.diff(f[inside]) >= -1e-9)


def test_edge_trim_drops_band_edges(tmp_path: Path):
    from astropy.table import Table

    out = run_step(StitchSegments, two_segments(tmp_path), tmp_path, params={"crossovers": [1.9], "edge_trim": 5})
    tab = Table.read([o for o in out if o.suffix == ".ecsv"][0])
    w = np.asarray(tab["WAVELENGTH"])
    step = 1.0 / 199
    assert w.min() == pytest.approx(1.0 + 5 * step) and w.max() == pytest.approx(3.0 - 5 * 1.2 / 199)
