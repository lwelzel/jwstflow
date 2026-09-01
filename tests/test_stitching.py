"""Tests of the generic segment stitcher (jwstflow.stitching.StitchSegments)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jwstflow.stitching import StitchSegments
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


def test_rescale_chains_onto_the_reference(tmp_path: Path):
    outputs = run_step(StitchSegments, two_segments(tmp_path), tmp_path,
                       params={"rescale": True, "crossovers": [1.9]})
    from astropy.table import Table

    tab = Table.read([o for o in outputs if o.suffix == ".ecsv"][0])
    # default reference is the reddest segment: the blue one is scaled up by the ratio 2
    assert tab.meta["scales_applied"] == [pytest.approx(2.0), 1.0]
    assert np.all(np.asarray(tab["FLUX"]) == pytest.approx(2.0))


def test_overlap_ratio_ignores_noise_crossing_zero(tmp_path: Path):
    # a noisy but significantly positive overlap: the ratio of overlap medians
    # must come out right even when single samples of `a` sit at or below zero
    rng = np.random.default_rng(5)
    n = 400
    blue = synthetic_x1d(tmp_path / "jw001_nirspec_g235h-f170lp_s1d.fits",
                         wave=np.linspace(1.0, 2.0, n), flux=0.10 + 0.08 * rng.standard_normal(n),
                         instrument="NIRSPEC", GRATING="G235H", FILTER="F170LP")
    red = synthetic_x1d(tmp_path / "jw001_nirspec_g395h-f290lp_s1d.fits",
                        wave=np.linspace(1.6, 3.0, n), flux=0.20 + 0.08 * rng.standard_normal(n),
                        instrument="NIRSPEC", GRATING="G395H", FILTER="F290LP")
    outputs = run_step(StitchSegments, [blue, red], tmp_path, params={"rescale": True})
    from astropy.table import Table

    tab = Table.read([o for o in outputs if o.suffix == ".ecsv"][0])
    assert tab.meta["neighbour_ratios"][0] == pytest.approx(2.0, rel=0.25)
    assert tab.meta["scales_applied"][0] == pytest.approx(2.0, rel=0.25)


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
    from jwstflow.contrib.qa import PlotStitch

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
    from jwstflow.contrib.qa import PlotStitch

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
    from jwstflow.contrib.qa import PlotStitchBackground

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
    from jwstflow.contrib.qa import PlotStitchBackground

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
