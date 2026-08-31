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
    assert [o.suffix for o in pngs] == [".png"]
