"""Archival photometry next to a reduction (jwstflow.spectra.archival) and the plot_archival QA step."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jwstflow.qa.steps import PlotArchival
from jwstflow.spectra.archival import find_archival, photometry_table, read_archival
from jwstflow.testing import make_context, run_step, synthetic_x1d


def _photometry(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    tab = photometry_table([2.15, 3.4, 24.0, 160.0], [1.0, 0.6, 8.0, 600.0], [0.1, 0.02, 0.8, 200.0],
                           upper_limit=[False, False, False, True], bandwidth_um=[0.26, 0.66, 5.3, 85.0],
                           reference=["skrutskie2006", "cutri2012", "luhman2008", "winston2012"],
                           citation=["Skrutskie et al. 2006", "Cutri et al. 2012", "Luhman et al. 2008",
                                     "Winston et al. 2012"],
                           meta={"target": "Test Target", "description": "test"}, upper_limit_sigma=3)
    path = directory / "t_photometry.ecsv"
    tab.write(path, format="ascii.ecsv")
    return path


def test_photometry_round_trip_in_jy_with_upper_limits(tmp_path):
    data = read_archival(_photometry(tmp_path / "archival"))
    assert data.kind == "photometry" and data.upper_limit_sigma == 3.0
    assert np.allclose(data.flux, [1.0e-3, 0.6e-3, 8.0e-3, 0.6]) and list(data.upper_limit) == [False] * 3 + [True]
    assert np.allclose(data.bandwidth, [0.26, 0.66, 5.3, 85.0]) and data.citation[3] == "Winston et al. 2012"
    assert find_archival(tmp_path) == [tmp_path / "archival" / "t_photometry.ecsv"]


def test_upper_limits_need_their_significance(tmp_path):
    with pytest.raises(ValueError, match="upper_limit_sigma"):
        photometry_table([1.0], [1.0], [0.1], upper_limit=[True])


def test_archival_figures_over_the_spectrum_and_the_sed(tmp_path):
    ctx = make_context(tmp_path / "work", stage="plot_archival")
    _photometry(ctx.target_dir / "archival")
    spectrum = synthetic_x1d(tmp_path / "t_s1d.fits", wave=np.linspace(1.7, 27.0, 800),
                             flux=np.geomspace(1e-3, 8e-3, 800), instrument="MIRI")
    figures = run_step(PlotArchival, [spectrum], tmp_path, ctx=ctx)
    assert [p.name for p in figures] == ["t_s1d_archival.png", "t_s1d_archival_sed.png"]
    assert all(p.stat().st_size > 20_000 for p in figures)


def test_no_archival_files_no_figure(tmp_path):
    spectrum = synthetic_x1d(tmp_path / "t_s1d.fits", wave=np.linspace(5.0, 6.0, 50), flux=np.ones(50))
    assert run_step(PlotArchival, [spectrum], tmp_path) == []
