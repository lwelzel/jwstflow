"""The mock-observation toolkit (jwstflow.testing.mock) and a complete engine run on it.

`test_mini_workflow_end_to_end` is the core integration test: mock uncal files
through discovery, associations, the process-pool executor, DMS product naming,
the workflow-graph rendering, QA steps, and checkpoint/resume -- in seconds and
fully offline. The contributed-step repositories build their deeper chains
(disk masks, MRS extraction) on the same toolkit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jwstflow.config.loader import load_config
from jwstflow.engine.runner import Runner
from jwstflow.testing.mock import (
    MockScene,
    miri_mrs_observation,
    nirspec_ifu_observation,
    offline_overrides,
    stub_pipelines,
    write_observation,
)

YAML = Path(__file__).parent / "data" / "mini_nirspec.yaml"


# ------------------------------------------------------------------ observations


def test_nirspec_observation_matches_the_real_structure():
    obs = nirspec_ifu_observation(dithers=4)  # the real jw01751 obs 6 layout
    assert len(obs.exposures) == 2 * 4 * 2  # gratings x dithers x detectors
    names = [e.filename for e in obs.exposures]
    assert names[0] == "jw01751006001_02101_00001_nrs1_uncal.fits"
    assert len(set(names)) == len(names)
    hdr = obs.exposures[0].header
    assert (hdr["PROGRAM"], hdr["OBSERVTN"], hdr["EXP_TYPE"]) == ("01751", "006", "NRS_IFU")
    assert hdr["GRATING"] in ("G235H", "G395H") and hdr["BKGDTARG"] is False
    assert obs.obs_ids == ["jw01751-o006_t005_nirspec"]


def test_miri_observation_covers_both_detectors_and_backgrounds():
    sci = miri_mrs_observation(observation=10, dithers=2)
    bkg = miri_mrs_observation(observation=12, dithers=1, background=True)
    assert len(sci.exposures) == 3 * 2 * 2  # bands x dithers x detectors
    assert {e.header["DETECTOR"] for e in sci.exposures} == {"MIRIFUSHORT", "MIRIFULONG"}
    assert {e.header["CHANNEL"] for e in sci.exposures} == {"12", "34"}
    assert all(e.header["BKGDTARG"] for e in bkg.exposures)
    assert bkg.scene.point_flux_jy == 0.0 and bkg.scene.disk_peak == 0.0  # sky only


def test_write_observation_creates_uncals_and_merges_the_mast_log(tmp_path: Path):
    raw = tmp_path / "raw"
    write_observation(raw, nirspec_ifu_observation(dithers=1, nwave=10, size=9))
    write_observation(raw, miri_mrs_observation(dithers=1, nwave=10, size=9))
    assert len(list(raw.glob("*_uncal.fits"))) == 4 + 6
    rows = json.loads((raw / "mast_observations.json").read_text())
    assert {r["obs_id"] for r in rows} == {"jw01751-o006_t005_nirspec", "jw01751-o010_t005_miri"}
    # the runner parses the DMS target id from exactly this file
    from jwstflow.data.download import target_ids_from_log

    assert target_ids_from_log(raw) == {("01751", "006"): "t005", ("01751", "010"): "t005"}


def test_scene_round_trips_through_headers():
    scene = MockScene(background=2.0, disk_peak=500.0, lines=((2.12, 100.0),), hot_pixel=None, seed=7)
    from astropy.io import fits

    hdr = fits.Header()
    for k, v in scene.to_header().items():
        hdr[k] = v
    assert MockScene.from_header(hdr) == scene


def test_scene_rendering_is_deterministic_and_scaled():
    scene = MockScene(background=1.0, disk_peak=100.0, hot_pixel=(0, 0), hot_value=1000.0)
    a = scene.cube(np.array([2.0, 2.1]), 15, 0.1)
    b = scene.cube(np.array([2.0, 2.1]), 15, 0.1)
    assert np.array_equal(a, b)
    assert a[0, 7, 7] == pytest.approx(101.0)  # disk peak + background at the centre
    assert a[0, 0, 0] == pytest.approx(1001.0, rel=1e-3)  # hot pixel over the sky


# ------------------------------------------------------------------ the mini workflow


@pytest.mark.integration
def test_mini_workflow_end_to_end(tmp_path: Path):
    write_observation(tmp_path / "reductions" / "eso-ha-569" / "raw",
                      nirspec_ifu_observation(dithers=2, nwave=30, size=21))
    cfg = load_config(YAML, overrides=offline_overrides(tmp_path, backend="process"))
    summary = Runner(cfg, skip_download=True).run()
    assert summary.ok and summary.failed == 0
    by_stage = {s.stage: s for s in summary.stages}
    assert by_stage["calwebb_detector1"].success == 8   # 2 gratings x 2 dithers x 2 detectors
    assert by_stage["calwebb_spec2"].success == 8
    assert by_stage["calwebb_spec3"].success == 2       # one association per grating
    assert by_stage["psf_cube"].success == 2            # one PSF cube per grating cube

    run = cfg.run_dir
    for product in [
        "stage1/calwebb_detector1/jw01751006001_02101_00001_nrs1_rate.fits",
        "stage2/calwebb_spec2/jw01751006001_02101_00001_nrs1_cal.fits",
        "stage3/calwebb_spec3/jw01751-o006_t005_nirspec_g235h-f170lp_s3d.fits",
        "stage3/calwebb_spec3/jw01751-o006_t005_nirspec_g395h-f290lp_x1d.fits",
        "stage4/psf_cube/jw01751-o006_t005_nirspec_g235h-f170lp_psfcube.fits",
        "qa/qa_psf_cube/jw01751-o006_t005_nirspec_g235h-f170lp_psfcube.png",
        "qa/quicklook_image/jw01751-o006_t005_nirspec_g235h-f170lp_s3d.png",
        "qa/plot_spectrum/jw01751-o006_t005_nirspec_g235h-f170lp_x1d.png",
    ]:
        assert (run / product).is_file(), product

    # the PSF library samples the cube's spaxels at oversample 2, on an odd grid
    from jwstflow.psf import PsfCubeProduct

    psf = PsfCubeProduct.read(run / "stage4/psf_cube/jw01751-o006_t005_nirspec_g235h-f170lp_psfcube.fits")
    assert psf.data.shape == (5, 33, 33)                       # ceil(1.6" / 0.05") -> odd
    assert psf.pixelscale_arcsec == pytest.approx(0.1 / 2)     # mock spaxels are 0.1"
    assert psf.frame == "ideal"
    assert psf.header["GRATING"] == "G235H"
    asns = sorted((run / "associations" / "calwebb_spec3").glob("*_asn.json"))
    assert [a.name for a in asns] == ["jw01751-o006_t005_nirspec_g235h_spec3_asn.json",
                                      "jw01751-o006_t005_nirspec_g395h_spec3_asn.json"]
    data = json.loads(asns[0].read_text())
    assert data["asn_type"] == "spec3" and len(data["products"][0]["members"]) == 4

    # cubes carry the scene: centre plane = disk peak + background, hot pixel excluded
    from stdatamodels.jwst import datamodels

    with datamodels.open(run / "stage3/calwebb_spec3/jw01751-o006_t005_nirspec_g235h-f170lp_s3d.fits") as cube:
        assert cube.data.shape == (30, 21, 21)
        assert float(cube.data[0].max()) > 900.0  # disk peak ~1000 over sky 1

    # -- resume: a second run computes nothing ---------------------------------
    summary2 = Runner(cfg, skip_download=True).run()
    assert summary2.ok
    assert all(s.cached == s.total and s.success == 0 for s in summary2.stages)

    # -- selective iteration: --only reruns just that stage under force --------
    summary3 = Runner(cfg, skip_download=True, only=["calwebb_spec3"], force=True).run()
    assert [s.stage for s in summary3.stages] == ["calwebb_spec3"]
    assert summary3.stages[0].success == 2

    # -- editing an input invalidates its tasks (fast fingerprint: mtime/size) -
    victim = next((tmp_path / "reductions" / "eso-ha-569" / "raw").glob("*_00001_nrs1_uncal.fits"))
    victim.touch()
    summary4 = Runner(cfg, skip_download=True).run()
    d1 = {s.stage: s for s in summary4.stages}["calwebb_detector1"]
    assert d1.success == 1 and d1.cached == 7


@pytest.mark.integration
def test_workflow_graph_renders_when_enabled(tmp_path: Path):
    write_observation(tmp_path / "reductions" / "eso-ha-569" / "raw",
                      nirspec_ifu_observation(dithers=1, nwave=6, size=9))
    overrides = [o for o in offline_overrides(tmp_path, backend="serial") if o != "workflow_graph=false"]
    cfg = load_config(YAML, overrides=overrides)
    with stub_pipelines():
        summary = Runner(cfg, skip_download=True, until="calwebb_detector1").run()
    assert summary.ok
    rendered = {p.suffix for p in (cfg.run_dir / "qa" / "workflow_graph").iterdir()}
    assert ".dot" in rendered and (".svg" in rendered or ".pdf" in rendered or ".png" in rendered)


def test_stub_pipelines_registry_round_trip():
    from jwstflow.steps.base import _REGISTRY, resolve_target

    assert "detector1" not in _REGISTRY
    with stub_pipelines():
        from jwstflow.testing.mock import StubDetector1

        assert resolve_target("detector1") is StubDetector1
    assert "detector1" not in _REGISTRY  # restored


def test_stub_declarations_are_valid():
    from jwstflow.testing import check_step
    from jwstflow.testing.mock import DEFAULT_STUBS

    for cls in DEFAULT_STUBS.values():
        _, problems = check_step(cls)
        assert problems == [], f"{cls.__name__}: {problems}"
