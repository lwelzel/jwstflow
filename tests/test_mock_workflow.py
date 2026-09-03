"""The mock-observation toolkit (jwstflow.testing.mock) and a complete engine run on it.

`test_mini_workflow_end_to_end` is the core integration test: mock uncal files
through discovery, associations, the process-pool executor, DMS product naming,
the workflow-graph rendering, QA steps, and checkpoint/resume -- in seconds and
fully offline. The contributed-step repositories build their deeper chains
(disk masks, MRS extraction) on the same toolkit.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from jwstflow.config.loader import load_config
from jwstflow.engine.runner import Runner
from jwstflow.testing.mock import (
    DEFAULT_NIRSPEC_SCENE,
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


def test_imager_star_field_and_catalog_are_consistent(tmp_path: Path):
    from astropy.io import fits
    from astropy.table import Table

    from jwstflow.testing.mock import write_gaia_catalog

    obs = miri_mrs_observation(dithers=2, nwave=6, size=9, imager=True, n_stars=7,
                               pointing_error=(0.4, -0.3))
    imagers = [e for e in obs.exposures if e.header["EXP_TYPE"] == "MIR_IMAGE"]
    assert len(imagers) == 2 and all(e.header["DETECTOR"] == "MIRIMAGE" for e in imagers)
    assert imagers[0].header["JWFMKPRA"][0] == 0.4
    # every mock exposure carries the pointing reference wcs_offset shifts
    assert all(e.header["RA_REF"] == e.header["TARG_RA"] for e in obs.exposures)
    cat = write_gaia_catalog(tmp_path / "gaia.ecsv", obs)
    table = Table.read(cat)
    assert len(table) == 7 and {"ra", "dec", "pmra", "pmdec"} <= set(table.colnames)
    # the catalogue is the *truth*: identical for both dithers, independent of the error
    obs2 = miri_mrs_observation(dithers=1, nwave=6, size=9, imager=True, n_stars=7,
                                pointing_error=(0.0, 0.0))
    table2 = Table.read(write_gaia_catalog(tmp_path / "gaia2.ecsv", obs2))
    assert np.allclose(np.asarray(table["ra"]), np.asarray(table2["ra"]))

    # the image3 stub renders the stars displaced by the injected error
    raw = tmp_path / "raw"
    write_observation(raw, obs)
    from jwstflow.testing import run_step
    from jwstflow.testing.mock import StubDetector1, StubImage3

    uncal = sorted(raw.glob("*mirimage_uncal.fits"))[0]
    (rate,) = run_step(StubDetector1, [uncal], tmp_path, stage="d1")
    asn = tmp_path / "img3_asn.json"
    asn.write_text('{"products": [{"name": "jw01751-o010_t005_mirimage", "members": '
                   f'[{{"expname": "{rate}", "exptype": "science"}}]}}]}}')
    (i2d,) = run_step(StubImage3, [asn], tmp_path, stage="img3")
    assert i2d.name == "jw01751-o010_t005_mirimage_i2d.fits"
    with fits.open(i2d) as hdul:
        assert hdul[0].header["DATE-OBS"] and hdul["SCI"].data.max() > 10


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


#: the mini workflow's scene: the default disk plus a bad cluster in dither 2 of G395H, 4 px
#: east/north of the field centre (what the YAML's flag_spaxel_clusters region points at)
MINI_SCENE = replace(DEFAULT_NIRSPEC_SCENE, cluster_dither=2, cluster_center=(14, 6), cluster_radius=3.0,
                     cluster_wave=(4.5, 4.65), cluster_value=500.0)


@pytest.mark.integration
def test_mini_workflow_end_to_end(tmp_path: Path):
    write_observation(tmp_path / "reductions" / "eso-ha-569" / "raw",
                      nirspec_ifu_observation(dithers=3, nwave=30, size=21, scene=MINI_SCENE))
    cfg = load_config(YAML, overrides=offline_overrides(tmp_path, backend="process"))
    summary = Runner(cfg, skip_download=True).run()
    assert summary.ok and summary.failed == 0
    by_stage = {s.stage: s for s in summary.stages}
    assert by_stage["calwebb_detector1"].success == 12  # 2 gratings x 3 dithers x 2 detectors
    assert by_stage["calwebb_spec2"].success == 12
    assert by_stage["cube_build-perdither"].success == 6  # one cube per grating and dither
    assert by_stage["flag_spaxel_clusters"].success == 1  # batch=all
    assert by_stage["propagate_cluster_flags"].success == 12
    assert by_stage["calwebb_spec3"].success == 2       # one association per grating
    assert by_stage["psf_cube"].success == 2            # one PSF cube per grating cube

    run = cfg.run_dir
    for product in [
        "stage1/calwebb_detector1/jw01751006001_02101_00001_nrs1_rate.fits",
        "stage2/calwebb_spec2/jw01751006001_02101_00001_nrs1_cal.fits",
        "stage3/cube_build-perdither/jw01751-o006_t005_nirspec_dither2_g395h-f290lp_s3d.fits",
        "stage4/flag_spaxel_clusters/jw01751-o006_t005_nirspec_dither2_g395h-f290lp_clustermask.fits",
        "stage2/propagate_cluster_flags/jw01751006001_02101_00005_nrs2_cal.fits",
        "qa/qa_spaxel_clusters/jw01751-o006_t005_nirspec_g395h-f290lp_mock-blob_clusters.png",
        "stage3/calwebb_spec3/jw01751-o006_t005_nirspec_g235h-f170lp_s3d.fits",
        "stage3/calwebb_spec3/jw01751-o006_t005_nirspec_g395h-f290lp_x1d.fits",
        "stage4/psf_cube/jw01751-o006_t005_nirspec_g235h-f170lp_psfcube.fits",
        "qa/qa_psf_cube/jw01751-o006_t005_nirspec_g235h-f170lp_psfcube.png",
        "qa/quicklook_image/jw01751-o006_t005_nirspec_g235h-f170lp_s3d.png",
        "qa/plot_spectrum/jw01751-o006_t005_nirspec_g235h-f170lp_x1d.png",
        "qa/plot_spectrum/plot_spectrum_all.png",   # >= 2 spectra: the combined log-log overview
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
    assert data["asn_type"] == "spec3" and len(data["products"][0]["members"]) == 6
    # ... and the per-dither cubes' members are the two detectors of one dither each
    perdither = json.loads((run / "associations" / "cube_build-perdither"
                            / "jw01751-o006_t005_nirspec_dither2_g395h_spec3_asn.json").read_text())
    assert sorted(Path(m["expname"]).name for m in perdither["products"][0]["members"]) == [
        "jw01751006001_02101_00005_nrs1_cal.fits", "jw01751006001_02101_00005_nrs2_cal.fits"]

    # cubes carry the scene: centre plane = disk peak + background, hot pixel excluded
    from stdatamodels.jwst import datamodels

    with datamodels.open(run / "stage3/calwebb_spec3/jw01751-o006_t005_nirspec_g235h-f170lp_s3d.fits") as cube:
        assert cube.data.shape == (30, 21, 21)
        assert float(cube.data[0].max()) > 900.0  # disk peak ~1000 over sky 1

    # -- the bad-cluster chain: dither 2 of G395H found, its NRS2 frame flagged, cluster gone
    from jwstflow.clusters import read_cluster_product

    masks = {d: read_cluster_product(run / "stage4/flag_spaxel_clusters"
                                     / f"jw01751-o006_t005_nirspec_dither{d}_g395h-f290lp_clustermask.fits")
             for d in (1, 2, 3)}
    assert [bool(masks[d]["regions"]["included"][0]) for d in (1, 2, 3)] == [False, True, False]
    flagged_planes = np.flatnonzero(masks[2]["mask"].any(axis=(1, 2)))
    assert flagged_planes.tolist() == [20, 21] and masks[2]["mask"][20, 14, 6]
    for d in (1, 3):
        assert not masks[d]["mask"].any()
    npx = {p.name: fits.getheader(p)["JWFCLNPX"] for p in (run / "stage2/propagate_cluster_flags").glob("*_cal.fits")}
    # plane_pad (default 1) widens the two flagged planes to four: twice the spaxel-planes
    assert len(npx) == 12 and npx["jw01751006001_02101_00005_nrs2_cal.fits"] == 2 * int(masks[2]["mask"].sum())
    assert sum(npx.values()) == npx["jw01751006001_02101_00005_nrs2_cal.fits"]   # nothing else touched
    with datamodels.open(run / "stage3/calwebb_spec3/jw01751-o006_t005_nirspec_g395h-f290lp_s3d.fits") as cube:
        final = np.asarray(cube.data, float)
    with datamodels.open(run / "stage3/cube_build-perdither/jw01751-o006_t005_nirspec_dither2_g395h-f290lp_s3d.fits") as cube:
        dirty = np.asarray(cube.data, float)
    truth = DEFAULT_NIRSPEC_SCENE.plane(21, 0.1, float(masks[2]["waves"][20]))
    assert dirty[20, 14, 6] > truth[14, 6] + 400                    # the per-dither cube shows the cluster
    assert np.allclose(final[20], truth, rtol=1e-5) and np.isfinite(final[20]).all()   # the final one does not

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
    assert d1.success == 1 and d1.cached == 11


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
