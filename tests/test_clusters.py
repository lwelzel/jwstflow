"""The bad-cluster chain (jwstflow.clusters): regions, the per-dither comparison, the
back-propagation to detector pixels through a real gwcs, and the QA figure."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from jwstflow.clusters import (
    CLUSTERMASK_SUFFIX,
    FlagSpaxelClusters,
    PropagateClusterFlags,
    dither_of,
    planes_in_window,
    read_cluster_product,
    region_aperture,
)
from jwstflow.contrib.clusters import QaSpaxelClusters
from jwstflow.masks import celestial_wcs
from jwstflow.testing import check_step, run_step, synthetic_cube
from jwstflow.testing.mock import (
    DEFAULT_NIRSPEC_SCENE,
    StubCubeBuild,
    StubDetector1,
    StubSpec2,
    StubSpec3,
    mock_cal_layout,
    nirspec_ifu_observation,
    write_observation,
)

PIX = 0.13          # synthetic_cube spaxel scale [arcsec]
NWAVE, SIZE = 20, 30
BLOB = (20, 8)      # (y, x) centre of the injected cluster
PLANES = (8, 9, 10)


def blob_mask() -> np.ndarray:
    """A deliberately non-circular cluster footprint: an L of 3x3 + 1x3 spaxels."""
    m = np.zeros((SIZE, SIZE), bool)
    y, x = BLOB
    m[y - 1:y + 2, x - 1:x + 2] = True
    m[y + 2:y + 5, x] = True
    return m


def cubes_with_cluster(tmp_path: Path, bad_dither: int = 2, dithers: int = 3) -> list[Path]:
    paths = []
    for d in range(1, dithers + 1):
        p = synthetic_cube(tmp_path / f"jw01751-o006_t005_nirspec_dither{d}_g395h-f290lp_s3d.fits", instrument="NIRSPEC",
                           nwave=NWAVE, size=SIZE, wave_min=4.5, wave_step=0.01, PATT_NUM=d, PROGRAM="01751",
                           OBSERVTN="006")
        if d == bad_dither:
            with fits.open(p, mode="update") as hdul:
                for k in PLANES:
                    hdul["SCI"].data[k][blob_mask()] += 500.0
        paths.append(p)
    return paths


def region_at(cube: Path, y: float, x: float, **extra) -> dict:
    from stdatamodels.jwst import datamodels

    with datamodels.open(cube) as model:
        ra, dec = celestial_wcs(model).all_pix2world(x, y, 0)
    return {"id": "test", "ra": float(ra), "dec": float(dec), "radius": 0.6, "wave_min": 4.58, "wave_max": 4.60, **extra}


def test_declarations_are_valid():
    for cls in (FlagSpaxelClusters, PropagateClusterFlags, QaSpaxelClusters):
        _, problems = check_step(cls)
        assert problems == [], f"{cls.__name__}: {problems}"


def test_region_apertures_on_a_cube_grid(tmp_path: Path):
    from stdatamodels.jwst import datamodels

    cube = synthetic_cube(tmp_path / "c_s3d.fits", instrument="NIRSPEC", nwave=2, size=SIZE)
    with datamodels.open(cube) as model:
        wcs = celestial_wcs(model)
    ra, dec = wcs.all_pix2world(15, 15, 0)
    circle = region_aperture({"shape": "circle", "ra": float(ra), "dec": float(dec), "radius": 0.5}, wcs, (SIZE, SIZE))
    assert circle[15, 15] and circle.sum() == pytest.approx(np.pi * (0.5 / PIX) ** 2, rel=0.2)
    ellipse = region_aperture({"shape": "ellipse", "ra": float(ra), "dec": float(dec), "radius": 0.8,
                               "radius_minor": 0.2, "angle": 0.0}, wcs, (SIZE, SIZE))
    assert ellipse[15 + 5, 15] and not ellipse[15, 15 + 5]      # major axis along north (y)
    ra_e, dec_n = wcs.all_pix2world([12, 18, 18, 12], [12, 12, 18, 18], 0)
    poly = region_aperture({"shape": "polygon", "vertices": [[a, b] for a, b in zip(ra_e, dec_n)]}, wcs, (SIZE, SIZE))
    assert poly[15, 15] and poly.sum() == pytest.approx(36, abs=8)
    # sexagesimal centres are accepted
    sexa = region_aperture({"shape": "circle", "ra": "00:40:00", "dec": "-20:00:00", "radius": 0.5}, wcs, (SIZE, SIZE))
    assert sexa[15, 15]


def test_planes_in_window_uses_bin_overlap_and_never_comes_back_empty():
    wave = 4.5 + 0.01 * np.arange(NWAVE)
    assert planes_in_window(wave, 4.58, 4.60).tolist() == [8, 9, 10]
    assert planes_in_window(wave, 4.5812, 4.5814).tolist() == [8]       # narrower than a bin
    assert planes_in_window(wave, 4.58, 4.60, pad=1).tolist() == [7, 8, 9, 10, 11]
    assert planes_in_window(wave, 9.0, 9.1).tolist() == [NWAVE - 1]      # outside: the nearest


def test_dither_of_reads_patt_num_then_the_name():
    assert dither_of(fits.Header({"PATT_NUM": 3}), "x.fits") == 3
    assert dither_of(fits.Header(), "jw01751-o006_t005_nirspec_dither2_g395h-f290lp_s3d.fits") == 2
    assert dither_of(fits.Header(), "jw01751006001_02101_00004_nrs2_cal.fits") == 4


def test_auto_mode_flags_only_the_deviant_dither_with_the_clusters_own_shape(tmp_path: Path):
    cubes = cubes_with_cluster(tmp_path)
    region = region_at(cubes[0], *BLOB)
    outputs = run_step(FlagSpaxelClusters, cubes, tmp_path, params={"regions": [region], "grow": 0})
    assert [o.name for o in outputs] == [f"jw01751-o006_t005_nirspec_dither{d}_g395h-f290lp_{CLUSTERMASK_SUFFIX}.fits"
                                         for d in (1, 2, 3)]
    products = [read_cluster_product(o) for o in outputs]
    rows = {int(p["regions"]["dither"][0]): p["regions"][0] for p in products}
    assert bool(rows[2]["included"]) and not bool(rows[1]["included"]) and not bool(rows[3]["included"])
    assert rows[2]["mode"] == "auto" and int(rows[2]["n_others"]) == 2
    assert float(rows[2]["score"]) > 100 and float(rows[1]["score"]) < 1
    bad = products[1]
    # exactly the blob's spaxels on the window planes: the aperture is round, the mask is not
    for k in PLANES:
        assert np.array_equal(bad["mask"][k], blob_mask()), k
    assert not bad["mask"][[k for k in range(NWAVE) if k not in PLANES]].any()
    assert bad["header"]["JWFCLNPX"] == 3 * int(blob_mask().sum()) and bad["header"]["JWFCLINC"] == "test"
    assert not products[0]["mask"].any() and products[0]["header"]["JWFCLINC"] == "none"
    # QA payload: window planes plus 2 context planes on each side, with the aperture
    sl = bad["slices"][0]
    assert sl["data"].shape == (7, SIZE, SIZE) and np.allclose(sl["wave"], 4.5 + 0.01 * np.arange(6, 13))
    assert sl["aperture"][BLOB] and int(sl["aperture"].sum()) > int(blob_mask().sum())
    assert np.nanmax(sl["sigma"][2][blob_mask()]) > 100 and np.nanmax(sl["ref"]) < 4000


def test_explicit_and_all_modes_and_grow(tmp_path: Path):
    cubes = cubes_with_cluster(tmp_path)
    explicit = region_at(cubes[0], *BLOB, dithers=[1])           # a dither with nothing deviant
    outputs = run_step(FlagSpaxelClusters, cubes, tmp_path, params={"regions": [explicit], "grow": 0})
    p1, p2 = (read_cluster_product(o) for o in outputs[:2])
    assert bool(p1["regions"]["included"][0]) and not bool(p1["regions"]["refined"][0])
    aperture = p1["slices"][0]["aperture"]
    assert np.array_equal(p1["mask"][8], aperture)                  # whole aperture, nothing to refine
    assert not p2["mask"].any() and p2["regions"]["reason"][0] == "dither 2 not listed"
    grown = region_at(cubes[0], *BLOB, dithers="all", refine=False)
    outputs = run_step(FlagSpaxelClusters, cubes, tmp_path, params={"regions": [grown], "grow": 1},
                       stage="grown")
    for o in outputs:
        p = read_cluster_product(o)
        assert p["mask"][9].sum() > aperture.sum() and p["mask"][9][aperture].all()


def test_regions_outside_the_field_or_band_are_reported_not_fatal(tmp_path: Path, caplog):
    cubes = cubes_with_cluster(tmp_path)
    region = region_at(cubes[0], *BLOB, where={"GRATING": "G395H"})   # synthetic cubes say G235H
    far = {"id": "far", "ra": 11.0, "dec": -20.0, "radius": 0.3, "wave_min": 4.58, "wave_max": 4.60}
    outputs = run_step(FlagSpaxelClusters, cubes, tmp_path, params={"regions": [region, far]})
    for o in outputs:
        p = read_cluster_product(o)
        assert not p["mask"].any()
        assert list(p["regions"]["id"]) == ["far"] and int(p["regions"]["n_aperture"][0]) == 0
    assert "no cube covers" in caplog.text and "falls outside" in caplog.text


# --------------------------------------------------------------------------- back-propagation


@pytest.fixture()
def mock_chain(tmp_path: Path) -> dict:
    """Three-dither mock G395H cal files (with their gwcs) and per-dither cubes, dither 2 carrying a cluster."""
    scene = replace(DEFAULT_NIRSPEC_SCENE, cluster_dither=2, cluster_center=(11, 4), cluster_radius=2.5,
                    cluster_wave=(4.5, 4.7), cluster_value=800.0)
    obs = nirspec_ifu_observation(dithers=3, nwave=24, size=15, gratings=(("G395H", "F290LP"),), scene=scene)
    raw = tmp_path / "raw"
    write_observation(raw, obs)
    cals = []
    for uncal in sorted(raw.glob("*_uncal.fits")):
        (rate,) = run_step(StubDetector1, [uncal], tmp_path, stage="d1")
        (cal,) = run_step(StubSpec2, [rate], tmp_path, stage="s2")
        cals.append(cal)
    cubes = []
    for d in (1, 2, 3):
        members = [c for c in cals if fits.getheader(c)["PATT_NUM"] == d]
        asn = tmp_path / f"dither{d}_asn.json"
        asn.write_text(json.dumps({"products": [{"name": f"jw01751-o006_t005_nirspec_dither{d}_g395h",
                                                 "members": [{"expname": str(m), "exptype": "science"} for m in members]}]}))
        cubes += run_step(StubCubeBuild, [asn], tmp_path, stage="cb")
    return {"scene": scene, "cals": cals, "cubes": cubes, "raw": raw}


def test_mock_cal_files_carry_a_faithful_ifu_layout(mock_chain):
    from stdatamodels.jwst import datamodels

    cal = next(c for c in mock_chain["cals"] if c.name.endswith("_nrs2_cal.fits"))
    hdr = fits.getheader(cal)
    layout = mock_cal_layout(hdr)
    assert layout["shape"] == (15 * 15, 12) and layout["planes"] == (12, 24)
    with datamodels.open(cal) as model:
        assert model.data.shape == layout["shape"]
        ra, dec, lam = model.meta.wcs(3, 11 * 15 + 4)          # column 3 -> plane 15, row -> spaxel (11, 4)
        assert lam == pytest.approx(layout["wave"][15])
        cube = next(c for c in mock_chain["cubes"] if "dither1" in c.name)
        with datamodels.open(cube) as cube_model:
            x, y = celestial_wcs(cube_model).all_world2pix(ra, dec, 0)
        assert (round(float(x)), round(float(y))) == (4, 11)
        assert np.isnan(model.meta.wcs(3, 15 * 15 + 5)[2])      # outside the bounding box


def test_flags_propagate_to_exactly_the_detector_pixels_behind_the_cluster(mock_chain, tmp_path: Path):
    from stdatamodels.jwst import datamodels

    cubes, cals = mock_chain["cubes"], mock_chain["cals"]
    region = {"id": "blob", "ra": None, "dec": None, "radius": 0.36, "wave_min": 4.5, "wave_max": 4.7}
    with datamodels.open(cubes[0]) as model:
        ra, dec = celestial_wcs(model).all_pix2world(4, 11, 0)
    region.update(ra=float(ra), dec=float(dec))
    products = run_step(FlagSpaxelClusters, cubes, tmp_path, stage="flag_spaxel_clusters",
                        params={"regions": [region], "grow": 0})
    ctx_stage_dirs = {"flag_spaxel_clusters": products[0].parent}
    bad = read_cluster_product(next(p for p in products if "dither2" in p.name))
    assert bad["header"]["JWFCLNPX"] > 0
    flagged_planes = np.flatnonzero(bad["mask"].any(axis=(1, 2)))
    assert flagged_planes.min() >= 12                            # the cluster sits in the NRS2 half of G395H

    from jwstflow.testing import make_context

    ctx = make_context(tmp_path, stage="propagate_cluster_flags")
    ctx.stage_dirs = ctx_stage_dirs
    flagged = {}
    for cal in cals:
        (out,) = run_step(PropagateClusterFlags, [cal], tmp_path, ctx=ctx, params={"plane_pad": 0})
        h = fits.getheader(out)
        flagged[cal.name] = (h["JWFCLNPX"], h["JWFCLMSK"], h["JWFCLRGS"], out)
    # only NRS2 of dither 2 is touched, with one detector pixel per flagged spaxel-plane
    bad_cal = next(n for n in flagged if "_00002_nrs2" in n)
    n, msk, rgs, out = flagged[bad_cal]
    assert n == int(bad["mask"].sum()) and msk.endswith("_clustermask.fits") and rgs == "blob"
    for name, (n_other, _, _, _) in flagged.items():
        if name != bad_cal:
            assert n_other == 0, name
    dq = fits.getdata(out, "DQ")
    rows, cols = np.nonzero(dq & 16)
    iy, ix = np.divmod(rows, 15)
    assert np.array_equal(np.sort(cols + 12), np.sort(np.nonzero(bad["mask"])[0]))
    assert bad["mask"][cols + 12, iy, ix].all()                 # every flagged pixel lands on a flagged spaxel
    assert (dq[rows, cols] & 1).all()                            # DO_NOT_USE as well
    # the untouched frames are plain copies
    clean = next(flagged[n][3] for n in flagged if n != bad_cal)
    assert fits.getdata(clean, "DQ").sum() == 0

    # -- the cubes built from the flagged frames no longer carry the cluster ----------------
    propagated = sorted(out.parent.glob("*_cal.fits"))
    asn = tmp_path / "all_asn.json"
    asn.write_text(json.dumps({"products": [{"name": "jw01751-o006_t005_nirspec_g395h",
                                             "members": [{"expname": str(m), "exptype": "science"} for m in propagated]}]}))
    (combined,) = run_step(StubSpec3, [asn], tmp_path, stage="s3")
    with datamodels.open(combined) as cube:
        data = np.asarray(cube.data, float)
    k = int(flagged_planes[0])
    wave = fits.getdata(products[0], "WAVES")
    truth = DEFAULT_NIRSPEC_SCENE.plane(15, 0.1, float(wave[k]))
    assert np.allclose(data[k], truth, rtol=1e-5) and np.isfinite(data[k]).all()
    # ... whereas the same frames without the propagation average the cluster in
    asn_raw = tmp_path / "raw_asn.json"
    asn_raw.write_text(json.dumps({"products": [{"name": "jw01751-o006_t005_nirspec_g395h",
                                                 "members": [{"expname": str(m), "exptype": "science"} for m in cals]}]}))
    (dirty,) = run_step(StubSpec3, [asn_raw], tmp_path, stage="s3raw")
    with datamodels.open(dirty) as cube:
        assert float(cube.data[k][11, 4]) > truth[11, 4] + 200


def test_qa_figure_has_one_row_per_dither(mock_chain, tmp_path: Path):
    cubes = mock_chain["cubes"]
    from stdatamodels.jwst import datamodels

    with datamodels.open(cubes[0]) as model:
        ra, dec = celestial_wcs(model).all_pix2world(4, 11, 0)
    region = {"id": "blob 1", "ra": float(ra), "dec": float(dec), "radius": 0.36, "wave_min": 4.5, "wave_max": 4.7}
    products = run_step(FlagSpaxelClusters, cubes, tmp_path, stage="flag", params={"regions": [region]})
    (png,) = run_step(QaSpaxelClusters, products, tmp_path, stage="qa", params={"panel": 1.6})
    assert png.name == "jw01751-o006_t005_nirspec_g395h-f290lp_blob-1_clusters.png" and png.stat().st_size > 10_000
