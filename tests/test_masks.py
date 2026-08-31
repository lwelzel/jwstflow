"""Tests of the mask-product contract (jwstflow.masks)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jwstflow.masks import (
    clean_blob,
    find_mask_product,
    matches_instrument,
    read_mask_product,
    write_mask_product,
)
from jwstflow.testing import synthetic_cube


@pytest.fixture()
def cube(tmp_path: Path) -> Path:
    return synthetic_cube(tmp_path / "jw001_nirspec_g235h-f170lp_s3d.fits", instrument="NIRSPEC", nwave=10, size=20)


def test_roundtrip_with_features(cube: Path, tmp_path: Path):
    mask = np.zeros((10, 20, 20), bool)
    mask[:, 8:12, 8:12] = True
    cont = mask[0]
    contimg = np.random.default_rng(0).normal(1.0, 0.1, (20, 20))
    features = {"id": ["h2_1-0_s1"], "label": ["H2 1-0 S(1)"], "species": ["H2"],
                "wave_min_um": [2.07], "wave_max_um": [2.17], "nplanes": [3], "added_pix": [4], "changed": [True]}
    out = tmp_path / "jw001_nirspec_g235h-f170lp_diskmask.fits"
    write_mask_product(out, cube, mask, cont, contimg,
                       features=features, featimg=[contimg], featmask=[cont], keys={"JWFTEST": 1})
    dm = read_mask_product(out)
    assert dm["mask"].shape == (10, 20, 20) and dm["mask"].dtype == bool
    assert np.array_equal(dm["cont"], cont)
    assert np.allclose(dm["contimg"], contimg, atol=1e-6)
    assert dm["features"]["id"][0] == "h2_1-0_s1" and bool(dm["features"]["changed"][0])
    assert dm["featmask"].shape == (1, 20, 20)
    assert dm["header"]["JWFTEST"] == 1
    assert dm["header"]["GRATING"] == "G235H"  # instrument keys travel from the cube


def test_find_matches_instrument_config(cube: Path, tmp_path: Path):
    other = synthetic_cube(tmp_path / "jw001_nirspec_g395h-f290lp_s3d.fits", instrument="NIRSPEC", nwave=10, size=20)
    from astropy.io import fits

    with fits.open(other, mode="update") as hdul:
        hdul[0].header["GRATING"], hdul[0].header["FILTER"] = "G395H", "F290LP"
    m = np.zeros((10, 20, 20), bool)
    for src, name in ((cube, "a_diskmask.fits"), (other, "b_diskmask.fits")):
        write_mask_product(tmp_path / name, src, m, m[0], np.zeros((20, 20)))
    found = find_mask_product(other, tmp_path, "diskmask")
    assert found["path"].name == "b_diskmask.fits"
    with pytest.raises(FileNotFoundError):
        find_mask_product(cube, tmp_path, "regmask")


def test_matches_instrument_ignores_absent_keys():
    from astropy.io import fits

    a, b = fits.Header({"GRATING": "G235H"}), fits.Header({"GRATING": "g235h", "FILTER": "F170LP"})
    assert matches_instrument(a, b)
    b["GRATING"] = "G395H"
    assert not matches_instrument(a, b)


def test_clean_blob_keeps_anchor_connected_structure():
    m = np.zeros((20, 20), bool)
    m[5:10, 5:10] = True   # the disk
    m[0:2, 15:17] = True   # a hot blob far away
    anchor = np.zeros_like(m)
    anchor[6:8, 6:8] = True
    cleaned = clean_blob(m, anchor=anchor)
    assert cleaned[7, 7] and not cleaned[1, 16]
