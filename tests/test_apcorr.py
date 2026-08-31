"""Tests of the shared aperture-correction helpers (jwstflow.apcorr)."""

from __future__ import annotations

import numpy as np
import pytest

from jwstflow.apcorr import apcorr_factor, mrs_apcorr_table, nrs_apcorr_table


def test_apcorr_factor_interpolates_and_clamps():
    table = {"wavelength": np.array([1.0, 2.0]),
             "radius": np.array([[0.1, 0.1], [1.0, 1.0]]),
             "apcorr": np.array([[2.0, 4.0], [1.0, 1.0]]),
             "radius_units": "arcsec"}
    wave = np.array([1.0, 1.5, 2.0])
    factor = apcorr_factor(table, wave, np.array([0.1, 0.55, 5.0]))
    assert np.isclose(factor[0], 2.0)          # small aperture at 1 um
    assert np.isclose(factor[1], 2.0)          # midpoint in wavelength and radius: (3.0+1.0)/2
    assert np.isclose(factor[2], 1.0)          # beyond the largest radius: clamped, never < 1


def test_apcorr_factor_converts_pixel_radii():
    table = {"wavelength": np.array([1.0, 2.0]),
             "radius": np.array([[1.0, 1.0], [10.0, 10.0]]),
             "apcorr": np.array([[3.0, 3.0], [1.0, 1.0]]),
             "radius_units": "pixels"}
    factor = apcorr_factor(table, np.array([1.5]), np.array([0.55]), pix_arcsec=0.1)  # 5.5 pix: midpoint
    assert np.isclose(factor[0], 2.0)
    with pytest.raises(ValueError, match="pixel scale"):
        apcorr_factor(table, np.array([1.5]), np.array([0.55]))


def test_mrs_table_tolerates_transposed_layout():
    class Node:
        wavelength = np.array([5.0, 6.0, 7.0])
        radius = np.array([[0.2, 0.5], [0.3, 0.6], [0.4, 0.7]])   # (nwave x nradius): transposed
        apcorr = np.array([[2.0, 1.0], [2.1, 1.1], [2.2, 1.2]])
        radius_units = "arcsec"

    table = mrs_apcorr_table(Node())
    assert table["radius"].shape == (2, 3) and table["apcorr"].shape == (2, 3)
    assert np.allclose(table["radius"][0], [0.2, 0.3, 0.4])


def test_nrs_table_selects_the_row_and_truncates():
    rows = [
        {"filter": "F290LP", "grating": "G395H", "nelem_wl": 2,
         "wavelength": np.array([2.9, 4.1, 0.0]), "radius": np.array([[0.1, 0.2, 9.0], [1.0, 1.1, 9.0]]),
         "apcorr": np.array([[2.0, 2.1, 9.0], [1.0, 1.0, 9.0]])},
        {"filter": "F170LP", "grating": "G235H", "nelem_wl": 3,
         "wavelength": np.array([1.7, 2.4, 3.1]), "radius": np.array([[0.1, 0.2, 0.3]]),
         "apcorr": np.array([[2.0, 2.1, 2.2]])},
    ]
    table = nrs_apcorr_table(rows, "f170lp", "g235h", sizeunit="arcsec")
    assert table["wavelength"].tolist() == [1.7, 2.4, 3.1]
    truncated = nrs_apcorr_table(rows, "F290LP", "G395H")
    assert truncated["wavelength"].tolist() == [2.9, 4.1] and truncated["radius"].shape == (2, 2)
    with pytest.raises(ValueError, match="no row for PRISM/CLEAR"):
        nrs_apcorr_table(rows, "CLEAR", "PRISM")


def test_load_apcorr_reads_a_mirmrs_reference(tmp_path):
    datamodels = pytest.importorskip("stdatamodels.jwst.datamodels")

    from jwstflow.apcorr import load_apcorr
    from jwstflow.testing import synthetic_cube

    ref = datamodels.MirMrsApcorrModel()
    ref.apcorr_table.channel, ref.apcorr_table.band = "1", "SHORT"
    ref.apcorr_table.wavelength = np.array([5.0, 6.0], dtype="f4")
    ref.apcorr_table.radius = np.array([[0.2, 0.3], [0.5, 0.6], [1.0, 1.2]], dtype="f4")  # 3 radii x 2 waves
    ref.apcorr_table.apcorr = np.array([[2.0, 2.2], [1.3, 1.4], [1.0, 1.05]], dtype="f4")
    ref.apcorr_table.radius_units = "arcsec"
    ref.meta.instrument.name, ref.meta.reftype = "MIRI", "apcorr"
    path = tmp_path / "apcorr.asdf"
    ref.save(str(path))

    cube_path = synthetic_cube(tmp_path / "jw_ch1-short_s3d.fits", instrument="MIRI")
    with datamodels.open(cube_path) as cube:
        table = load_apcorr(cube, str(path))
    assert table["radius"].shape == (3, 2) and table["radius_units"] == "arcsec"
    factor = apcorr_factor(table, np.array([5.0, 5.5, 6.0]), np.array([0.5, 0.55, 5.0]), 0.13)
    assert factor[0] == pytest.approx(1.3) and 1.3 < factor[1] < 1.4 and factor[2] == pytest.approx(1.05)
