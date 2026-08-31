"""Tests for write_x1d extra columns and the packaged reference data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jwstflow.spectra import Spectrum1D, read_x1d, write_x1d


def test_write_x1d_carries_extra_columns(tmp_path: Path):
    wave = np.linspace(1.0, 2.0, 50)
    spec = Spectrum1D(wave, np.ones(50), np.full(50, 0.1))
    out = write_x1d(tmp_path / "a_s1d.fits", spec,
                    columns={"npixels": np.full(50, 7.0), "sb_error": np.full(50, 0.3)})
    from astropy.table import Table

    tab = Table.read(out, hdu="EXTRACT1D")
    assert np.all(np.asarray(tab["NPIXELS"]) == 7.0)
    assert np.allclose(np.asarray(tab["SB_ERROR"]), 0.3)
    assert read_x1d(out).flux == pytest.approx(np.ones(50))


def test_write_x1d_rejects_unknown_columns(tmp_path: Path):
    spec = Spectrum1D(np.linspace(1, 2, 5), np.ones(5), np.ones(5))
    with pytest.raises(ValueError, match="unknown EXTRACT1D column"):
        write_x1d(tmp_path / "b_s1d.fits", spec, columns={"bogus": np.ones(5)})


def test_reference_data_ships_with_the_package(tmp_path: Path, monkeypatch):
    """The spectral-feature tables resolve without a project data/ directory or env var."""
    import jwstflow.features as features

    monkeypatch.delenv("JWSTFLOW_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)  # a bare directory: no project root, no data/
    d = features.data_dir()
    assert (d / "spectral_features" / "gas_lines.ecsv").is_file()
    assert d == Path(features.__file__).resolve().parent / "refdata"
    lines = features.load_dataset("gas_lines")
    assert any(f.id == "h2_1-0_s1" for f in lines)


def test_project_data_overrides_package_data(tmp_path: Path, monkeypatch):
    import jwstflow.features as features

    monkeypatch.delenv("JWSTFLOW_DATA_DIR", raising=False)
    (tmp_path / ".jwstflow-root").touch()
    override = tmp_path / "data" / "spectral_features"
    override.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert features.data_dir() == tmp_path / "data"
