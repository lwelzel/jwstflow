"""How to test custom steps: no real data, no run directory, no jwst pipeline call."""

from pathlib import Path

from jwstflow.testing import check_step, run_step, synthetic_image, synthetic_x1d

from my_steps import SpectrumReport, tag_header


def test_declarations():
    for step in (SpectrumReport, tag_header):
        _, problems = check_step(step)
        assert problems == [], problems


def test_spectrum_report(tmp_path: Path):
    spectra = [synthetic_x1d(tmp_path / f"jw_g{i}_x1d.fits", GRATING=f"G{i}H") for i in (235, 395)]
    (out,) = run_step(SpectrumReport, spectra, tmp_path, params={"min_snr": 5})
    assert out.name.endswith("_report.json")
    import json

    report = json.loads(out.read_text())
    assert report["n"] == 2 and {r["grating"] for r in report["rows"]} == {"G235H", "G395H"}


def test_tag_header(tmp_path: Path):
    src = synthetic_image(tmp_path / "jw00001001001_01101_00001_nrs1_cal.fits")
    (out,) = run_step(tag_header, [src], tmp_path, params={"keyword": "PROJECT", "value": "x"})
    assert out.name == "jw00001001001_01101_00001_nrs1_tagged.fits"
    from astropy.io import fits

    assert fits.getheader(out)["PROJECT"] == "x"
