"""image_astrometry and qa_image_astrometry on synthetic mosaics with injected offsets; Gaia mocked."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from jwstflow.images import band_of, read_image
from jwstflow.images.astrometry import (
    GaiaQueryError,
    ImageAstrometry,
    catalog_path,
    observation_mjd,
    propagate,
    robust_offset,
    sky_offsets_mas,
)
from jwstflow.images.astrometry import gaia as gaia_module
from jwstflow.images.astrometry.gaia import normalise
from jwstflow.testing import check_step, make_context, run_step, synthetic_i2d

RA0, DEC0 = 65.0, 28.0
MAS = 1.0 / 3.6e6
EPOCH_MJD = 60310.5          # J2024.0: eight years after the Gaia epoch
PM_SCATTER = 30.0            # [mas/yr] proper motions of the synthetic stars


def _stars(seed: int = 7) -> dict[str, np.ndarray]:
    """36 stars on a jittered 1.6 arcsec grid (at least 1 arcsec apart), with proper motions; the last
    four have none in "Gaia" (two-parameter solutions) and do not move."""
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(6) - 2.5, np.arange(6) - 2.5)
    x = gx.ravel() * 1.6 + rng.uniform(-0.3, 0.3, 36)
    y = gy.ravel() * 1.6 + rng.uniform(-0.3, 0.3, 36)
    pmra, pmdec = rng.normal(0, PM_SCATTER, 36), rng.normal(0, PM_SCATTER, 36)
    pmra[-4:] = pmdec[-4:] = 0.0
    dec = DEC0 + y / 3600.0
    ra = RA0 + x / 3600.0 / np.cos(np.radians(dec))
    return {"ra": ra, "dec": dec, "pmra": pmra, "pmdec": pmdec, "id": np.arange(36, dtype=np.int64) + 10_000}


def _gaia_table(stars: dict[str, np.ndarray]):
    """The stars as Gaia DR3 has them: at J2016.0, eight years before the images."""
    from astropy.table import Table

    dt = 2024.0 - 2016.0
    dec = stars["dec"] - stars["pmdec"] * dt * MAS
    ra = stars["ra"] - stars["pmra"] * dt * MAS / np.cos(np.radians(dec))
    pmra, pmdec = stars["pmra"].copy(), stars["pmdec"].copy()
    pmra[-4:] = pmdec[-4:] = np.nan
    return Table({"source_id": stars["id"], "ra": ra, "dec": dec, "ra_error": np.full(36, 0.1),
                  "dec_error": np.full(36, 0.1), "pmra": pmra, "pmdec": pmdec, "ref_epoch": np.full(36, 2016.0),
                  "phot_g_mean_mag": np.full(36, 19.0)})


def _mosaic(directory: Path, name: str, stars: dict[str, np.ndarray], *, instrument: str, filter: str, pupil: str | None,
            pix: float, offset: tuple[float, float], noise: float, seed: int, drop: int = 0,
            blank_first: bool = False) -> Path:
    """An i2d of a 12 x 12 arcsec field and its _cat.ecsv: the stars displaced by ``offset`` [mas] (image
    minus truth at J2024.0) plus centroid noise ``noise`` [mas], ``drop`` of them missing; two extended
    sources, two faint ones and a close pair (cut by isolation) besides. ``blank_first`` blanks the core
    of the first star (NaN, as a saturated core)."""
    from astropy.coordinates import SkyCoord
    from astropy.table import Table

    rng = np.random.default_rng(seed)
    n = round(12.0 / pix)
    data = rng.normal(0.0, 0.01, (n, n))
    path = synthetic_i2d(directory / f"{name}_i2d.fits", data=data, instrument=instrument, filter=filter, pupil=pupil,
                         shape=(n, n), pix_arcsec=pix, ra=RA0, dec=DEC0, EXPSTART=EPOCH_MJD - 0.1, EXPEND=EPOCH_MJD + 0.1)
    keep = np.arange(drop, 36)                   # the stars without a Gaia proper motion are the last four
    dra = offset[0] + rng.normal(0, noise, keep.size)
    ddec = offset[1] + rng.normal(0, noise, keep.size)
    dec = stars["dec"][keep] + ddec * MAS
    ra = stars["ra"][keep] + dra * MAS / np.cos(np.radians(dec))
    # the others: two extended, two faint, a close pair (0.2 FWHM apart) -- none may be used
    extra_x = np.array([-5.0, 5.0, -5.0, 5.0, 0.0, 0.0])
    extra_y = np.array([-5.0, -5.0, 5.0, 5.0, 4.8, 4.8])
    fwhm = band_of(instrument, filter, pupil).fwhm_arcsec
    extra_x[5] += 0.2 * fwhm
    extra_dec = DEC0 + extra_y / 3600.0
    extra_ra = RA0 + extra_x / 3600.0 / np.cos(np.radians(extra_dec))
    ra, dec = np.concatenate([ra, extra_ra]), np.concatenate([dec, extra_dec])
    flux = np.concatenate([np.full(keep.size, 100.0), [100.0, 100.0, 4.0, 4.0, 100.0, 100.0]])
    extended = np.concatenate([np.zeros(keep.size, bool), [True, True, False, False, False, False]])
    image = read_image(path)
    x, y = image.wcs.all_world2pix(ra, dec, 0)
    if blank_first:
        from astropy.io import fits

        with fits.open(path, mode="update") as hdul:
            iy, ix = round(y[0]), round(x[0])
            hdul["SCI"].data[iy - 1:iy + 2, ix - 1:ix + 2] = np.nan
    table = Table({"label": np.arange(1, ra.size + 1), "xcentroid": x, "ycentroid": y,
                   "sky_centroid": SkyCoord(ra, dec, unit="deg"), "aper_total_flux": flux,
                   "aper_total_flux_err": np.ones(ra.size), "is_extended": extended})
    table.write(catalog_path(path), format="ascii.ecsv", overwrite=True)
    return path


#: Injected offsets [mas], image minus truth (Gaia at the image epoch), and centroid noise.
INJECTED = {"F150W": ((4.0, -3.0), 3.0), "F444W": ((24.0, -13.0), 6.0), "F770W": ((-36.0, 27.0), 15.0)}


def _run_set(directory: Path, *, f444w_offset: tuple[float, float] | None = None) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    stars = _stars()
    off444 = f444w_offset or INJECTED["F444W"][0]
    return [
        _mosaic(directory, "jw01234-o001_t001_nircam_clear-f150w", stars, instrument="NIRCAM", filter="F150W", pupil="CLEAR",
                pix=0.031, offset=INJECTED["F150W"][0], noise=INJECTED["F150W"][1], seed=1, blank_first=True),
        _mosaic(directory, "jw01234-o001_t001_nircam_f444w-clear", stars, instrument="NIRCAM", filter="F444W", pupil="CLEAR",
                pix=0.063, offset=off444, noise=INJECTED["F444W"][1], seed=2, drop=3),
        _mosaic(directory, "jw01234-o001_t001_miri_f770w", stars, instrument="MIRI", filter="F770W", pupil=None, pix=0.11,
                offset=INJECTED["F770W"][0], noise=INJECTED["F770W"][1], seed=3, drop=5),
    ]


@pytest.fixture
def mock_gaia(monkeypatch):
    """query_gaia answered from the synthetic stars; counts its calls."""
    calls = []

    def fake(ra, dec, radius, *, services=gaia_module.SERVICES):
        calls.append((ra, dec, radius))
        return normalise(_gaia_table(_stars())), "mock"

    monkeypatch.setattr(gaia_module, "query_gaia", fake)
    return calls


def _tables(product: Path):
    from astropy.io import fits
    from astropy.table import Table

    with fits.open(product) as hdul:
        return hdul[0].header.copy(), *(Table(hdul[n].data) for n in ("IMAGES", "SUMMARY", "MATCHES"))


def _row(summary, band: str, against: str):
    (row,) = [r for r in summary if r["band"].strip() == band and r["against"].strip() == against]
    return row


def _within(row, expected: tuple[float, float], k: float = 3.0) -> None:
    for key, value in zip(("dra", "ddec"), expected, strict=True):
        assert abs(row[key] - value) < k * row[f"{key}_error"] + 0.05, (row["band"], row["against"], key, row[key], value,
                                                                         row[f"{key}_error"])


def test_injected_offsets_come_back_against_the_reference_image_and_gaia(tmp_path, mock_gaia):
    inputs = _run_set(tmp_path / "in")
    outs = run_step(ImageAstrometry, inputs, tmp_path, stage="image_astrometry")
    assert [p.name for p in outs] == ["jw01234-o001_t001_astrom.fits", "jw01234-o001_t001_gaiadr3.ecsv"]
    hdr, images, summary, matches = _tables(outs[0])
    assert hdr["JWFASRBD"] == "F150W" and hdr["JWFASGAI"] == "ok" and hdr["JWFASGSV"] == "mock" and len(mock_gaia) == 1
    assert list(images["band"]) == ["F150W", "F444W", "F770W"] and list(images["reference"]) == [True, False, False]
    f150 = images[0]
    # 36 stars + 6 others; 2 extended, 2 faint, the close pair, the blanked core go
    assert (f150["n_catalog"], f150["n_finite"], f150["n_compact"], f150["n_snr"], f150["n_isolated"], f150["n_point"]) == (
        42, 42, 40, 38, 36, 35)
    ref = INJECTED["F150W"][0]
    for band in ("F444W", "F770W"):
        inj = INJECTED[band][0]
        row = _row(summary, band, "reference")
        assert row["n_used"] >= 25 and row["against_name"].strip() == "F150W"
        _within(row, (inj[0] - ref[0], inj[1] - ref[1]))
        # the scatter is the centroid noise of both images in quadrature
        expected = np.hypot(INJECTED[band][1], INJECTED["F150W"][1])
        assert 0.5 * expected < row["dra_scatter"] < 1.8 * expected
    for band, (inj, _) in INJECTED.items():
        row = _row(summary, band, "gaia")
        _within(row, inj)
        gaia_rows = matches[(np.char.strip(matches["band"]) == band) & (np.char.strip(matches["against"]) == "gaia")]
        no_pm = ~gaia_rows["ref_has_pm"].astype(bool)
        assert no_pm.any() and not gaia_rows["used"][no_pm].any()       # matched, recorded, not in the median
        assert row["n_used"] == gaia_rows["used"].sum() >= row["n_matched"] - no_pm.sum() - 2
    # sign convention: image minus reference
    row = _row(summary, "F770W", "gaia")
    assert row["dra"] < 0 < row["ddec"]


def test_proper_motions_are_needed_to_match_gaia(tmp_path, mock_gaia):
    stars = _stars()
    gaia = _gaia_table(stars)
    ra, dec, has_pm = propagate(normalise(gaia), 2024.0)
    dra, ddec = sky_offsets_mas(ra, dec, stars["ra"], stars["dec"])
    assert has_pm.sum() == 32 and np.max(np.hypot(dra, ddec)[has_pm]) < 1e-3
    still = np.hypot(*sky_offsets_mas(gaia["ra"], gaia["dec"], stars["ra"], stars["dec"]))
    assert np.median(still[:32]) > 150.0                  # without propagation the stars sit 0.15-0.5 arcsec off


def test_a_shift_larger_than_the_match_radius_is_found(tmp_path):
    inputs = _run_set(tmp_path / "in", f444w_offset=(304.0, -203.0))
    outs = run_step(ImageAstrometry, inputs, tmp_path, stage="image_astrometry", params={"gaia": False})
    assert [p.name for p in outs] == ["jw01234-o001_t001_astrom.fits"]
    hdr, _, summary, _ = _tables(outs[0])
    assert hdr["JWFASGAI"] == "off"
    row = _row(summary, "F444W", "reference")
    assert row["radius"] * 1e3 < 200.0 and row["n_used"] >= 25          # 1 FWHM of F444W: 0.15 arcsec
    _within(row, (300.0, -200.0))
    assert abs(row["shift_dra"] - 300.0) < 30.0 and abs(row["shift_ddec"] + 200.0) < 30.0


def test_the_gaia_cache_is_reused_while_it_covers_the_footprint(tmp_path, mock_gaia):
    inputs = _run_set(tmp_path / "in")
    ctx = make_context(tmp_path, stage="image_astrometry")
    first = _tables(run_step(ImageAstrometry, inputs, tmp_path, ctx=ctx)[0])
    second = run_step(ImageAstrometry, inputs, tmp_path, ctx=ctx)
    assert len(mock_gaia) == 1 and first[0]["JWFASGAI"] == "ok" and _tables(second[0])[0]["JWFASGAI"] == "cached"
    np.testing.assert_allclose(first[2]["dra"], _tables(second[0])[2]["dra"])
    # a cache of a smaller circle does not cover the footprint: asked again
    from astropy.table import Table

    cache = Table.read(second[1], format="ascii.ecsv")
    cache.meta["gaia_query"]["radius"] = 1e-4
    cache.write(second[1], format="ascii.ecsv", overwrite=True)
    third = run_step(ImageAstrometry, inputs, tmp_path, ctx=ctx)
    assert len(mock_gaia) == 2 and _tables(third[0])[0]["JWFASGAI"] == "ok"


def test_a_failed_gaia_query_leaves_the_relative_comparison(tmp_path, monkeypatch):
    def down(*args, **kwargs):
        raise ConnectionError("archive down for maintenance")

    monkeypatch.setattr(gaia_module, "_query_esa", down)
    monkeypatch.setattr(gaia_module, "_query_stsci", down)
    outs = run_step(ImageAstrometry, _run_set(tmp_path / "in"), tmp_path, stage="image_astrometry")
    hdr, _, summary, _ = _tables(outs[0])
    assert [p.suffix for p in outs] == [".fits"] and hdr["JWFASGAI"] == "failed" and hdr["JWFASGER"].startswith("esa: ConnectionError")
    assert set(np.char.strip(summary["against"])) == {"reference"} and len(summary) == 2


def test_the_next_service_answers_when_the_first_fails(monkeypatch):
    def down(*args, **kwargs):
        raise ConnectionError("down")

    monkeypatch.setattr(gaia_module, "_query_esa", down)
    monkeypatch.setattr(gaia_module, "_query_stsci", lambda ra, dec, radius: _gaia_table(_stars()))
    table, service = gaia_module.query_gaia(RA0, DEC0, 0.01)
    assert service == "stsci" and len(table) == 36 and table["pmra"].unit == "mas / yr"
    with pytest.raises(GaiaQueryError, match="esa: ConnectionError"):
        gaia_module.query_gaia(RA0, DEC0, 0.01, services=["esa"])


def test_the_reference_can_be_named_and_must_exist(tmp_path):
    inputs = _run_set(tmp_path / "in")
    outs = run_step(ImageAstrometry, inputs, tmp_path, stage="image_astrometry", params={"gaia": False, "reference": "f444w"})
    hdr, images, summary, _ = _tables(outs[0])
    assert hdr["JWFASRBD"] == "F444W" and list(images["reference"]) == [False, True, False]
    _within(_row(summary, "F150W", "reference"), (4.0 - 24.0, -3.0 + 13.0))
    with pytest.raises(ValueError, match="neither the band nor the file"):
        run_step(ImageAstrometry, inputs, tmp_path, stage="image_astrometry", params={"gaia": False, "reference": "F200W"})


def test_robust_offset_clips_outliers_and_reports_the_error_of_the_median():
    rng = np.random.default_rng(11)
    dra, ddec = rng.normal(5.0, 2.0, 400), rng.normal(-3.0, 2.0, 400)
    dra[:20] += 80.0                                    # mismatches
    st = robust_offset(dra, ddec)
    assert st.n_used <= 380 and abs(st.dra - 5.0) < 0.4 and abs(st.ddec + 3.0) < 0.4
    sigma = max(st.dra_scatter, st.dra_std)
    assert 1.7 < st.dra_scatter < 2.3 and 1.7 < st.dra_std < 2.3
    assert st.dra_error == pytest.approx(1.2533 * sigma / np.sqrt(st.n_used), rel=1e-3)
    few = robust_offset(np.array([1.0, 2.0]), np.array([0.0, 1.0]))
    assert few.dra == 1.5 and np.isnan(few.dra_scatter) and np.isnan(few.dra_error)
    # three matches, two of them alike: the MAD collapses, the error follows the standard deviation;
    # and with fewer than five nothing is clipped
    three = robust_offset(np.array([10.0, 10.1, 30.0]), np.array([0.0, 0.0, 0.0]))
    assert three.n_used == 3 and three.dra_scatter < 0.2 and three.dra_error > 5.0


def test_the_epoch_is_the_middle_of_the_exposures(tmp_path):
    path = synthetic_i2d(tmp_path / "x_i2d.fits", EXPSTART=60000.0, EXPEND=60001.0)
    assert observation_mjd(read_image(path).primary) == 60000.5
    bare = synthetic_i2d(tmp_path / "y_i2d.fits")                        # DATE-OBS only
    assert observation_mjd(read_image(bare).primary) == pytest.approx(60310.0)


def test_qa_image_astrometry_draws_the_product(tmp_path, mock_gaia):
    from jwstflow.qa.steps import QaImageAstrometry

    (product, _) = run_step(ImageAstrometry, _run_set(tmp_path / "in"), tmp_path, stage="image_astrometry")
    (png,) = run_step(QaImageAstrometry, [product], tmp_path, stage="qa_image_astrometry")
    assert png.name == "jw01234-o001_t001_astrom.png" and png.stat().st_size > 40_000
    (off,) = run_step(ImageAstrometry, _run_set(tmp_path / "in2"), tmp_path / "b", stage="image_astrometry",
                      params={"gaia": False})
    (png2,) = run_step(QaImageAstrometry, [off], tmp_path / "b", stage="qa_image_astrometry", params={"ncols": 2})
    assert png2.stat().st_size > 30_000
    # one image, against Gaia only: one panel, one point per wavelength panel
    single = run_step(ImageAstrometry, _run_set(tmp_path / "in3")[1:2], tmp_path / "c", stage="image_astrometry")
    (png3,) = run_step(QaImageAstrometry, [single[0]], tmp_path / "c", stage="qa_image_astrometry")
    assert png3.stat().st_size > 20_000


def test_both_steps_pass_check_step_by_name():
    for name in ("image_astrometry", "qa_image_astrometry"):
        _, problems = check_step(name)
        assert problems == [], (name, problems)


def test_an_image_without_a_catalogue_is_listed_but_not_compared(tmp_path, monkeypatch):
    from jwstflow.qa.steps import QaImageAstrometry

    def down(*args, **kwargs):
        raise ConnectionError("down")

    monkeypatch.setattr(gaia_module, "_query_esa", down)
    monkeypatch.setattr(gaia_module, "_query_stsci", down)
    inputs = _run_set(tmp_path / "in")
    catalog_path(inputs[2]).unlink()                                  # F770W without its _cat.ecsv
    (product,) = run_step(ImageAstrometry, inputs, tmp_path, stage="image_astrometry")
    hdr, images, summary, _ = _tables(product)
    assert hdr["JWFASGAI"] == "failed" and images["catalog"][2].strip() == "" and images["n_catalog"][2] == 0
    assert list(np.char.strip(summary["band"])) == ["F444W", "F770W"] and summary["n_matched"][1] == 0
    # the figure of a product without Gaia, one comparison empty
    (png,) = run_step(QaImageAstrometry, [product], tmp_path, stage="qa_image_astrometry", params={"details": True})
    assert png.stat().st_size > 30_000


def test_a_product_without_comparisons_has_no_figure(tmp_path):
    from jwstflow.qa.steps import QaImageAstrometry

    (one,) = _run_set(tmp_path / "in")[:1]
    (product,) = run_step(ImageAstrometry, [one], tmp_path, stage="image_astrometry", params={"gaia": False})
    assert product.name == "jw01234-o001_t001_nircam_clear-f150w_astrom.fits" and len(_tables(product)[2]) == 0
    assert run_step(QaImageAstrometry, [product], tmp_path, stage="qa_image_astrometry") == []
