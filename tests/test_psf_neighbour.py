"""subtract_psf_neighbour: the fitting core on synthetic cubes, the step end to end, and the orientation of
PSF libraries it relies on. Everything offline: Gaussian libraries stand in for stpsf."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jwstflow.cubes.psf import PsfCubeProduct
from jwstflow.cubes.psf import neighbour as nb
from jwstflow.testing import make_context, run_step, synthetic_cube

PIX = 0.13          # spaxel [arcsec]
SHAPE = (44, 40)    # (ny, nx)


def _library(fwhm_arcsec=(0.27, 0.30), waves=(4.9, 5.5), scale=0.0325, npix=301, elongation=1.0,
             wing=0.0) -> PsfCubeProduct:
    """A Gaussian library (slices sum to 1 inside the field); ``elongation`` stretches +y, ``wing`` adds
    that share of the light in a Moffat-like halo (beta 1.5) -- real PSFs reach far beyond their core."""
    c = (npix - 1) / 2
    yy, xx = np.indices((npix, npix))
    planes = []
    for fwhm in fwhm_arcsec:
        s = fwhm / 2.3548 / scale
        plane = np.exp(-((xx - c) ** 2 / (2 * s**2) + (yy - c) ** 2 / (2 * (s * elongation) ** 2)))
        halo = (1.0 + ((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * s) ** 2) ** -1.5
        plane = (1.0 - wing) * plane / plane.sum() + wing * halo / halo.sum()
        planes.append(plane / plane.sum())
    meta = {"JWFPSFFR": ("ideal", ""), "JWFPSFPA": (0.0, "")}
    return PsfCubeProduct(np.stack(planes), np.asarray(waves, float), scale, meta=meta)


def _scene(psf: PsfCubeProduct, nwave: int = 24, neighbour=(30.3, 9.7), target=(14.0, 26.0), noise=0.02, seed=1):
    """A cube with a bright neighbour (the library itself, amplitude rising with wavelength), a faint
    extended target, a constant background and Gaussian noise; returns (data, err, wave, truth)."""
    rng = np.random.default_rng(seed)
    wave = np.linspace(4.9, 5.5, nwave)
    amp = 4000.0 * (wave / 5.0) ** 2                                    # [cube units x spaxel]
    (comp,) = nb.components([psf], SHAPE, *neighbour, wave, pixel_arcsec=PIX, rotation_deg=0.0)
    yy, xx = np.indices(SHAPE, dtype=float)
    disk = np.exp(-(((xx - target[0]) / 2.5) ** 2 + ((yy - target[1]) / 1.2) ** 2) / 2)
    target_cube = 30.0 * disk[None] * np.ones((nwave, 1, 1))
    data = amp[:, None, None] * comp + target_cube + 0.5
    err = np.full_like(data, noise)
    data = data + rng.normal(0.0, noise, data.shape)
    return data, err, wave, {"amp": amp, "comp": comp, "target": target_cube, "neighbour": neighbour, "target_xy": target}


def _protect(target, radius=12.0):
    yy, xx = np.indices(SHAPE, dtype=float)
    return np.hypot(xx - target[0], yy - target[1]) <= radius


# --------------------------------------------------------------------------- placing the library
def test_placed_matches_resampled_and_conserves_the_light():
    psf = _library(elongation=1.6)
    for rotation in (0.0, 30.0):
        on_grid = psf.placed((41, 41), 20.0, 20.0, PIX, rotation_deg=rotation)
        resampled = psf.resampled(PIX, 41, rotation_deg=rotation)
        np.testing.assert_allclose(on_grid, resampled.data, rtol=1e-6, atol=1e-9)   # one convention
    for sampling in ("point", "spaxel"):
        placed = psf.placed(SHAPE, 17.3, 21.6, PIX, sampling=sampling)[0]
        assert placed.sum() == pytest.approx(1.0, abs=0.01)                      # the share of all the light
        yy, xx = np.indices(SHAPE)
        assert (placed * xx).sum() / placed.sum() == pytest.approx(17.3, abs=0.02)
        assert (placed * yy).sum() / placed.sum() == pytest.approx(21.6, abs=0.02)
    outside = psf.placed(SHAPE, -6.0, 20.0, PIX)[0]                             # the core off the grid: its wing
    assert 0 < outside.sum() < 0.05


# --------------------------------------------------------------------------- the fit
@pytest.mark.parametrize("background", ["constant", "plane"])
def test_fit_recovers_the_neighbour_to_a_percent(background):
    psf = _library()
    data, err, _wave, truth = _scene(psf)
    protect = _protect(truth["target_xy"])
    fit = nb.fit_planes(data, err, [truth["comp"]], ~protect, background=background,
                        origin=truth["target_xy"], model_error=0.05)
    np.testing.assert_allclose(fit.amplitude, truth["amp"], rtol=0.01)
    np.testing.assert_allclose(fit.const, 0.5, atol=0.01)
    residual = data - fit.model([truth["comp"]]) - fit.background(SHAPE)
    # the target comes back as it was put in
    np.testing.assert_allclose(residual[:, protect].sum(axis=1), truth["target"][:, protect].sum(axis=1), rtol=0.01)


def test_fit_from_the_wing_of_a_neighbour_outside_the_field():
    psf = _library(wing=0.2)
    data, err, _wave, truth = _scene(psf, neighbour=(46.0, 6.0), noise=0.005)     # centre 6 px beyond the edge
    assert truth["comp"].sum(axis=(1, 2)).max() < 0.05                         # a few % of its light in the field
    fit = nb.fit_planes(data, err, [truth["comp"]], ~_protect(truth["target_xy"]), origin=truth["target_xy"])
    np.testing.assert_allclose(fit.amplitude, truth["amp"], rtol=0.01)


def test_control_ring_is_zero_after_a_correct_subtraction():
    psf = _library(wing=0.2)
    data, err, _wave, truth = _scene(psf)
    protect = _protect(truth["target_xy"])
    x0, y0 = truth["neighbour"]
    distance = float(np.hypot(truth["target_xy"][0] - x0, truth["target_xy"][1] - y0))
    ring = nb.ring_mask(SHAPE, (x0, y0), distance, 1.5) & ~protect
    fit = nb.fit_planes(data, err, [truth["comp"]], ~protect & ~ring, origin=truth["target_xy"])
    residual = data - fit.model([truth["comp"]]) - fit.background(SHAPE)
    mean, error, n = nb.ring_residual(residual, err, ring)
    before, _, _ = nb.ring_residual(data - 0.5, err, ring)
    assert n.min() > 20
    assert abs(np.mean(mean)) < 3 * np.mean(error) / np.sqrt(len(mean))
    assert np.mean(before) > 10 * np.mean(error) / np.sqrt(len(before))            # there was light to remove


def test_locate_fits_the_position_and_keeps_a_given_one_outside():
    psf = _library()
    data, err, wave, truth = _scene(psf)
    protect = _protect(truth["target_xy"])
    found = nb.locate(data, err, wave, psf, (30.7, 9.4), pixel_arcsec=PIX, rotation_deg=0.0, exclude=protect,
                      fwhm_pix=2.1, max_shift=3.0, n_blocks=3)
    assert found.refined and found.coverage > 0.9
    assert found.x == pytest.approx(30.3, abs=0.05) and found.y == pytest.approx(9.7, abs=0.05)
    outside = nb.locate(data, err, wave, psf, (46.0, 6.0), pixel_arcsec=PIX, rotation_deg=0.0, exclude=protect,
                        fwhm_pix=2.1, max_shift=3.0)
    assert not outside.refined and (outside.x, outside.y) == (46.0, 6.0)


def test_wing_profile_measures_what_the_model_lacks():
    nodes = nb.radial_nodes(2.0, 25.0)
    tents = nb.radial_tents(SHAPE, 20.0, 20.0, nodes, 25.0)
    np.testing.assert_allclose(tents.sum(axis=(1, 2)), 1.0)
    truth = np.linspace(0.02, -0.01, len(nodes))
    image = np.einsum("k,kyx->yx", truth, tents)
    coeffs = nb.fit_wing(image, np.full(SHAPE, 1e-3), tents, np.ones(SHAPE, bool))
    np.testing.assert_allclose(coeffs, truth, atol=1e-9)
    shape = nb.wing_shape(np.stack([truth, 2 * truth]), np.array([5.0, 6.0]), np.array([5.0, 5.5, 6.0]), tents)
    np.testing.assert_allclose(shape[1], 1.5 * image)


# --------------------------------------------------------------------------- the step, end to end
def _synthetic_pair(tmp_path: Path):
    """A MIRI ch1-short cube holding the scene of _scene, its Gaussian psf_cube library, and the
    neighbour's (ra, dec)."""
    from astropy.io import fits
    from astropy.wcs import WCS

    cube = synthetic_cube(tmp_path / "jw01234-o001_t001_miri_ch1-short_s3d.fits", instrument="MIRI", nwave=24,
                          size=40, wave_min=4.9, wave_step=0.6 / 23, pix_arcsec=PIX, PA_APER=0.0)
    lib_out = run_step("psf_cube", [cube], tmp_path / "lib", params={"engine": "gaussian", "fov_arcsec": 12.0,
                                                                      "position_angle_deg": 0.0})
    psf = PsfCubeProduct.read(lib_out[0])
    with fits.open(cube) as hdul:
        wcs = WCS(hdul["SCI"].header).celestial
    rng = np.random.default_rng(3)
    nwave, ny, nx = 24, 40, 40
    wave = 4.9 + np.arange(nwave) * 0.6 / 23
    amp = 4000.0 * (wave / 5.0) ** 2
    neighbour, target = (31.2, 8.6), (20.0, 20.0)
    (comp,) = nb.components([psf], (ny, nx), *neighbour, wave, pixel_arcsec=PIX, rotation_deg=0.0)
    yy, xx = np.indices((ny, nx), dtype=float)
    target_img = 30.0 * np.exp(-(((xx - target[0]) / 2.0) ** 2 + ((yy - target[1]) / 1.0) ** 2) / 2)
    data = amp[:, None, None] * comp + target_img[None] + 0.5 + rng.normal(0, 0.02, (nwave, ny, nx))
    with fits.open(cube, mode="update") as hdul:
        hdul["SCI"].data = data.astype("f4")
        hdul["ERR"].data = np.full(data.shape, 0.02, "f4")
        hdul[0].header["TARG_RA"], hdul[0].header["TARG_DEC"] = (float(v) for v in wcs.all_pix2world(*target, 0))
    ra, dec = (float(v) for v in wcs.all_pix2world(*neighbour, 0))
    return cube, lib_out[0].parent, (ra, dec), amp, target_img, psf


def test_subtract_psf_neighbour_end_to_end(tmp_path: Path):
    from astropy.io import fits

    from jwstflow.cubes.psf import SubtractPsfNeighbour
    from jwstflow.cubes.psf.subtraction import BLANKED, PsfSubtraction
    from jwstflow.testing import check_step

    assert check_step(SubtractPsfNeighbour)[1] == []
    cube, lib_dir, (ra, dec), amp, target_img, _psf = _synthetic_pair(tmp_path)
    ctx = make_context(tmp_path / "run", stage="subtract_psf_neighbour")
    ctx.stage_dirs["psf_cube"] = lib_dir
    # the given position is 0.05" off: the step finds the star
    outputs = run_step(SubtractPsfNeighbour, [cube], tmp_path / "run", ctx=ctx,
                       params={"ra": ra + 0.05 / 3600 / np.cos(np.deg2rad(dec)), "dec": dec})
    cleaned, record = outputs
    assert cleaned.name == cube.name and record.name.endswith("_nbrfit.fits")
    rec = PsfSubtraction.read(record)
    area = (PIX / 206264.806) ** 2 * 1e6
    np.testing.assert_allclose(rec.planes["FLUX_PSF"] / area, amp, rtol=0.01)
    assert rec.header["JWFNBREF"] and rec.header["JWFNBX"] == pytest.approx(31.2, abs=0.05)
    ring = np.asarray(rec.planes["RING"], float)
    assert abs(ring.mean()) < 3 * ring.std() / np.sqrt(len(ring))               # nothing left at the target's distance
    with fits.open(cleaned) as hdul:
        sci, dq = hdul["SCI"].data, hdul["DQ"].data
        assert hdul[0].header["JWFNBSUB"] and "subtract_psf_neighbour" in str(hdul[0].header["HISTORY"])
        names = [h.name for h in hdul]
    with fits.open(cube) as hdul:
        assert names == [h.name for h in hdul]                                  # the s3d format kept
    blanked = rec.mask(BLANKED)
    assert blanked.any() and np.isnan(sci[:, blanked]).all() and (dq[:, blanked] & 1).all()
    yy, xx = np.indices(target_img.shape)
    near = np.hypot(xx - 20, yy - 20) <= 5
    target_back = np.nanmedian(sci, axis=0)[near] - 0.5
    assert target_back.sum() == pytest.approx(target_img[near].sum(), rel=0.01)  # the target, free of the star
    from jwstflow.qa.steps.psf import QaPsfNeighbour

    figures = run_step(QaPsfNeighbour, [record], tmp_path / "run")
    assert len(figures) == 1 and figures[0].stat().st_size > 0


def test_subtract_psf_neighbour_needs_a_library(tmp_path: Path):
    from jwstflow.cubes.psf import SubtractPsfNeighbour

    cube = synthetic_cube(tmp_path / "jw01234-o001_t001_miri_ch1-short_s3d.fits", instrument="MIRI")
    with pytest.raises(FileNotFoundError, match="psf_cube"):
        run_step(SubtractPsfNeighbour, [cube], tmp_path, params={"ra": 10.0, "dec": -20.0, "psf_stage": ""})
    with pytest.raises(ValueError):
        SubtractPsfNeighbour.validate_params({})                                 # the neighbour must be named


# --------------------------------------------------------------------------- the orientation of a library
def test_psf_cube_records_the_pa_of_the_frame_it_was_computed_in():
    from jwstflow.cubes.psf.instrument import CubeGeometry, IfuConfig
    from jwstflow.cubes.psf.steps import _frame_position_angle

    class Engine:
        def __init__(self, angle):
            self.frame_v3_angle = angle

    def geometry(pa, yangle, instrument="MIRI"):
        config = IfuConfig("MIRI", channel="1", band="SHORT") if instrument == "MIRI" else IfuConfig("NIRSPEC")
        return CubeGeometry(config, np.array([5.0]), (10, 10), 0.13, pa, None, None,
                            aperture_name="MIRIFU_CHANNEL1A", v3_yangle_deg=yangle)

    # MRS: the SIAF gives the channel apertures no V3IdlYAngle, stpsf computes ~8 deg from V3
    assert _frame_position_angle(geometry(16.18, 0.0), Engine(8.42), Path("c.fits")) == pytest.approx(24.60)
    # NIRSpec: PA_APER already holds the aperture's angle, which is stpsf's frame
    assert _frame_position_angle(geometry(273.67, 138.97, "NIRSPEC"), Engine(138.97), Path("c.fits")) \
        == pytest.approx(273.67)
    # circular engines have no frame: PA_APER as before
    assert _frame_position_angle(geometry(16.18, 0.0), Engine(None), Path("c.fits")) == pytest.approx(16.18)
