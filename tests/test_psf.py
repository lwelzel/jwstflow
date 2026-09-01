"""The psfcube-product contract and the psf_cube / qa_psf_cube steps.

Everything runs offline: the Gaussian engine covers the product path end to
end, and a stub of the (verified) stpsf 2.x IFU API pins how `StpsfEngine`
must drive the real package -- configuration order (the pixel scale must be
set *after* band selection, which resets it), the calc_psf arguments, and the
DET_DIST extension choice.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from jwstflow.psf import (
    GaussianPsfEngine,
    IfuConfig,
    PsfCubeProduct,
    PsfGrid,
    StpsfEngine,
    approx_fwhm_arcsec,
    measure_fwhm_arcsec,
    model_pixel_scale,
    psf_from_stage,
    read_cube_geometry,
    rotate_cube_to_sky,
    subsample_wavelengths,
)
from jwstflow.testing import check_step, make_context, run_step, synthetic_cube

MIRI_1A = IfuConfig("MIRI", channel="1", band="SHORT")
NRS_G235H = IfuConfig("NIRSPEC", grating="G235H", filter="F170LP")


# --------------------------------------------------------------------------- sampling & grids
def test_subsample_endpoints_and_spacing():
    waves = subsample_wavelengths(4.9, 5.74, max_fractional_step=0.02)
    assert waves[0] == pytest.approx(4.9) and waves[-1] == pytest.approx(5.74)
    ratios = waves[1:] / waves[:-1]
    assert np.all(ratios <= 1.02 + 1e-9) and np.all(np.diff(waves) > 0)
    assert len(waves) >= 5


def test_subsample_overrides():
    assert len(subsample_wavelengths(1.0, 1.01, max_fractional_step=0.02)) == 5   # min_n floor
    assert len(subsample_wavelengths(2.87, 5.27, n=12)) == 12
    assert subsample_wavelengths(3.0, 3.0).tolist() == [3.0]
    with pytest.raises(ValueError):
        subsample_wavelengths(-1.0, 2.0)


def test_model_grid_convention():
    # at 1 pc, 1 AU subtends 1 arcsec: 256 px over 600 AU -> 2.34375 arcsec/px
    assert model_pixel_scale(1.0) == pytest.approx(600.0 / 256)
    assert model_pixel_scale(190.0) == pytest.approx(600.0 / 256 / 190.0)
    grid = PsfGrid.for_model(190.0, npix=256)
    assert grid.fov_arcsec == pytest.approx(600.0 / 190.0)


def test_ifu_config_from_headers():
    assert IfuConfig.from_header({"INSTRUME": "MIRI", "CHANNEL": "2", "BAND": "MEDIUM"}).mrs_band == "2B"
    assert MIRI_1A.label == "1A" and NRS_G235H.label == "G235H/F170LP"
    with pytest.raises(ValueError, match="output_type"):
        _ = IfuConfig.from_header({"INSTRUME": "MIRI", "CHANNEL": "12", "BAND": "SHORT"}).mrs_band
    with pytest.raises(ValueError, match="NIRSpec IFU and MIRI MRS"):
        IfuConfig.from_header({"INSTRUME": "NIRCAM"})


# --------------------------------------------------------------------------- engine & measurement
def test_gaussian_engine_widths_and_normalization():
    grid = PsfGrid(0.02, 96)
    waves = np.array([5.0, 7.0])
    stack = GaussianPsfEngine().compute(MIRI_1A, waves, grid)
    assert stack.shape == (2, 96, 96)
    assert np.nansum(stack[0]) == pytest.approx(1.0, abs=0.02)
    for plane, lam in zip(stack, waves):
        measured = measure_fwhm_arcsec(plane, grid.pixelscale_arcsec)
        assert measured == pytest.approx(float(approx_fwhm_arcsec(MIRI_1A, lam)), rel=0.06)
    # wider at longer wavelength
    assert measure_fwhm_arcsec(stack[1], grid.pixelscale_arcsec) > measure_fwhm_arcsec(stack[0], grid.pixelscale_arcsec)


def _sky_pa_of_elongation(image: np.ndarray) -> float:
    """PA [deg, N->E] of the major axis in a north-up, east-left frame (+y = N, east = -x)."""
    y, x = np.indices(image.shape)
    w = image / image.sum()
    my, mx = (w * y).sum(), (w * x).sum()
    dy, dx = y - my, x - mx
    cyy, cxx, cxy = (w * dy * dy).sum(), (w * dx * dx).sum(), (w * dx * dy).sum()
    ang = 0.5 * np.arctan2(2 * cxy, cxx - cyy)   # major axis, from +x toward +y
    ux, uy = np.cos(ang), np.sin(ang)
    if uy < 0:
        ux, uy = -ux, -uy
    return float(np.degrees(np.arctan2(-ux, uy))) % 180


def test_rotation_places_plus_y_at_position_angle():
    n = 65
    c = (n - 1) / 2
    yy, xx = np.indices((n, n))
    along_y = np.exp(-(((yy - c) / 10.0) ** 2 + ((xx - c) / 3.0) ** 2) / 2)
    assert _sky_pa_of_elongation(along_y) == pytest.approx(0.0, abs=0.5)
    for pa in (30.0, 120.0):
        rotated = rotate_cube_to_sky(along_y[None], pa)[0]
        assert _sky_pa_of_elongation(rotated) == pytest.approx(pa, abs=0.5)
        assert rotated.sum() == pytest.approx(along_y.sum(), rel=1e-3)   # rotation about the kernel centre


# --------------------------------------------------------------------------- product: I/O, interpolation, convolution
def _gaussian_product(npix: int, sigmas_pix: list[float], waves: list[float], scale: float = 0.02) -> PsfCubeProduct:
    c = (npix - 1) / 2
    yy, xx = np.indices((npix, npix))
    r2 = (yy - c) ** 2 + (xx - c) ** 2
    data = np.stack([np.exp(-r2 / (2 * s**2)) / (2 * np.pi * s**2) for s in sigmas_pix])
    return PsfCubeProduct(data, np.asarray(waves), scale)


def test_product_roundtrip(tmp_path: Path):
    cube = synthetic_cube(tmp_path / "jw001_miri_ch1-short_s3d.fits", instrument="MIRI")
    product = _gaussian_product(32, [2.0, 3.0], [5.0, 5.5])
    product.meta["JWFPSFEN"] = ("gaussian", "PSF engine")
    out = product.write(tmp_path / "jw001_miri_ch1-short_psfcube.fits", like=cube)
    back = PsfCubeProduct.read(out)
    assert back.data.shape == (2, 32, 32)
    assert back.wavelengths.tolist() == [5.0, 5.5]
    assert back.pixelscale_arcsec == pytest.approx(0.02)
    assert back.meta["JWFPSFEN"] == "gaussian"
    assert back.header["INSTRUME"] == "MIRI"          # instrument keys travel from the source cube
    assert np.allclose(back.sums(), product.sums(), rtol=1e-5)


def test_interpolation_between_slices():
    product = _gaussian_product(64, [2.0, 4.0], [4.0, 5.0], scale=0.1)
    exact = product.at(4.0)
    assert np.allclose(exact, product.data[0] / product.data[0].sum())
    mid = product.at(4.5)
    assert mid.sum() == pytest.approx(1.0, abs=1e-6)   # normalized interpolation stays unit-sum
    f_lo, f_mid, f_hi = (measure_fwhm_arcsec(p, 0.1) for p in (product.at(4.0), mid, product.at(5.0)))
    assert f_lo < f_mid < f_hi
    assert np.allclose(product.at(3.0), product.at(4.0))   # clamped, never extrapolated
    assert np.allclose(product.at(9.0), product.at(5.0))


@pytest.mark.parametrize("npix", [31, 32])   # odd (pixel-centred) and even (half-pixel-centred) kernels
def test_convolution_preserves_position_and_flux(npix: int):
    product = _gaussian_product(npix, [2.5, 2.5], [4.0, 5.0])
    model = np.zeros((1, 48, 48))
    model[0, 30, 19] = 2.0                     # a 2-unit point source off centre
    out = product.convolve(model, [4.5])
    assert out.shape == model.shape
    assert out[0].sum() == pytest.approx(2.0, rel=1e-6)          # unit-sum kernel conserves flux
    y, x = np.indices(out[0].shape)
    w = out[0] / out[0].sum()
    assert (w * y).sum() == pytest.approx(30.0, abs=1e-6)        # no half-pixel displacement
    assert (w * x).sum() == pytest.approx(19.0, abs=1e-6)
    # the response reproduces the kernel shape (measured on a stamp centred on the source)
    stamp = out[0][30 - 14: 30 + 15, 19 - 14: 19 + 15]
    assert measure_fwhm_arcsec(stamp, product.pixelscale_arcsec) == pytest.approx(
        measure_fwhm_arcsec(product.at(4.5), product.pixelscale_arcsec), rel=0.05)


def test_convolution_handles_nans_and_2d():
    product = _gaussian_product(17, [1.5], [4.0])
    model = np.ones((8, 8))
    model[2, 2] = np.nan
    out = product.convolve(model, 4.0)
    assert out.shape == (8, 8) and np.isfinite(out).all()


# --------------------------------------------------------------------------- cube geometry
def test_read_cube_geometry_from_synthetic(tmp_path: Path):
    path = synthetic_cube(tmp_path / "jw001_nirspec_g235h-f170lp_s3d.fits", instrument="NIRSPEC",
                          nwave=30, wave_min=1.66, wave_step=0.01, PA_APER=42.5)
    geom = read_cube_geometry(path)
    assert geom.config.instrument == "NIRSPEC" and geom.config.label == "G235H/F170LP"
    assert geom.wavelengths[0] == pytest.approx(1.66) and len(geom.wavelengths) == 30
    assert geom.pixelscale_arcsec == pytest.approx(0.13, rel=1e-3)
    assert geom.position_angle_deg == pytest.approx(42.5)
    assert geom.date_obs is None            # optional: the synthetic datamodel carries no DATE-OBS


def test_read_cube_geometry_plain_fits_and_roll_ref(tmp_path: Path):
    from astropy.io import fits

    sci = fits.ImageHDU(np.zeros((4, 6, 6), "f4"), name="SCI")
    sci.header.update({"CRVAL3": 4.9, "CDELT3": 0.1, "CRPIX3": 1.0, "CDELT2": 0.13 / 3600,
                       "ROLL_REF": 40.0, "V3I_YANG": 8.2})
    primary = fits.PrimaryHDU()
    primary.header.update({"INSTRUME": "MIRI", "CHANNEL": "1", "BAND": "SHORT", "DATE-OBS": "2024-01-01"})
    path = tmp_path / "plain_s3d.fits"
    fits.HDUList([primary, sci]).writeto(path)
    geom = read_cube_geometry(path)
    assert geom.config.mrs_band == "1A"
    assert geom.wavelengths.tolist() == pytest.approx([4.9, 5.0, 5.1, 5.2])
    assert geom.position_angle_deg == pytest.approx(48.2)   # ROLL_REF + V3I_YANG fallback
    assert geom.date_obs == "2024-01-01"


# --------------------------------------------------------------------------- the steps
def test_step_declarations():
    from jwstflow.contrib.psf import PsfCube, QaPsfCube

    for cls in (PsfCube, QaPsfCube):
        _, problems = check_step(cls)
        assert problems == []


def test_psf_cube_step_requires_a_scale():
    from jwstflow.contrib.psf import PsfCube

    with pytest.raises(ValueError, match="distance_pc"):
        PsfCube.validate_params({"engine": "gaussian"})
    assert PsfCube.validate_params({"grid": "native", "engine": "gaussian"})["grid"] == "native"


def test_psf_cube_step_on_synthetic_miri(tmp_path: Path):
    src = synthetic_cube(tmp_path / "jw01751-o010_t005_miri_ch1-short_s3d.fits", instrument="MIRI",
                         nwave=40, wave_min=4.9, wave_step=0.02, PA_APER=30.0)
    outputs = run_step("jwstflow.contrib.psf:PsfCube", [src], tmp_path,
                       params={"engine": "gaussian", "distance_pc": 190.0, "npix": 64})
    assert [o.name for o in outputs] == ["jw01751-o010_t005_miri_ch1-short_psfcube.fits"]
    product = PsfCubeProduct.read(outputs[0])
    assert product.pixelscale_arcsec == pytest.approx(600.0 / 64 / 190.0)
    assert product.wavelengths[0] == pytest.approx(4.9) and product.wavelengths[-1] == pytest.approx(4.9 + 39 * 0.02)
    assert len(product.wavelengths) >= 5 and np.all(np.diff(product.wavelengths) > 0)
    assert product.meta["JWFPSFPA"] == pytest.approx(30.0)
    assert product.meta["JWFPSFEN"] == "gaussian"
    assert product.header["CHANNEL"] == "1"
    assert np.all(product.sums() > 0.9)
    # FWHM grows with wavelength and matches the empirical relation on this grid
    fwhm = product.fwhm_arcsec()
    assert fwhm[-1] > fwhm[0]
    assert fwhm[0] == pytest.approx(float(approx_fwhm_arcsec(MIRI_1A, 4.9)), rel=0.1)


def test_psf_cube_step_native_grid_and_qa(tmp_path: Path):
    src = synthetic_cube(tmp_path / "jw01751-o006_t005_nirspec_g235h-f170lp_s3d.fits", instrument="NIRSPEC",
                         nwave=25, wave_min=1.66, wave_step=0.05, PA_APER=110.0)
    outputs = run_step("jwstflow.contrib.psf:PsfCube", [src], tmp_path,
                       params={"engine": "gaussian", "grid": "native", "npix": 48, "n_wavelengths": 6})
    product = PsfCubeProduct.read(outputs[0])
    assert product.pixelscale_arcsec == pytest.approx(0.13, rel=1e-3)   # the cube's own spaxel scale
    assert len(product.wavelengths) == 6
    figures = run_step("jwstflow.contrib.psf:QaPsfCube", outputs, tmp_path)
    assert len(figures) == 1 and figures[0].suffix == ".png" and figures[0].stat().st_size > 0


def test_psf_from_stage(tmp_path: Path):
    src = synthetic_cube(tmp_path / "jw001_miri_ch1-short_s3d.fits", instrument="MIRI", PA_APER=0.0)
    ctx = make_context(tmp_path, stage="downstream")
    outputs = run_step("jwstflow.contrib.psf:PsfCube", [src], tmp_path,
                       params={"engine": "gaussian", "distance_pc": 100.0, "npix": 32})
    ctx.stage_dirs["psf_cube"] = outputs[0].parent
    product = psf_from_stage(src, ctx, "psf_cube")
    assert product is not None and product.data.shape[1:] == (32, 32)
    assert psf_from_stage(src, ctx, "") is None
    assert psf_from_stage(src, ctx, "not_in_workflow") is None
    other = synthetic_cube(tmp_path / "jw001_nirspec_g395h_s3d.fits", instrument="NIRSPEC")
    with pytest.raises(FileNotFoundError, match="depends_on"):
        psf_from_stage(other, ctx, "psf_cube")


# --------------------------------------------------------------------------- the stpsf wiring (stubbed)
def _psf_hdulist(fov_pixels: int, pixelscale: float, wavelength_m: float, marker: float):
    """A calc_psf-like HDUList: DET_DIST carries a centred Gaussian + a marker pixel."""
    from astropy.io import fits

    n = fov_pixels
    c = (n - 1) / 2
    yy, xx = np.indices((n, n))
    sigma_pix = max(1.025 * wavelength_m / 6.603464 * 206264.8 / 2.3548 / pixelscale, 0.8)
    plane = np.exp(-((yy - c) ** 2 + (xx - c) ** 2) / (2 * sigma_pix**2))
    plane /= plane.sum()
    hdus = [fits.PrimaryHDU()]
    for name, data in (("OVERSAMP", plane * 0), ("DET_SAMP", plane * 0), ("OVERDIST", plane * 0),
                       ("DET_DIST", plane)):
        hdu = fits.ImageHDU(np.array(data), name=name)
        hdu.header["PIXELSCL"] = pixelscale
        hdus.append(hdu)
    hdus[-1].data[0, 0] = marker
    return fits.HDUList(hdus)


class _StubMiri:
    """Mimics the stpsf 2.x MIRI IFU surface the engine relies on, including the
    pixel-scale reset on band selection."""

    def __init__(self):
        self.options: dict = {}
        self.pixelscale = 0.11
        self.pupilopd = "stock"
        self.name = "MIRI"
        self._mode = "imaging"
        self._band = None
        self._ifu_slice_width = None
        self.calls: list[dict] = []

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        assert value == "IFU"
        self._mode = value
        self.pixelscale = 0.13

    @property
    def band(self):
        return self._band

    @band.setter
    def band(self, value):
        assert self._mode == "IFU", "band is only settable in IFU mode"
        assert value in {f"{c}{s}" for c in "1234" for s in "ABC"}
        self._band = value
        self._ifu_slice_width = 0.177
        self.pixelscale = 0.13          # stpsf resets the scale on any IFU aperture change

    def calc_psf(self, monochromatic=None, fov_pixels=None, oversample=None, add_distortion=None):
        self.calls.append({"monochromatic": monochromatic, "fov_pixels": fov_pixels,
                           "oversample": oversample, "pixelscale": self.pixelscale})
        return _psf_hdulist(fov_pixels, self.pixelscale, monochromatic, marker=0.5)


class _StubNirspec(_StubMiri):
    def __init__(self):
        super().__init__()
        self.name = "NIRSpec"
        self.disperser = None
        self.filter = "F110W"

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        assert value == "IFU"
        self._mode = value
        self.pixelscale = 0.10435


@pytest.fixture()
def stub_stpsf(monkeypatch):
    module = types.SimpleNamespace(MIRI=_StubMiri, NIRSpec=_StubNirspec, __version__="2.x-stub")
    monkeypatch.setitem(sys.modules, "stpsf", module)
    return module


def test_stpsf_engine_selects_the_broadened_extension(stub_stpsf):
    stack = StpsfEngine().compute(MIRI_1A, np.array([5.0, 5.6]), PsfGrid(0.0123, 24))
    assert stack.shape == (2, 24, 24)
    assert stack[0, 0, 0] == 0.5                       # DET_DIST (the broadened plane) was selected


def test_stpsf_engine_call_arguments(stub_stpsf, monkeypatch):
    captured: list[_StubMiri] = []
    original = stub_stpsf.MIRI
    monkeypatch.setattr(stub_stpsf, "MIRI", lambda: captured.append(original()) or captured[-1])
    grid = PsfGrid(0.0123, 24)
    StpsfEngine().compute(MIRI_1A, np.array([5.0, 5.6]), grid)
    (inst,) = captured
    assert inst.band == "1A" and inst.mode == "IFU"
    assert inst.pixelscale == pytest.approx(0.0123)     # the reset-on-band-selection was overridden
    assert [c["monochromatic"] for c in inst.calls] == pytest.approx([5.0e-6, 5.6e-6])
    assert all(c["fov_pixels"] == 24 and c["oversample"] == 1 for c in inst.calls)
    assert all(c["pixelscale"] == pytest.approx(0.0123) for c in inst.calls)


def test_stpsf_engine_nirspec_configuration(stub_stpsf, monkeypatch):
    captured: list[_StubNirspec] = []
    original = stub_stpsf.NIRSpec
    monkeypatch.setattr(stub_stpsf, "NIRSpec", lambda: captured.append(original()) or captured[-1])
    StpsfEngine(broadening="none").compute(NRS_G235H, np.array([2.0]), PsfGrid(0.02, 16))
    (inst,) = captured
    assert inst.disperser == "G235H" and inst.filter == "F170LP"
    assert inst.options["ifualign_rotation"] is False   # array +y must stay the ideal +y axis
    assert inst.options["ifu_broadening"] is None
    assert inst.pixelscale == pytest.approx(0.02)


def test_stpsf_missing_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "stpsf", None)
    with pytest.raises(RuntimeError, match=r"jwstflow\[psf\]"):
        StpsfEngine().compute(MIRI_1A, np.array([5.0]), PsfGrid(0.05, 16))
