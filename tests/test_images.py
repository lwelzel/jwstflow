"""The images toolkit: filters, reading imaging products, cutouts with a WCS of their own."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jwstflow.images import all_bands, band_of, diffraction_fwhm_arcsec, read_image
from jwstflow.testing import synthetic_i2d


def write_i2d(path: Path) -> Path:
    """120 x 100 pixels of 0.031 arcsec, reference pixel (40, 70), three exposures."""
    return synthetic_i2d(path, shape=(120, 100), ra=65.0893, dec=28.2303, crpix=(40.0, 70.0))


def test_band_of_takes_the_filter_from_whichever_wheel_holds_it():
    assert band_of("NIRCAM", "F200W", "CLEAR").name == "F200W"
    assert band_of("NIRCAM", "F444W", "F405N").name == "F405N"       # the narrow band behind the wide filter
    assert band_of("NIRISS", "CLEAR", "F150W").name == "F150W"
    miri = band_of("MIRI", "F770W", None)
    assert miri.name == "F770W" and 7.5 < miri.pivot_um < 7.8 and miri.wave_min_um < miri.pivot_um < miri.wave_max_um
    assert band_of("MIRI", "P750L") is None and band_of("NIRSPEC", "F290LP", "G395H") is None
    pivots = [b.pivot_um for b in all_bands()]
    assert pivots == sorted(pivots) and len(pivots) > 50


def test_diffraction_fwhm_follows_lambda_over_d():
    assert diffraction_fwhm_arcsec(2.0) == pytest.approx(0.0654, rel=0.01)
    assert band_of("MIRI", "F2100W").fwhm_arcsec == pytest.approx(0.68, rel=0.03)


def test_read_image_and_a_cutout_keep_the_sky_where_it_is(tmp_path: Path):
    image = read_image(write_i2d(tmp_path / "x_i2d.fits"))
    assert image.band.name == "F200W" and image.unit == "MJy/sr"
    assert image.pixscale_arcsec == pytest.approx(0.031)
    assert image.fwhm_pix == pytest.approx(diffraction_fwhm_arcsec(image.band.pivot_um) / 0.031)
    assert image.n_exposures[0, 0] == 3 and image.n_exposures[0, -1] == 1
    ra, dec = image.world(50.0, 30.0)
    cut = image.cutout(ra, dec, 1.0)                     # 1" = 32 pixels across, centred on (50, 30)
    assert cut.shape == (35, 35) and cut.origin == (33, 13)
    # the same pixel of the sky in the parent and in the cutout
    for y, x in ((0, 0), (17, 17), (34, 5)):
        assert cut.world(y, x) == pytest.approx(image.world(y + 33, x + 13), abs=1e-10)
        assert cut.data[y, x] == image.data[y + 33, x + 13]
    from astropy.wcs import WCS

    header_wcs = WCS(cut.header).celestial                  # the header written with a cutout: same sky
    assert [float(v) for v in header_wcs.all_pix2world(5, 34, 0)] == pytest.approx(image.world(34 + 33, 5 + 13))
    assert cut.header["JWFCUTY0"] == 33 and cut.header["JWFCUTX0"] == 13
    edge = image.cutout(*image.world(2.0, 2.0), 1.0)       # clipped at the corner of the image
    assert edge.origin == (0, 0) and edge.shape == (20, 20)        # 2 + 17 + 1 pixels
    nested = cut.cutout(*cut.world(17.0, 17.0), 0.5)        # a cutout of a cutout knows the parent's origin
    assert nested.world(0, 0) == pytest.approx(image.world(nested.origin[0], nested.origin[1]), abs=1e-10)
