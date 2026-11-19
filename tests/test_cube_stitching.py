"""Joint cubes (jwstflow.cubes.stitching): the grid, the placement of cubes on it, the stitch, its figures.

The synthetic cubes are small IFU cube models with a Gaussian source at one sky
position, on grids that differ the way real band cubes differ: another tangent
point, another spaxel scale, a lattice shifted by a fraction of a spaxel. What
the tests pin is what a joint cube promises: values are copied when the
lattices agree, flux is conserved when they do not, the source stays where the
sky says it is, no wavelength is resampled, and the overlap ratio is the one
the 1-D stitcher would measure.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from jwstflow.cubes import cube_wavelengths
from jwstflow.cubes.stitching import (
    IDENTITY,
    JOINT_SUFFIX,
    LATTICE_TOLERANCE_PIX,
    NO_DATA_DQ,
    REGRIDDED,
    Placement,
    SpatialGrid,
    StitchCubes,
    data_bounds,
    image_shift,
    joint_grid,
    joint_grid_of,
    overlap_matrix,
    read_joint_records,
    read_segment,
    reference_index,
    region_flux,
    rim,
    shared_window,
    source_region,
)
from jwstflow.qa.steps import QaCubeStitch
from jwstflow.spectra.overlaps import (
    OverlapRatio,
    chain_scales,
    mask_windows,
    overlap_midpoints,
)
from jwstflow.testing import check_step, run_step

RA0, DEC0 = 150.0, -30.0
ARCSEC = 1.0 / 3600.0
SR_PER_ARCSEC2 = (math.pi / 180.0 / 3600.0) ** 2


def grid(scale: float = 0.1, size: tuple[int, int] = (21, 21), *, east: float = 0.0, north: float = 0.0,
         rotation: float = 0.0, crpix: tuple[float, float] | None = None) -> SpatialGrid:
    """A grid in the layout of jwst cubes: ``scale`` arcsec spaxels, the tangent point ``east`` /
    ``north`` arcsec from (RA0, DEC0) on the central pixel, rotated by ``rotation`` degrees."""
    ny, nx = size
    angle = math.radians(rotation)
    s = scale * ARCSEC
    cd = ((-s * math.cos(angle), s * math.sin(angle)), (s * math.sin(angle), s * math.cos(angle)))
    crval = (RA0 + east * ARCSEC / math.cos(math.radians(DEC0)), DEC0 + north * ARCSEC)
    return SpatialGrid(crval, crpix or ((nx + 1) / 2, (ny + 1) / 2), cd, (ny, nx))


def source_image(g: SpatialGrid, *, east: float = 0.0, north: float = 0.0, fwhm: float = 0.35,
                 flux_jy: float = 1e-3) -> np.ndarray:
    """Surface brightness [MJy/sr] of a Gaussian source of ``flux_jy`` at ``east``/``north`` arcsec
    from (RA0, DEC0), integrated over the spaxels of ``g`` (5 x 5 sub-samples per spaxel)."""
    ny, nx = g.shape
    sub = (np.arange(5) + 0.5) / 5 - 0.5
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    total = np.zeros((ny, nx))
    sigma = fwhm / 2.3548
    for dy in sub:
        for dx in sub:
            ra, dec = g.pixel_to_sky(xx + dx, yy + dy)
            x = (ra - RA0) * math.cos(math.radians(DEC0)) / ARCSEC - east
            y = (dec - DEC0) / ARCSEC - north
            total += np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    profile = total / 25.0 / (2 * math.pi * sigma**2)            # per arcsec^2, unit integral
    return profile * flux_jy / 1e6 / SR_PER_ARCSEC2              # Jy/arcsec^2 -> MJy/sr


def write_cube(path: Path, g: SpatialGrid, wave: np.ndarray, image: np.ndarray, *, band: tuple[str, str],
               instrument: str = "NIRSPEC", spectrum: np.ndarray | None = None, error: float = 0.01,
               margin: int = 0) -> Path:
    """An IFU cube model on grid ``g``: ``image`` [MJy/sr] times ``spectrum`` in every plane;
    ``margin`` spaxels at the rim carry no data (NaN, DQ 513), as in a real cube."""
    from stdatamodels.jwst import datamodels

    nw = len(wave)
    planes = (np.ones(nw) if spectrum is None else np.asarray(spectrum, dtype=float))[:, None, None]
    data = (image[None] * planes).astype("f4")
    dq = np.zeros(data.shape, "u4")
    if margin:
        rim = np.ones(image.shape, dtype=bool)
        rim[margin:-margin, margin:-margin] = False
        data[:, rim], dq[:, rim] = np.nan, NO_DATA_DQ
    cube = datamodels.IFUCubeModel(data=data, err=np.where(np.isfinite(data), error, np.nan).astype("f4"), dq=dq,
                                   weightmap=np.isfinite(data).astype("f4") * 4)
    cube.meta.instrument.name = instrument
    if instrument == "MIRI":
        cube.meta.instrument.channel, cube.meta.instrument.band, cube.meta.exposure.type = band[0], band[1], "MIR_MRS"
    else:
        cube.meta.instrument.grating, cube.meta.instrument.filter, cube.meta.exposure.type = band[0], band[1], "NRS_IFU"
    cube.meta.target.proposer_name, cube.meta.target.ra, cube.meta.target.dec = "TEST-TARGET", RA0, DEC0
    cube.meta.bunit_data = "MJy/sr"
    w = cube.meta.wcsinfo
    for key, value in g.header_cards().items():
        setattr(w, key.lower(), value)
    w.ctype3, w.cunit3, w.crval3, w.crpix3, w.cdelt3, w.wcsaxes = "WAVE", "um", float(wave[0]), 1.0, float(wave[1] - wave[0]), 3
    cube.meta.photometry.pixelarea_arcsecsq = g.pixel_area_arcsec2
    cube.meta.photometry.pixelarea_steradians = g.pixel_area_sr
    cube.save(str(path))
    return Path(path)


def flux_of(image: np.ndarray, g: SpatialGrid) -> float:
    """Flux [Jy] of a surface-brightness image [MJy/sr] on grid ``g``."""
    return float(np.nansum(image)) * g.pixel_area_sr * 1e6


# --------------------------------------------------------------------------- the grid
def test_grid_round_trips_through_a_cube_header(tmp_path: Path):
    g = grid(0.13, (23, 27), east=0.4, north=-0.2)
    path = write_cube(tmp_path / "jw001_miri_ch1-short_s3d.fits", g, np.linspace(5.0, 5.5, 6), source_image(g),
                      band=("1", "SHORT"), instrument="MIRI")
    back = SpatialGrid.from_fits(path)
    assert back.shape == g.shape and back.crpix == pytest.approx(g.crpix)
    assert back.crval == pytest.approx(g.crval, abs=1e-10)
    assert back.scale == pytest.approx(0.13, rel=1e-9) and back.rotation == pytest.approx(0.0, abs=1e-9)
    assert back.lattice_offset(g) == (0, 0)


def test_joint_grid_takes_the_finest_scale_over_the_union_of_the_fields():
    fine = grid(0.13, (21, 21))                                  # 2.73" field
    coarse = grid(0.35, (15, 15), east=0.3, north=-0.2)          # 5.25" field, another tangent point
    joint = joint_grid([fine, coarse])
    assert joint.scale == pytest.approx(0.13)
    assert joint.crval == fine.crval                             # the lattice of the finest grid, extended
    assert joint.lattice_offset(fine) is not None
    ny, nx = joint.shape
    assert nx % 2 == 1 and ny % 2 == 1 and joint.crpix == ((nx + 1) / 2, (ny + 1) / 2)
    # every corner of the coarse field lies inside (to the lattice tolerance), and one spaxel less on
    # either side would not hold it
    cx, cy = joint.pixels_of(coarse, [-0.5, 14.5, -0.5, 14.5], [-0.5, -0.5, 14.5, 14.5])
    tol = LATTICE_TOLERANCE_PIX
    assert cx.min() >= -0.5 - tol and cx.max() <= nx - 0.5 + tol and cy.min() >= -0.5 - tol and cy.max() <= ny - 0.5 + tol
    assert max(abs(cx.min() - (nx - 1) / 2), abs(cx.max() - (nx - 1) / 2)) > (nx - 3) / 2 + 0.5 + tol
    # the order decides among equally fine grids; the reference is the first of them
    other = grid(0.13, (21, 21), east=0.05)
    assert joint_grid([other, fine, coarse]).crval == other.crval


def test_joint_grid_of_matching_grids_is_that_grid():
    g = grid(0.1, (31, 35))
    assert joint_grid([g, g, g]) == g
    # data footprints trim it: cubes whose rims are empty need less room
    trimmed = joint_grid([g, g], [(3, 31, 2, 28), (4, 30, 3, 27)])
    assert trimmed.shape == (2 * 13 + 1, 2 * 14 + 1) and trimmed.lattice_offset(g) == (-3, -2)


def test_joint_grid_keeps_the_lattice_of_a_reference_laid_out_otherwise():
    """An even grid has its tangent point between pixels: its lattice is kept all the same."""
    fine = grid(0.1, (20, 24))                                   # CRPIX (12.5, 10.5): no pixel on the tangent point
    coarse = grid(0.3, (11, 11), east=0.2)
    joint = joint_grid([coarse, fine])
    assert reference_index([coarse, fine]) == 1 and joint.scale == pytest.approx(0.1)
    x0, y0 = joint.lattice_offset(fine)                          # the finest grid is a window of the joint grid
    assert joint.crpix == (fine.crpix[0] + x0, fine.crpix[1] + y0)
    cx, cy = joint.pixels_of(coarse, [-0.5, 10.5], [-0.5, 10.5])
    assert cx.min() >= -0.5 - 1e-3 and cx.max() <= joint.shape[1] - 0.5 + 1e-3
    assert cy.min() >= -0.5 - 1e-3 and cy.max() <= joint.shape[0] - 0.5 + 1e-3
    assert Placement.plan(fine, joint).mode == IDENTITY
    with pytest.raises(ValueError, match="odd number"):
        joint.cube_build_parameters()                            # ... but cube_build could not make it


def test_grid_as_cube_build_parameters():
    g = joint_grid([grid(0.13, (21, 21)), grid(0.2, (25, 25), east=0.2)])
    pars = g.cube_build_parameters()
    assert pars == {"scalexy": pytest.approx(0.13), "ra_center": g.crval[0], "dec_center": g.crval[1],
                    "nspax_x": g.shape[1], "nspax_y": g.shape[0]}
    # cube_build lays a cube out from exactly these numbers (ifu_cube.set_geometry): tangent point on
    # the central spaxel of an odd grid -- so a cube built with them has this grid
    built = grid(pars["scalexy"], (pars["nspax_y"], pars["nspax_x"]))
    assert g.lattice_offset(built) == (0, 0) and built.shape == g.shape
    assert joint_grid([grid(0.1, rotation=20.0)]).cube_build_parameters()["cube_pa"] == pytest.approx(20.0)
    with pytest.raises(ValueError, match="odd number"):
        grid(0.1, (20, 20)).cube_build_parameters()


def test_lattices_agree_only_to_a_thousandth_of_a_pixel():
    g = grid(0.1, (21, 21))
    assert g.lattice_offset(grid(0.1, (11, 11), crpix=(3.0, 2.0))) == (8, 9)      # same lattice, another window
    assert g.lattice_offset(grid(0.1, (21, 21), east=0.0075)) is None            # 0.075 spaxel: not the same
    assert g.lattice_offset(grid(0.1, (21, 21), east=0.00005)) is not None       # 0.0005 spaxel: the same
    assert g.lattice_offset(grid(0.2, (21, 21))) is None
    assert g.lattice_offset(grid(0.1, (21, 21), rotation=0.5)) is None
    assert g.phase_of(grid(0.1, (21, 21), east=0.03)) == pytest.approx((-0.3, 0.0), abs=1e-6)   # east is -x


# --------------------------------------------------------------------------- overlaps of pixels
def test_overlap_matrix_of_a_shifted_lattice_is_the_product_of_the_shifts():
    g = grid(0.1, (9, 9))
    shifted = grid(0.1, (9, 9), east=-0.025, north=0.05)         # +0.25 pixel in x, +0.5 in y on the target
    matrix = overlap_matrix(shifted, g).toarray().reshape(9, 9, 9, 9)     # [ty, tx, sy, sx]
    assert matrix[4, 4, 4, 4] == pytest.approx(0.75 * 0.5)
    assert matrix[4, 5, 4, 4] == pytest.approx(0.25 * 0.5)
    assert matrix[5, 4, 4, 4] == pytest.approx(0.75 * 0.5)
    assert matrix[5, 5, 4, 4] == pytest.approx(0.25 * 0.5)
    assert matrix[:, :, 4, 4].sum() == pytest.approx(1.0)        # an interior source pixel is handed over whole
    assert matrix[4, 4].sum() == pytest.approx(1.0)              # an interior target pixel is covered whole


def test_overlap_matrix_conserves_area_for_any_scale_and_rotation():
    target = grid(0.1, (61, 61))
    for source in (grid(0.35, (9, 9), east=0.13, north=-0.07), grid(0.2, (11, 11), rotation=33.0, east=0.05),
                   grid(0.1, (15, 15), rotation=-12.0, north=0.031)):
        matrix = overlap_matrix(source, target)
        columns = np.asarray(matrix.sum(axis=0)).ravel()         # the area of every source pixel, in target pixels
        assert columns == pytest.approx((source.scale / target.scale) ** 2, rel=1e-6)
        assert np.asarray(matrix.sum(axis=1)).max() <= 1.0 + 1e-9


def test_overlap_matrix_of_one_lattice_is_the_identity():
    g = grid(0.13, (7, 9))
    assert np.allclose(overlap_matrix(g, g).toarray(), np.eye(63))


# --------------------------------------------------------------------------- cubes on a grid
def test_a_cube_on_the_joint_lattice_is_copied_value_for_value():
    target = grid(0.1, (25, 25))
    source = grid(0.1, (11, 13), crpix=(5.0, 3.0))               # the same lattice: a window of the target
    rng = np.random.default_rng(1)
    data = rng.normal(5.0, 1.0, (4, 11, 13)).astype("f4")
    data[1, 3, 4] = np.nan
    err, dq = np.full(data.shape, 0.2, "f4"), np.zeros(data.shape, "u4")
    dq[2, 5, 6], dq[3, 0, 0] = 1, 4                              # DO_NOT_USE: no measurement; another flag: kept
    placement = Placement.plan(source, target)
    assert placement.mode == IDENTITY and placement.correlation == 1.0
    out = placement.cube(data, err, dq, np.ones(data.shape, "f4"))
    x0, y0 = placement.offset
    window = out["data"][:, y0:y0 + 11, x0:x0 + 13]
    expected = data.copy()
    expected[2, 5, 6] = np.nan
    assert np.array_equal(window, expected, equal_nan=True)      # bit for bit
    assert np.isnan(out["data"]).sum() == out["data"].size - np.isfinite(expected).sum()
    assert out["dq"][0, 0, 0] == NO_DATA_DQ and out["dq"][1, y0 + 3, x0 + 4] == NO_DATA_DQ
    assert out["dq"][3, y0, x0] == 4 and out["dq"][0, y0 + 2, x0 + 2] == 0
    assert out["err"][0, y0 + 2, x0 + 2] == np.float32(0.2) and out["wmap"][0, 0, 0] == 0


@pytest.mark.parametrize("source", [grid(0.35, (15, 15), east=0.11, north=-0.23),      # coarser, off-lattice
                                    grid(0.1, (41, 41), east=0.043, north=0.031),       # same scale, shifted
                                    grid(0.2, (25, 25), rotation=27.0, east=-0.07)],    # rotated
                         ids=["coarser", "shifted", "rotated"])
def test_resampling_conserves_flux_and_keeps_the_source_in_place(source: SpatialGrid):
    target = joint_grid([grid(0.1, (15, 15)), source])
    image = source_image(source, east=0.25, north=-0.15, fwhm=0.6)
    placement = Placement.plan(source, target)
    assert placement.mode == REGRIDDED
    out = placement.cube(image[None].astype("f4"), np.full((1, *image.shape), 0.01, "f4"))
    assert flux_of(out["data"][0], target) == pytest.approx(flux_of(image, source), rel=2e-5)
    # the centroid of everything, in sky offsets: the same before and after
    def centroid(img: np.ndarray, g: SpatialGrid) -> tuple[float, float]:
        yy, xx = np.indices(img.shape)
        ok = np.isfinite(img)
        ra, dec = g.pixel_to_sky(xx[ok], yy[ok])
        weight = img[ok]
        return (float(np.sum(weight * (ra - RA0)) / weight.sum() * math.cos(math.radians(DEC0)) / ARCSEC),
                float(np.sum(weight * (dec - DEC0)) / weight.sum() / ARCSEC))

    before, after = centroid(image, source), centroid(out["data"][0], target)
    assert after == pytest.approx(before, abs=1e-3)              # arcsec: a hundredth of the target spaxel
    assert before == pytest.approx((0.25, -0.15), abs=5e-3)


def test_resampling_spreads_nothing_beyond_the_pixels_a_source_pixel_touches():
    """A finer target shows the coarse spaxel it lies in -- no interpolation between spaxels."""
    source, target = grid(0.3, (5, 5)), grid(0.1, (15, 15))     # three target spaxels per source spaxel, aligned
    image = np.zeros((5, 5))
    image[2, 2] = 9.0
    out = Placement.plan(source, target).image(image)
    assert np.all(out[6:9, 6:9] == pytest.approx(9.0))
    out[6:9, 6:9] = 0.0
    assert np.nanmax(np.abs(out)) < 1e-6                         # nothing beyond rounding at the shared edges


def test_resampling_propagates_errors_weights_and_missing_data():
    source, target = grid(0.2, (11, 11), east=0.05), grid(0.1, (25, 25))
    data = np.full((2, 11, 11), 3.0, "f4")
    data[0, 5, 5] = np.nan                                       # a hole
    data[:, :2] = np.nan
    placement = Placement.plan(source, target, pixels=np.isfinite(data).any(axis=0))
    out = placement.cube(data, np.full(data.shape, 0.5, "f4"), None, np.full(data.shape, 8.0, "f4"), min_coverage=0.5)
    inside = np.isfinite(out["data"][1])
    assert np.all(out["data"][1][inside] == pytest.approx(3.0))  # a constant stays the constant
    # a target spaxel inside one source spaxel keeps its error; one shared between two averages them
    assert np.nanmax(out["err"][1]) == pytest.approx(0.5) and np.nanmin(out["err"][1]) == pytest.approx(0.5 / math.sqrt(2), rel=1e-5)
    assert np.all(out["wmap"][1][inside] == pytest.approx(8.0)) and np.all(out["wmap"][1][~inside] == 0.0)
    assert np.all(out["dq"][1][inside] == 0) and np.all(out["dq"][1][~inside] == NO_DATA_DQ)
    # the hole: target spaxels wholly inside it are empty, those half inside it are kept
    hole = np.isfinite(out["data"][1]) & ~np.isfinite(out["data"][0])
    assert 0 < hole.sum() <= 4
    # each source spaxel covers 2 x 2 target spaxels, half a spaxel off in x: overlaps (0.5, 1, 0.5) x (1, 1),
    # so the variance of a sum is 4^2 / (1.5 * 2) = 16/3 times the variances added up
    assert placement.correlation == pytest.approx(16 / 3, rel=0.02)
    stricter = placement.cube(data, min_coverage=1.0)["data"][0]
    assert np.isfinite(stricter).sum() < np.isfinite(out["data"][0]).sum()


def test_a_value_without_an_error_does_not_count_as_exact():
    source, target = grid(0.1, (9, 9), east=0.03), grid(0.1, (9, 9))
    data, err = np.ones((1, 9, 9), "f4"), np.full((1, 9, 9), 0.2, "f4")
    err[0, 4, 4] = np.nan
    out = Placement.plan(source, target).cube(data, err)
    unknown = np.isfinite(out["data"][0]) & ~np.isfinite(out["err"][0])
    assert {tuple(i) for i in np.argwhere(unknown)} == {(4, 4), (4, 3)}      # the spaxels it went into; east is -x
    assert np.isfinite(out["err"][0, 2, 2])


def test_dq_bits_of_contributing_spaxels_are_passed_on():
    source, target = grid(0.1, (9, 9), east=0.03), grid(0.1, (9, 9))
    data, dq = np.ones((1, 9, 9), "f4"), np.zeros((1, 9, 9), "u4")
    dq[0, 4, 4] = 4
    out = Placement.plan(source, target).cube(data, dq=dq)
    flagged = np.argwhere(out["dq"][0] == 4)
    assert len(flagged) == 2 and {tuple(f) for f in flagged} == {(4, 4), (4, 3)}     # east is -x


# --------------------------------------------------------------------------- the stitch
WAVE_A, WAVE_B = np.arange(100) * 0.01 + 2.0, np.arange(80) * 0.02 + 2.7     # overlap 2.70-2.99 um


def two_cubes(tmp_path: Path, *, grid_b: SpatialGrid | None = None, ratio: float = 1.25, source_b: tuple[float, float] = (0.0, 0.0),
              background: float = 0.0) -> tuple[Path, Path, SpatialGrid, SpatialGrid]:
    """Two cubes of one source: ``b`` ``ratio`` times brighter, on ``grid_b`` (default: the grid of ``a``)."""
    grid_a = grid(0.1, (25, 25))
    grid_b = grid_b or grid_a
    a = write_cube(tmp_path / "jw001-o001_t001_nirspec_g235h-f170lp_s3d.fits", grid_a, WAVE_A,
                   source_image(grid_a) + background, band=("G235H", "F170LP"), margin=2)
    b = write_cube(tmp_path / "jw001-o001_t001_nirspec_g395h-f290lp_s3d.fits", grid_b, WAVE_B,
                   ratio * (source_image(grid_b, east=source_b[0], north=source_b[1]) + background),
                   band=("G395H", "F290LP"), margin=2)
    return a, b, grid_a, grid_b


def test_declaration_is_valid():
    for step in (StitchCubes, QaCubeStitch):
        _, problems = check_step(step)
        assert problems == []


def test_matching_cubes_are_joined_without_touching_a_value(tmp_path: Path):
    a, b, _, _ = two_cubes(tmp_path)
    (out,) = run_step(StitchCubes, [b, a], tmp_path)
    assert out.name == f"jw001-o001_t001_nirspec_{JOINT_SUFFIX}.fits"
    from stdatamodels.jwst import datamodels

    with datamodels.open(str(out)) as cube:                      # an IFU cube like its inputs
        assert type(cube).__name__ == "IFUCubeModel"
        wave = cube_wavelengths(cube)
        data, err, dq = np.array(cube.data), np.array(cube.err), np.array(cube.dq)
        assert cube.meta.instrument.grating == "MULTIPLE" and cube.meta.instrument.filter == "MULTIPLE"
        assert cube.meta.wcsinfo.ctype3 == "WAVE-TAB"
        sky = cube.meta.wcs(10.0, 10.0, [0, 50, 100])            # the gwcs knows the table
        assert np.asarray(sky[2]) == pytest.approx(wave[[0, 50, 100]], rel=1e-6)
    crossover = 0.5 * (2.70 + 2.99)                              # the midpoint of the overlap
    used_a, used_b = WAVE_A[WAVE_A <= crossover], WAVE_B[WAVE_B > crossover]
    assert wave == pytest.approx(np.concatenate([used_a, used_b]), rel=1e-7)     # no wavelength was resampled
    rec = read_joint_records(out)
    segments = rec["segments"]
    assert [s.strip() for s in segments["label"]] == ["g235h-f170lp", "g395h-f290lp"]
    assert list(segments["placement"]) == [IDENTITY, IDENTITY] and list(segments["scale"]) == [1.0, 1.0]
    assert list(segments["n_planes"]) == [len(used_a), len(used_b)]
    assert list(segments["plane_first"]) == [0, len(used_a)]
    # the joint grid is the data footprint of the inputs (25 - 2 * 2 spaxels), their values in place
    assert data.shape == (len(wave), 21, 21)
    with fits.open(a) as hdul:
        assert np.array_equal(data[:len(used_a)], hdul["SCI"].data[:len(used_a), 2:-2, 2:-2])
        assert np.array_equal(err[:len(used_a)], hdul["ERR"].data[:len(used_a), 2:-2, 2:-2])
    with fits.open(b) as hdul:
        first = int(np.argmax(WAVE_B > crossover))
        assert np.array_equal(data[len(used_a):], hdul["SCI"].data[first:, 2:-2, 2:-2])
    assert not dq.any()
    # measured, not applied
    overlap = rec["overlaps"][0]
    assert overlap["crossover"] == pytest.approx(crossover) and overlap["ratio"] == pytest.approx(1.25, rel=1e-5)
    assert overlap["window_min"] == pytest.approx(crossover * 0.975) and overlap["window_max"] == pytest.approx(crossover * 1.025)
    assert abs(overlap["dra"]) < 1e-4 and abs(overlap["ddec"]) < 1e-4            # arcsec: the same place
    assert rec["header"]["JWFJCNSG"] == 2 and rec["header"]["JWFJCNRS"] == 0 and rec["header"]["JWFJCSCL"] == "none"


def test_rescale_chains_onto_the_reference_and_carries_the_uncertainty(tmp_path: Path):
    a, b, _, _ = two_cubes(tmp_path)
    (out,) = run_step(StitchCubes, [a, b], tmp_path, params={"rescale": True, "crossovers": [2.9]})
    rec = read_joint_records(out)
    scale, scale_error = float(rec["segments"]["scale"][0]), float(rec["segments"]["scale_error"][0])
    assert scale == pytest.approx(1.25, rel=1e-5) and float(rec["segments"]["scale"][1]) == 1.0   # onto the reddest
    assert scale_error > 0 and float(rec["segments"]["scale_error"][1]) == 0.0
    assert rec["header"]["JWFJCREF"] == "g395h-f290lp" and rec["header"]["JWFJCSCL"] == "measured"
    n_a = int(rec["segments"]["n_planes"][0])
    assert n_a == int((WAVE_A <= 2.9 + 1e-9).sum())
    with fits.open(out) as hdul, fits.open(a) as first:
        sci, err = hdul["SCI"].data, hdul["ERR"].data
        raw, raw_err = first["SCI"].data[:n_a, 2:-2, 2:-2], first["ERR"].data[:n_a, 2:-2, 2:-2]
        assert sci[:n_a] == pytest.approx(raw * scale, rel=1e-6)
        # the scale uncertainty is part of the error of every rescaled spaxel
        assert err[:n_a] == pytest.approx(np.hypot(raw_err * scale, raw * scale_error), rel=1e-5)
    # the cube is continuous across the crossover now: the plain sum of the field, and the source's excess
    _, flux = region_flux(out, rec["common"])
    assert flux[n_a - 1] == pytest.approx(flux[n_a], rel=1e-4)
    _, excess = region_flux(out, rec["source"], rec["common"])
    assert excess[n_a - 1] == pytest.approx(excess[n_a], rel=1e-4) and 0 < rec["source"].sum() < rec["common"].sum()
    assert excess[0] < flux[0]                                               # the core, not the whole field
    (shortest,) = run_step(StitchCubes, [a, b], tmp_path / "s", params={"rescale": True, "reference": "shortest"})
    assert list(read_joint_records(shortest)["segments"]["scale"]) == pytest.approx([1.0, 1 / 1.25], rel=1e-5)


def test_cubes_on_other_grids_are_resampled_once_flux_conserved_source_in_place(tmp_path: Path, caplog):
    coarse = grid(0.25, (19, 19), east=0.07, north=-0.11)       # another scale, another tangent point, a wider field
    a, b, grid_a, grid_b = two_cubes(tmp_path, grid_b=coarse, ratio=1.0)
    with caplog.at_level("WARNING"):
        (out,) = run_step(StitchCubes, [a, b], tmp_path, params={"ratio_region": "footprint"})
    assert "cube_build_matched" in caplog.text                   # the step says what it had to do, and the remedy
    rec = read_joint_records(out)
    assert list(rec["segments"]["placement"]) == [IDENTITY, REGRIDDED]
    assert float(rec["segments"]["spaxel_arcsec"][1]) == pytest.approx(0.25)
    assert 6.25 < float(rec["segments"]["correlation"][1]) < 9.5     # at least its (0.25 / 0.1)^2 joint spaxels
    assert rec["header"]["JWFJCNRS"] == 1 and rec["header"]["JWFJCGRD"].strip() == "g235h-f170lp"
    with fits.open(out) as hdul:
        sci, header = hdul["SCI"].data, hdul["SCI"].header
        joint = SpatialGrid.from_header(header)
    assert joint.scale == pytest.approx(0.1) and joint.shape[0] > 25 and joint.lattice_offset(grid_a) is not None
    n_a = int(rec["segments"]["n_planes"][0])
    # flux: what the coarse cube held over its field is in the joint plane
    with fits.open(b) as hdul:
        coarse_flux = flux_of(hdul["SCI"].data[-1], grid_b)
    assert flux_of(sci[-1], joint) == pytest.approx(coarse_flux, rel=1e-4)
    assert flux_of(sci[0], joint) == pytest.approx(1e-3, rel=0.01)           # the fine cube, as it was
    # position: the source of the coarse cube sits where the fine cube has it
    def first_moment(image: np.ndarray) -> tuple[float, float]:
        weight = np.nan_to_num(image)
        yy, xx = np.indices(image.shape)
        return float((weight * xx).sum() / weight.sum()), float((weight * yy).sum() / weight.sum())

    assert first_moment(sci[-1]) == pytest.approx(first_moment(sci[0]), abs=0.05)   # joint spaxels of 0.1 arcsec
    # ... and the registration of the two finds them together, to what blocks of 0.25 arcsec allow
    assert abs(rec["overlaps"][0]["dra"]) < 0.04 and abs(rec["overlaps"][0]["ddec"]) < 0.04
    assert rec["overlaps"][0]["ratio"] == pytest.approx(1.0, rel=0.01)       # same sky, same flux: no factor
    assert sci.shape[0] == n_a + int(rec["segments"]["n_planes"][1])


def test_a_position_difference_between_segments_is_measured_not_hidden(tmp_path: Path):
    a, b, _, _ = two_cubes(tmp_path, source_b=(0.05, -0.03))    # the redder cube puts the source elsewhere
    (out,) = run_step(StitchCubes, [a, b], tmp_path)
    overlap = read_joint_records(out)["overlaps"][0]
    assert overlap["dra"] == pytest.approx(0.05, abs=0.01) and overlap["ddec"] == pytest.approx(-0.03, abs=0.01)


def test_two_images_are_registered_whatever_their_scale_level_and_hot_spaxels():
    g = grid(0.1, (31, 31))
    a = source_image(g, fwhm=0.3)
    b = 1.3 * source_image(g, east=-0.03, north=0.045, fwhm=0.3) + 7.0      # shifted (+0.3, +0.45) px, scaled, on a pedestal
    field = np.ones(g.shape, dtype=bool)
    window = shared_window(a, b, field)
    assert 20 < window.sum() < 200 and window[15, 15]
    dx, dy = image_shift(a, b, window)
    assert (dx, dy) == pytest.approx((0.3, 0.45), abs=0.02)
    sharp, moved = source_image(g, fwhm=0.2), 0.8 * source_image(g, east=0.02, north=-0.01, fwhm=0.2)   # two-spaxel FWHM
    assert image_shift(sharp, moved, shared_window(sharp, moved, field)) == pytest.approx((-0.2, -0.1), abs=0.02)
    assert image_shift(a, 2.0 * a - 3.0, window) == pytest.approx((0.0, 0.0), abs=1e-6)     # no shift: none measured
    hot = a.copy()
    hot[16, 14] *= 30.0                                          # a deviant spaxel in one image, inside the window
    assert image_shift(hot, b, shared_window(hot, b, field)) == pytest.approx((0.3, 0.45), abs=0.05)
    assert np.isnan(image_shift(a, b, np.zeros(g.shape, dtype=bool))[0])


def test_the_pair_is_compared_on_the_core_both_segments_share(tmp_path: Path):
    """Another source at the edge of the field, brighter in one segment, must not be taken for the target."""
    grid_a = grid(0.1, (41, 41))
    target = source_image(grid_a, fwhm=0.35)
    companion = source_image(grid_a, east=-1.9, north=-1.7, fwhm=0.9)        # its wing reaches into the corner of the field
    a = write_cube(tmp_path / "jw001_nirspec_g235h-f170lp_s3d.fits", grid_a, WAVE_A, target + 2.0 * companion,
                   band=("G235H", "F170LP"))
    b = write_cube(tmp_path / "jw001_nirspec_g395h-f290lp_s3d.fits", grid_a, WAVE_B,
                   source_image(grid_a, east=0.02, north=0.0, fwhm=0.35) + 60.0 * companion, band=("G395H", "F290LP"))
    with fits.open(a) as ha, fits.open(b) as hb:                 # in b the companion is the brightest thing in the field
        peak_a = np.unravel_index(int(np.nanargmax(ha["SCI"].data[0])), (41, 41))
        peak_b = np.unravel_index(int(np.nanargmax(hb["SCI"].data[0])), (41, 41))
    assert peak_a == (20, 20) and peak_b != peak_a
    (out,) = run_step(StitchCubes, [a, b], tmp_path, params={"ratio_region": "footprint"})
    overlap = read_joint_records(out)["overlaps"][0]
    assert abs(overlap["dra"]) < 0.06 and abs(overlap["ddec"]) < 0.06        # one shared window: no arcsecond "shift"


def test_only_whole_sources_enter_the_measurement_region():
    """What the edge of the field cuts is held differently by every band: it is no place to compare them."""
    g = grid(0.1, (41, 41))
    field = np.ones(g.shape, dtype=bool)
    assert rim(field).sum() == 41**2 - 37**2                      # two rings, the edge of the array being an edge
    target = source_image(g, fwhm=0.35)
    alone = source_region(target, target, field, 0.1)
    assert alone[20, 20] and 20 < alone.sum() < 120 and not (alone & rim(field)).any()
    # a brighter object centred one spaxel inside the edge: left out, and the threshold stays the target's
    cut = target + 5.0 * source_image(g, east=-1.9, north=-1.2, fwhm=0.4)
    assert np.nanargmax(cut) != np.nanargmax(target)
    assert (source_region(cut, cut, field, 0.1) == alone).all()
    # a second object deep inside the field is a source like the target
    second = target + 0.5 * source_image(g, east=0.9, north=-0.8)
    two = source_region(second, second, field, 0.1)
    assert two[20, 20] and two[12, 11] and not two[16, 15]
    # a hole inside the field is no edge: the target around a missing spaxel stays the region
    holed = field.copy()
    holed[21, 22] = False
    around = source_region(target, target, holed, 0.1)
    assert around[20, 20] and not around[21, 22] and abs(int(around.sum()) - int(alone.sum())) <= 3
    # nothing but a source the field cuts: what stands out of it clear of the rim
    wing = source_image(g, east=-2.0, fwhm=1.0)
    left = source_region(wing, wing, field, 0.1)
    assert left.sum() >= 9 and not (left & rim(field)).any() and not left[:, :28].any()


def test_a_source_the_field_cuts_does_not_enter_the_ratio(tmp_path: Path):
    grid_a = grid(0.1, (41, 41))
    target = source_image(grid_a, fwhm=0.35)
    companion = source_image(grid_a, east=-1.9, north=-1.2, fwhm=0.4)        # centred one spaxel inside the edge
    a = write_cube(tmp_path / "jw001_nirspec_g235h-f170lp_s3d.fits", grid_a, WAVE_A, target + 3.0 * companion,
                   band=("G235H", "F170LP"))
    b = write_cube(tmp_path / "jw001_nirspec_g395h-f290lp_s3d.fits", grid_a, WAVE_B, 1.1 * target + 6.0 * companion,
                   band=("G395H", "F290LP"))                     # the bands disagree on what the edge cuts
    (out,) = run_step(StitchCubes, [a, b], tmp_path)
    rec = read_joint_records(out)
    region, overlap = rec["overlap_regions"][0], rec["overlaps"][0]
    assert region[20, 20] and not region[:, 30:].any()
    assert overlap["ratio"] == pytest.approx(1.1, rel=1e-3)                   # the target's
    assert abs(overlap["dra"]) < 0.005 and abs(overlap["ddec"]) < 0.005      # registered on the target


def test_the_ratio_is_that_of_the_source_whatever_the_field_levels(tmp_path: Path):
    """A level that differs between the segments -- sky, a subtraction residual -- is no calibration factor."""
    grid_a = grid(0.1, (25, 25))
    a = write_cube(tmp_path / "jw001_nirspec_g235h-f170lp_s3d.fits", grid_a, WAVE_A, source_image(grid_a) - 1.5,
                   band=("G235H", "F170LP"))
    b = write_cube(tmp_path / "jw001_nirspec_g395h-f290lp_s3d.fits", grid_a, WAVE_B,
                   1.1 * source_image(grid_a) + 4.0, band=("G395H", "F290LP"))                  # another pedestal
    (source,) = run_step(StitchCubes, [a, b], tmp_path / "source", params={"rescale": True})
    (field,) = run_step(StitchCubes, [a, b], tmp_path / "field", params={"ratio_region": "footprint"})
    in_source, in_field = read_joint_records(source)["overlaps"][0], read_joint_records(field)["overlaps"][0]
    assert in_source["n_spaxels"] < in_field["n_spaxels"]
    assert in_source["ratio"] == pytest.approx(1.1, rel=1e-4)                # the excess over the level: the factor
    assert abs(in_field["ratio"] - 1.1) > 0.05                               # plain sums: the pedestals leak in
    assert (in_source["level_a"], in_source["level_b"]) == pytest.approx((-1.5, 4.0), abs=0.05)    # recorded
    # the factor is applied to the cube; the levels stay what they are -- a step no factor removes
    with fits.open(source) as hdul:
        sci = hdul["SCI"].data
        n_a = int(read_joint_records(source)["segments"]["n_planes"][0])
        assert float(sci[n_a - 1, 0, 0]) == pytest.approx(-1.5 * 1.1, rel=1e-3)
        assert float(sci[n_a, 0, 0]) == pytest.approx(4.0, rel=1e-3)
        peak = np.unravel_index(int(np.nanargmax(sci[0])), sci[0].shape)
        assert float(sci[n_a - 1][peak] + 1.5 * 1.1) == pytest.approx(float(sci[n_a][peak] - 4.0), rel=1e-3)


def test_scales_of_a_stitched_spectrum_can_be_adopted(tmp_path: Path):
    """Cube and spectrum of the same data can carry the same factors (``scales_from``)."""
    from astropy.table import Table

    from jwstflow.testing import make_context

    a, b, _, _ = two_cubes(tmp_path)
    ctx = make_context(tmp_path, stage="stitch_cubes")
    spectra = ctx.root / "stage4" / "stitch_segments"
    spectra.mkdir(parents=True)
    ctx.stage_dirs["stitch_segments"] = spectra
    stitched = Table({"WAVELENGTH": [2.0, 3.0], "FLUX": [1.0, 1.0]})
    stitched.meta.update({"segments": ["g235h-f170lp", "g395h-f290lp"], "scales_applied": [1.3, 1.0], "scale_errors": [0.02, 0.0]})
    stitched.write(spectra / "jw001_nirspec_s1dcomb.ecsv", format="ascii.ecsv")
    (out,) = run_step(StitchCubes, [a, b], tmp_path, ctx=ctx, params={"scales_from": "stitch_segments"})
    rec = read_joint_records(out)
    assert list(rec["segments"]["scale"]) == [1.3, 1.0] and list(rec["segments"]["scale_error"]) == [0.02, 0.0]
    assert rec["header"]["JWFJCSCL"] == "adopted" and rec["header"]["JWFJCSFM"] == "stitch_segments"
    assert rec["overlaps"]["ratio"][0] == pytest.approx(1.25, rel=1e-5)      # the cube's own ratio, still recorded
    with fits.open(out) as hdul, fits.open(a) as first:
        assert hdul["SCI"].data[0] == pytest.approx(first["SCI"].data[0, 2:-2, 2:-2] * 1.3, rel=1e-6)
    with pytest.raises(ValueError, match="give one"):
        run_step(StitchCubes, [a, b], tmp_path, ctx=ctx, params={"scales_from": "stitch_segments", "rescale": True})
    with pytest.raises(KeyError, match="not a stage"):
        run_step(StitchCubes, [a, b], tmp_path, ctx=ctx, params={"scales_from": "nowhere"})
    with pytest.raises(ValueError, match="matches 0 file"):
        run_step(StitchCubes, [a, b], tmp_path, ctx=ctx, params={"scales_from": "stitch_segments", "scales_pattern": "x*.ecsv"})
    stitched.meta["segments"] = ["g140h-f100lp", "g395h-f290lp"]
    stitched.write(spectra / "jw001_nirspec_s1dcomb.ecsv", format="ascii.ecsv", overwrite=True)
    with pytest.raises(ValueError, match="no scale for"):
        run_step(StitchCubes, [a, b], tmp_path, ctx=ctx, params={"scales_from": "stitch_segments"})


def test_three_segments_one_reference_and_feature_masks(tmp_path: Path):
    g = grid(0.13, (21, 21))
    image = source_image(g, fwhm=0.5)
    waves = [np.arange(60) * 0.01 + 5.0, np.arange(60) * 0.01 + 5.5, np.arange(60) * 0.01 + 6.0]
    paths = []
    for (channel, band), wave, factor in zip((("1", "SHORT"), ("1", "MEDIUM"), ("1", "LONG")), waves, (1.0, 1.1, 1.21)):
        spectrum = np.full(len(wave), factor)
        if band == "SHORT":
            spectrum = spectrum + 5.0 * np.exp(-0.5 * ((wave - 5.511) / 0.006) ** 2)    # H2 0-0 S(7), in the first overlap
        paths.append(write_cube(tmp_path / f"jw001-o001_t001_miri_ch{channel}-{band.lower()}_s3d.fits", g, wave, image,
                                band=(channel, band), instrument="MIRI", spectrum=spectrum))
    (out,) = run_step(StitchCubes, paths, tmp_path, params={"rescale": True, "reference": "1-short",
                                                           "features": {"gas_lines": ["H2"]}})
    rec = read_joint_records(out)
    assert out.name == f"jw001-o001_t001_miri_{JOINT_SUFFIX}.fits"
    assert [s.strip() for s in rec["segments"]["label"]] == ["1-short", "1-medium", "1-long"]
    assert list(rec["segments"]["scale"]) == pytest.approx([1.0, 1 / 1.1, 1 / 1.21], rel=2e-3)
    assert list(rec["overlaps"]["crossover"]) == pytest.approx([5.545, 6.045])
    assert rec["header"]["CHANNEL"] == "1" and rec["header"]["BAND"] == "MULTIPLE"
    # the planes under the H2 line were left out of the ratio of the first pair
    (unmasked,) = run_step(StitchCubes, paths, tmp_path / "unmasked", params={"rescale": True, "reference": "1-short"})
    assert int(rec["overlaps"]["n_samples"][0]) < int(read_joint_records(unmasked)["overlaps"]["n_samples"][0])
    with pytest.raises(ValueError, match="not among"):
        run_step(StitchCubes, paths, tmp_path / "x", params={"rescale": True, "reference": "2-short"})
    with pytest.raises(ValueError, match="need 2 crossovers"):
        run_step(StitchCubes, paths, tmp_path / "y", params={"crossovers": [5.5]})


def test_no_wavelength_a_segment_holds_is_left_out(tmp_path: Path):
    a, b, grid_a, _ = two_cubes(tmp_path)                        # 2.00-2.99 and 2.70-4.28 um
    for crossover in (2.6, 3.2):                                 # outside the overlap: one segment would be cut short
        with pytest.raises(ValueError, match="outside the overlap"):
            run_step(StitchCubes, [a, b], tmp_path / f"c{crossover}", params={"crossovers": [crossover]})
    run_step(StitchCubes, [a, b], tmp_path / "edge", params={"crossovers": [2.70]})          # its edge is inside
    inside = write_cube(tmp_path / "jw001-o001_t001_nirspec_g235m-f170lp_s3d.fits", grid_a, np.arange(30) * 0.01 + 2.2,
                        source_image(grid_a), band=("G235M", "F170LP"), margin=2)
    with pytest.raises(ValueError, match="lies inside the range"):
        run_step(StitchCubes, [a, inside, b], tmp_path / "nested")
    # segments that do not overlap are joined across the gap; nothing is measured there
    far = write_cube(tmp_path / "jw001-o001_t001_nirspec_g395m-f290lp_s3d.fits", grid_a, np.arange(30) * 0.02 + 4.6,
                     source_image(grid_a), band=("G395M", "F290LP"), margin=2)
    (out,) = run_step(StitchCubes, [a, far], tmp_path / "gap")
    rec = read_joint_records(out)
    assert int(rec["segments"]["n_planes"].sum()) == len(WAVE_A) + 30 and not np.isfinite(rec["overlaps"]["ratio"][0])
    with pytest.raises(ValueError, match="outside the gap"):
        run_step(StitchCubes, [a, far], tmp_path / "gap2", params={"crossovers": [2.5]})


def test_groups_and_refusals(tmp_path: Path):
    a, b, grid_a, _ = two_cubes(tmp_path)
    lone = write_cube(tmp_path / "jw001-o002_t001_miri_ch1-short_s3d.fits", grid_a, np.arange(20) * 0.01 + 5.0,
                      source_image(grid_a), band=("1", "SHORT"), instrument="MIRI")
    outputs = run_step(StitchCubes, [a, b, lone], tmp_path)      # the lone MIRI cube: nothing to join, a warning
    assert [o.name for o in outputs] == [f"jw001-o001_t001_nirspec_{JOINT_SUFFIX}.fits"]
    with pytest.raises(ValueError, match="no group with at least two cubes"):
        run_step(StitchCubes, [a, lone], tmp_path / "none")
    assert joint_grid_of([a, b]).shape == (21, 21)               # what cube_build_matched would prescribe
    assert read_segment(a).label == "g235h-f170lp" and data_bounds(read_segment(a).valid) == (2, 22, 2, 22)


def test_cube_build_matched_prescribes_the_joint_grid(tmp_path: Path, monkeypatch):
    from jwst.step import CubeBuildStep

    from jwstflow.cubes.stitching.matched import CubeBuildMatched

    fine, coarse = grid(0.13, (21, 21)), grid(0.35, (15, 15), east=0.3, north=-0.2)
    cubes = tmp_path / "calwebb_spec3"
    cubes.mkdir()
    for g, band, start in ((fine, ("1", "SHORT"), 5.0), (coarse, ("4", "LONG"), 24.0)):
        write_cube(cubes / f"jw001_miri_ch{band[0]}-{band[1].lower()}_s3d.fits", g, np.arange(8) * 0.01 + start,
                   source_image(g, fwhm=0.8), band=band, instrument="MIRI")
    seen: dict[str, object] = {}

    def record(self, input_data):
        seen.update({k: getattr(self, k) for k in ("scalexy", "ra_center", "dec_center", "nspax_x", "nspax_y",
                                                   "cube_pa", "coord_system", "output_type")})
        return input_data

    monkeypatch.setattr(CubeBuildStep, "process", record)
    step = CubeBuildMatched(grid_from=str(cubes))
    assert step.process("the association") == "the association"
    expected = joint_grid([fine, coarse])
    assert seen["scalexy"] == pytest.approx(0.13) and (seen["ra_center"], seen["dec_center"]) == fine.crval
    assert (seen["nspax_x"], seen["nspax_y"]) == (expected.shape[1], expected.shape[0])
    assert seen["cube_pa"] is None and seen["coord_system"] == "skyalign" and seen["output_type"] == "band"
    with pytest.raises(ValueError, match="grid_from"):
        CubeBuildMatched().process("x")
    with pytest.raises(FileNotFoundError):
        CubeBuildMatched(grid_from=str(tmp_path / "nowhere")).process("x")


# --------------------------------------------------------------------------- the figures, and what they share
def test_qa_cube_stitch_draws_the_figures(tmp_path: Path):
    a, b, _, _ = two_cubes(tmp_path, grid_b=grid(0.2, (17, 17), east=0.04))
    (out,) = run_step(StitchCubes, [a, b], tmp_path, params={"rescale": True})
    figures = run_step(QaCubeStitch, [out], tmp_path, stage="qa_cube_stitch")
    assert sorted(f.name.rsplit("_", 1)[1] for f in figures) == ["overlaps.png", "positions.png", "spectrum.png"]
    assert all(f.stat().st_size > 10_000 for f in figures)
    # with several crossovers the centroid differences get a figure of their own
    g = grid(0.13, (21, 21))
    bands = [write_cube(tmp_path / f"jw002_miri_ch1-{band.lower()}_s3d.fits", g, np.arange(60) * 0.01 + start,
                        source_image(g, east=shift, fwhm=0.5), band=("1", band), instrument="MIRI")
             for band, start, shift in (("SHORT", 5.0, 0.0), ("MEDIUM", 5.5, 0.03), ("LONG", 6.0, 0.01))]
    (joint,) = run_step(StitchCubes, bands, tmp_path / "miri")
    names = sorted(f.name.rsplit("_", 1)[1] for f in run_step(QaCubeStitch, [joint], tmp_path / "miri", stage="qa"))
    assert names == ["offsets.png", "overlaps.png", "positions.png", "spectrum.png"]


def test_stock_cube_figures_read_a_joint_cube(tmp_path: Path):
    from jwstflow import qa
    from jwstflow.qa.steps import QaCubeSlices, QaCubeSummary

    a, b, _, _ = two_cubes(tmp_path)
    (out,) = run_step(StitchCubes, [a, b], tmp_path)
    cube = qa.read_cube(out)
    assert cube["sky"] is not None and cube["sky"].north_up      # the celestial axes, despite the tabular wavelength axis
    assert len(cube["wave"]) == cube["data"].shape[0] and np.all(np.diff(cube["wave"]) > 0)
    assert run_step(QaCubeSummary, [out], tmp_path, stage="s")[0].stat().st_size > 10_000
    assert run_step(QaCubeSlices, [out], tmp_path, stage="c")[0].stat().st_size > 10_000


def test_overlap_helpers():
    assert overlap_midpoints([(1.0, 2.0), (1.8, 3.0), (3.2, 4.0)]) == pytest.approx([1.9, 3.1])
    # a scale is positive: a ratio that is not is no link of the chain
    scales, errors = chain_scales(["a", "b", "c"], [OverlapRatio(-0.4, 0.1, (1.0, 2.0), 9), OverlapRatio(2.0, 0.2, (2.0, 3.0), 9)], "a")
    assert scales == pytest.approx([1.0, 1.0, 0.5]) and errors == pytest.approx([0.0, 0.0, 0.05])
    wave, flux = np.linspace(1.0, 2.0, 11), np.ones(11)
    masked = mask_windows(wave, flux, [(1.25, 1.45)])
    assert np.isnan(masked[3:5]).all() and np.isfinite(masked).sum() == 9
    assert mask_windows(wave, flux, []) is flux and mask_windows(wave, flux, [(5.0, 6.0)]) is flux
    # windows covering the whole overlap leave the flux alone: better a contaminated ratio than none
    assert mask_windows(wave, flux, [(1.5, 2.1)], within=(1.55, 2.0)) is flux
    assert np.isnan(mask_windows(wave, flux, [(1.5, 1.75)], within=(1.55, 2.0))).sum() == 3
