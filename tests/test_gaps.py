"""The detector gap (jwstflow.cubes.gaps): window, coverage, staircase, the cross-exposure test, product, QA.

The tests build a small slicer with a real ``gwcs``: ``N_SLICES`` slices of
``N_ALONG`` pixels side by side on the sky (0.1 arcsec apart, like the
NIRSpec IFU), every slice dispersed over ``N_COL`` columns of NRS1 and of
NRS2. Slice ``s`` leaves NRS1 at ``EDGE + s * STAGGER`` and lands on NRS2 at
``START + s * STAGGER`` -- the staircase of the real instrument in miniature.
Dither ``k`` is the same exposure shifted ``k`` slice widths East, so sky
slice X is seen through slices X, X-1, X-2, ... whose edges lie one STAGGER
apart: the staircase, and which other exposures can check each of its
pixels, follow from the geometry. Every exposure sees the same scene -- a
continuum with an emission line next to the gap -- so a real line must
survive and only an artefact injected into one exposure may be flagged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from jwstflow.cubes.gaps import (
    GAPCOV_SUFFIX,
    FlagGapOutliers,
    coverage_limits,
    exposure_of,
    gap_limits_at,
    gap_window,
    group_name,
    read_gap_product,
    slice_edges,
)
from jwstflow.qa.steps import QaGapCoverage
from jwstflow.testing import check_step, run_step

N_SLICES, N_ALONG, N_COL = 8, 6, 40
SLICE = 0.1                      # arcsec: slice width = pixel size along the slice
STEP = 0.001                     # um per column
STAGGER = 0.003                  # um between the gap edges of neighbouring slices (3 columns)
EDGE, START = 2.40, 2.45         # um: slice 0 leaves NRS1 / lands on NRS2
RA0, DEC0 = 169.0, -20.0
DO_NOT_USE, OUTLIER = 1, 16
LINE, LINE_WIDTH = EDGE + 2 * STAGGER, 0.0015    # an emission line in the staircase of the blue side


def scene(lam: np.ndarray) -> np.ndarray:
    """What every exposure sees [MJy/sr]: a continuum of 1 and a line peaking at 6."""
    return 1.0 + 5.0 * np.exp(-0.5 * ((lam - LINE) / LINE_WIDTH) ** 2)


def slicer_wcs(detector: str, east_shift: float = 0.0):
    """(x, y) -> (RA, Dec, wavelength [um]) of the toy slicer; ``east_shift`` [arcsec] is the dither."""
    from astropy import units as u
    from astropy.coordinates import ICRS
    from astropy.modeling import models
    from gwcs import WCS
    from gwcs import coordinate_frames as cf

    rows = np.arange(N_SLICES * N_ALONG, dtype=float)
    s, j = np.divmod(rows.astype(int), N_ALONG)
    east = s * SLICE + east_shift
    north = j * SLICE
    if detector == "NRS1":
        base = EDGE + s * STAGGER - (N_COL - 1) * STEP       # the last column holds the slice's edge
    else:
        base = START + s * STAGGER                           # the first column holds its start
    tab = {name: models.Tabular1D(points=rows, lookup_table=np.asarray(v, float), bounds_error=False,
                                  fill_value=np.nan, name=name)
           for name, v in (("east", east), ("north", north), ("base", base))}
    wave = (models.Mapping((0,), n_inputs=2) | models.Scale(STEP)) + (models.Mapping((1,), n_inputs=2) | tab["base"])
    sky = ((models.Scale(1 / 3600.0) & models.Scale(1 / 3600.0)) | models.Pix2Sky_TAN()
           | models.RotateNative2Celestial(RA0, DEC0, 180.0))
    forward = models.Mapping((1, 1, 0, 1)) | (tab["east"] & tab["north"] & wave) | (sky & models.Identity(1))
    detector_frame = cf.Frame2D(name="detector", axes_order=(0, 1), unit=(u.pix, u.pix))
    world = cf.CompositeFrame([cf.CelestialFrame(name="sky", reference_frame=ICRS(), axes_order=(0, 1)),
                               cf.SpectralFrame(name="spectral", axes_order=(2,), unit=(u.micron,),
                                                axes_names=("wavelength",))], name="world")
    w = WCS([(detector_frame, forward), (world, None)])
    w.bounding_box = ((-0.5, N_COL - 0.5), (-0.5, N_SLICES * N_ALONG - 0.5))
    return w


def cal_file(directory: Path, exposure: int, detector: str, *, east_shift: float = 0.0, grating: str = "G235H",
             filt: str = "F170LP", observation: str = "005", wavelength_image: bool = True,
             exp_type: str = "NRS_IFU", activity: str = "03101", spikes: dict | None = None) -> Path:
    """A toy NIRSpec IFU cal file with its gwcs (and WAVELENGTH/REGIONS images)."""
    from stdatamodels.jwst import datamodels

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"jw01234{observation}001_{activity}_{exposure:05d}_{detector.lower()}_cal.fits"
    shape = (N_SLICES * N_ALONG, N_COL)
    hdr = fits.Header({"TELESCOP": "JWST", "INSTRUME": "NIRSPEC", "DETECTOR": detector, "EXP_TYPE": exp_type,
                       "GRATING": grating, "FILTER": filt, "PROGRAM": "01234", "OBSERVTN": observation,
                       "VISIT": "001", "BKGDTARG": False, "PATT_NUM": exposure, "TARG_RA": RA0, "TARG_DEC": DEC0,
                       "DATAMODL": "IFUImageModel"})
    w = slicer_wcs(detector, east_shift)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    lam = np.asarray(w(xx.astype(float), yy.astype(float))[2], dtype="f4")
    sci = scene(lam.astype(float)).astype("f4")
    for (y, x), extra in (spikes or {}).items():
        sci[y, x] += extra
    hdus = [fits.PrimaryHDU(header=hdr),
            fits.ImageHDU(sci, name="SCI"),
            fits.ImageHDU(np.full(shape, 0.1, "f4"), name="ERR"),
            fits.ImageHDU(np.zeros(shape, "u4"), name="DQ")]
    if wavelength_image:
        hdus.append(fits.ImageHDU(lam, name="WAVELENGTH"))
        hdus.append(fits.ImageHDU((yy // N_ALONG + 1).astype("i4"), name="REGIONS"))
    fits.HDUList(hdus).writeto(path, overwrite=True)
    with datamodels.open(path) as model:
        model.meta.wcs = w
        model.save(str(path))
    return path


def dithers(directory: Path, n: int = 4, spikes: dict | None = None, **kwargs) -> list[Path]:
    """``n`` exposures, each shifted one slice width further East, both detectors; ``spikes`` maps
    (exposure, detector) to {(y, x): added value}."""
    return [cal_file(directory, k + 1, det, east_shift=k * SLICE, spikes=(spikes or {}).get((k + 1, det)), **kwargs)
            for k in range(n) for det in ("NRS1", "NRS2")]


def flagged(path: Path) -> np.ndarray:
    return (fits.getdata(path, "DQ") & DO_NOT_USE) > 0


def staircase_column(edge_offset: int) -> int:
    """The NRS1 column of a slice holding the wavelength ``edge_offset`` STAGGERs below its own edge."""
    return N_COL - 1 - edge_offset * round(STAGGER / STEP)


def test_declarations_are_valid():
    for cls in (FlagGapOutliers, QaGapCoverage):
        _, problems = check_step(cls)
        assert problems == [], f"{cls.__name__}: {problems}"


def test_detector_sky_coordinates_reaches_the_last_columns(tmp_path: Path):
    """The coarse column search must not stop short of the box's end (the NRS1 gap edge)."""
    from stdatamodels.jwst import datamodels

    from jwstflow.cubes import detector_sky_coordinates

    with datamodels.open(cal_file(tmp_path, 1, "NRS1")) as model:
        cols = np.concatenate([p[1] for p in detector_sky_coordinates(model, [(EDGE - 0.002, EDGE + 1.0)])])
    assert cols.max() == N_COL - 1


def test_names_and_window(tmp_path: Path):
    assert exposure_of("jw05299004001_03101_00001_nrs1_cal.fits") == "jw05299004001_03101_00001"
    assert exposure_of("jw05299004001_03101_00001_nrs2_o004_crf.fits") == "jw05299004001_03101_00001"
    hdr = fits.Header({"PROGRAM": "05299", "OBSERVTN": "004", "INSTRUME": "NIRSPEC", "GRATING": "G235H",
                       "FILTER": "F170LP", "BKGDTARG": True})
    assert group_name(hdr, ["PROGRAM", "OBSERVTN", "GRATING", "FILTER", "BKGDTARG"]) == "jw05299-o004_nirspec_g235h-f170lp_bkg"
    p1 = cal_file(tmp_path, 1, "NRS1")
    p2 = cal_file(tmp_path, 1, "NRS2")
    e1, e2 = slice_edges(p1), slice_edges(p2)
    assert np.allclose(np.sort(e1["edges"]), EDGE + np.arange(N_SLICES) * STAGGER)
    assert np.allclose(np.sort(e2["edges"]), START + np.arange(N_SLICES) * STAGGER)
    assert e1["dispersion"] == pytest.approx(STEP, rel=1e-3)          # float32 wavelength image
    lo, hi = gap_window(e1["edges"], e2["edges"], STEP)
    assert lo == pytest.approx(EDGE - 3 * STEP, abs=1e-5)
    assert hi == pytest.approx(START + 7 * STAGGER + 3 * STEP, abs=1e-5)


def test_physical_flux_is_never_flagged(tmp_path: Path):
    """Four dithers of one scene with a line in the staircase: every staircase pixel that two other
    exposures can check agrees with them, the rest cannot be checked -- nothing is flagged, nothing
    is lost."""
    files = dithers(tmp_path / "in")
    outs = run_step(FlagGapOutliers, files, tmp_path, stage="flag_gap_outliers")
    cals = {p.name: p for p in outs if p.name.endswith("_cal.fits")}
    assert len(cals) == len(files) and all(not flagged(p).any() for p in cals.values())
    (prod,) = [p for p in outs if p.name.endswith(f"_{GAPCOV_SUFFIX}.fits")]
    hdr = fits.getheader(prod)
    assert hdr["JWFGPNPX"] == 0
    assert hdr["JWFGPNST"] > hdr["JWFGPNTS"] > 0          # some staircase pixels tested, some not testable
    for src in files:
        with fits.open(src) as a, fits.open(cals[src.name]) as b:
            assert np.array_equal(a["SCI"].data, b["SCI"].data)


def test_an_artefact_in_one_exposure_is_flagged_and_nothing_else(tmp_path: Path):
    """Dither 1 sees sky slice 5 through its slice 5; at slice 3's edge (6 columns before its own)
    dithers 2 and 3 cover the same position and wavelength, so a spike there is caught -- and only
    that pixel. The same spike where only one other exposure reaches stays (untestable)."""
    row = 5 * N_ALONG + 2
    testable = (row, staircase_column(2))           # wavelength edge(3): dithers 2 and 3 reach it
    untestable = (row, staircase_column(1))         # wavelength edge(4): only dither 2 reaches it
    files = dithers(tmp_path / "in", spikes={(1, "NRS1"): {testable: 50.0, untestable: 50.0}})
    outs = run_step(FlagGapOutliers, files, tmp_path, stage="flag_gap_outliers")
    by_name = {p.name: p for p in outs}
    for src in files:
        got = flagged(by_name[src.name])
        if "_00001_nrs1" in src.name:
            want = np.zeros_like(got)
            want[testable] = True
            assert np.array_equal(got, want), np.argwhere(got)
            assert fits.getdata(by_name[src.name], "DQ")[testable] & OUTLIER
            assert fits.getheader(by_name[src.name])["JWFGPNPX"] == 1
        else:
            assert not got.any(), src.name


def test_one_exposure_has_no_staircase(tmp_path: Path):
    files = [cal_file(tmp_path / "in", 1, det) for det in ("NRS1", "NRS2")]
    outs = run_step(FlagGapOutliers, files, tmp_path, stage="flag_gap_outliers")
    (prod,) = [p for p in outs if p.name.endswith(f"_{GAPCOV_SUFFIX}.fits")]
    assert fits.getheader(prod)["JWFGPNST"] == 0
    assert not any(flagged(p).any() for p in outs if p.name.endswith("_cal.fits"))


def test_product_records_the_coverage_and_answers_lookups(tmp_path: Path):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.modeling import models

    outs = run_step(FlagGapOutliers, dithers(tmp_path / "in"), tmp_path, stage="flag_gap_outliers")
    (path,) = [p for p in outs if p.name.endswith(f"_{GAPCOV_SUFFIX}.fits")]
    prod = read_gap_product(path)
    assert prod["header"]["JWFGPNEX"] == 4
    assert len(prod["files"]) == 8 and set(prod["files"]["detector"]) == {"NRS1", "NRS2"}
    assert int(np.sum(prod["files"]["n_staircase"])) == prod["header"]["JWFGPNST"]
    covered = prod["nexplo"] > 0
    assert np.all(prod["limlo"][covered] <= prod["reachlo"][covered] + 1e-12)
    assert np.all(prod["limhi"][prod["nexphi"] > 0] >= prod["reachhi"][prod["nexphi"] > 0] - 1e-12)
    # sky slice 5, mid-way along it: seen through slices 5, 4, 3, 2 -- all four reach slice 2's edge
    # before the gap and slice 5's start after it
    sky = (models.Scale(1 / 3600.0) & models.Scale(1 / 3600.0)) | models.Pix2Sky_TAN() | models.RotateNative2Celestial(RA0, DEC0, 180.0)
    ra, dec = sky(5 * SLICE, 2 * SLICE)
    lo, hi = gap_limits_at(path, ra, dec)
    assert float(lo) == pytest.approx(EDGE + 2 * STAGGER, abs=1e-6)
    assert float(hi) == pytest.approx(START + 5 * STAGGER, abs=1e-6)
    far = SkyCoord(RA0 * u.deg, DEC0 * u.deg).directional_offset_by(0 * u.deg, 1 * u.arcmin)
    assert np.isnan(gap_limits_at(path, far.ra.deg, far.dec.deg)[0])


def test_without_wavelength_images_the_wcs_gives_the_same_answer(tmp_path: Path):
    spikes = {(1, "NRS1"): {(5 * N_ALONG + 2, staircase_column(2)): 50.0}}
    a = run_step(FlagGapOutliers, dithers(tmp_path / "a", spikes=spikes), tmp_path, stage="with_images")
    b = run_step(FlagGapOutliers, dithers(tmp_path / "b", spikes=spikes, wavelength_image=False), tmp_path,
                 stage="from_wcs")
    fa = {p.name: flagged(p) for p in a if p.name.endswith("_cal.fits")}
    fb = {p.name: flagged(p) for p in b if p.name.endswith("_cal.fits")}
    assert fa.keys() == fb.keys()
    assert all(np.array_equal(fa[k], fb[k]) for k in fa)
    assert sum(int(v.sum()) for v in fa.values()) == 1


def test_groups_are_kept_apart_and_odd_files_pass_through(tmp_path: Path):
    files = dithers(tmp_path / "in")                                            # G235H, both detectors
    other = [cal_file(tmp_path / "in", 1, "NRS1", grating="G395H", filt="F290LP", activity="03103"),
             cal_file(tmp_path / "in", 2, "NRS1", grating="G395H", filt="F290LP", activity="03103",
                      east_shift=SLICE)]                                       # G395H without NRS2: no gap
    slit = cal_file(tmp_path / "in", 7, "NRS1", exp_type="NRS_FIXEDSLIT")      # not an IFU exposure
    outs = run_step(FlagGapOutliers, files + other + [slit], tmp_path, stage="flag_gap_outliers")
    names = sorted(p.name for p in outs)
    assert len([n for n in names if n.endswith("_cal.fits")]) == len(files) + 3
    assert [n for n in names if n.endswith(f"_{GAPCOV_SUFFIX}.fits")] == ["jw01234-o005_nirspec_g235h-f170lp_gapcov.fits"]
    for p in outs:
        if "_03103_" in p.name or "_00007_" in p.name:
            assert not flagged(p).any()
            assert fits.getheader(p)["JWFGPPRD"] == "none"


def test_coverage_limits_alone():
    """The core on bare samples: one exposure reaching 2.40 everywhere, one reaching 2.39."""
    def sample(exposure, wave_max):
        east = np.repeat(np.arange(5) * SLICE, 5)
        north = np.tile(np.arange(5) * SLICE, 5)
        ra = RA0 + east / 3600.0 / np.cos(np.radians(DEC0))
        dec = DEC0 + north / 3600.0
        lam = np.full(east.size, wave_max)
        return {"exposure": exposure, "detector": "NRS1", "ra": ra, "dec": dec, "wave": lam}
    lim = coverage_limits([sample("a", 2.40), sample("b", 2.39)])
    cov = lim["nexplo"] == 2
    assert cov.any() and np.allclose(lim["limlo"][cov], 2.39) and np.allclose(lim["reachlo"][cov], 2.40)
    lim = coverage_limits([sample("a", 2.40), sample("b", 2.39)], fraction=0.5)
    assert np.allclose(lim["limlo"][cov], 2.40)


def test_qa_figure_renders(tmp_path: Path):
    outs = run_step(FlagGapOutliers, dithers(tmp_path / "in"), tmp_path, stage="flag_gap_outliers")
    (prod,) = [p for p in outs if p.name.endswith(f"_{GAPCOV_SUFFIX}.fits")]
    (png,) = run_step(QaGapCoverage, [prod], tmp_path, stage="qa_gap_coverage")
    assert png.suffix == ".png" and png.stat().st_size > 10_000
