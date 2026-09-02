"""Mock observations and stub pipelines: run a *complete* workflow in seconds.

The unit-level tools in :mod:`jwstflow.testing` execute one step on synthetic
products. This module tests the level above: a whole workflow YAML through the
real engine (config loading, input discovery, associations, task planning,
checkpointing, product naming), with only the expensive official pipeline
stages replaced by fast, scene-faithful stand-ins. No network, no CRDS cache,
no MAST -- a full NIRSpec-IFU or MIRI-MRS reduction runs in seconds.

Three pieces:

* **Mock observations** -- :func:`nirspec_ifu_observation` /
  :func:`miri_mrs_observation` describe an observation exactly like a real one
  (DMS file names, header keywords, dither/detector/grating structure, MAST
  download log), modelled on the MIDAS ESO-Ha 569 program (jw01751), but with
  strongly reduced data sizes: a few dithers, tens of spectral planes instead
  of thousands, ~30 px fields. The astrophysical scene (:class:`MockScene`:
  an edge-on disk and/or point source, flat sky, emission-line halos, a hot
  pixel) travels in ``JWFMK*`` header keywords from the ``_uncal`` files all
  the way to the cubes, so every product is deterministic and assertable.
  :func:`write_observation` materialises the ``_uncal`` files plus the MAST
  observation log (the source of the DMS target id).

* **Stub pipelines** -- :class:`StubDetector1`, :class:`StubSpec2` and
  :class:`StubSpec3` mimic the official pipelines' *interfaces*: same stage
  names (``calwebb_detector1`` ...), same product names (``_rate``, ``_cal``,
  DMS level-3 cube names with the band appended), consuming the same
  association files, honouring the parameters that shape the data flow
  (``steps.extract_1d.skip``, background members, ``background_dir``).
  The data they write is synthesised from the scene keywords: ``spec2``
  subtracts the sky of associated background members like ``bkg_subtract``
  would, ``spec3`` renders the scene into IFU cubes, and
  ``spec3_with_background`` subtracts the measured ``*_bkgspec.fits`` sky
  like ``master_background`` would. Custom (jwstflow / contributed) steps
  then run *for real* on those cubes.

* **Runners** -- :func:`stub_pipelines` shadows the official step aliases in
  the in-process registry (use the serial backend / ``workers=1``, where
  tasks run in this process; by dotted path,
  ``jwstflow.testing.mock:StubDetector1`` etc. also work in spawned workers),
  :func:`offline_overrides` pins the CRDS context and disables everything
  that would touch the network, and :func:`run_mock_workflow` bundles both
  around ``load_config`` + ``Runner`` for one-line integration tests::

      write_observation(work / "reductions/eso-ha-569/raw", nirspec_ifu_observation())
      cfg, summary = run_mock_workflow("reductions/eso-ha-569/nirspec_ifu.yaml", work)
      assert summary.ok

Everything here is deliberately import-light at module import time (numpy
only); astropy / stdatamodels load inside the functions that need them, like
the rest of jwstflow.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..steps.base import _REGISTRY, RunContext, Step

__all__ = [
    "MockScene", "MockExposure", "MockObservation",
    "nirspec_ifu_observation", "miri_mrs_observation", "write_observation",
    "star_field", "write_gaia_catalog",
    "StubDetector1", "StubSpec2", "StubSpec3", "StubSpec3WithBackground",
    "StubImage2", "StubImage3",
    "stub_pipelines", "offline_overrides", "run_mock_workflow",
    "NRS_GRATINGS", "MRS_BANDS",
]

# --------------------------------------------------------------------------- instrument tables

#: NIRSpec IFU high-resolution configurations -> wavelength coverage [um].
NRS_GRATINGS: dict[tuple[str, str], tuple[float, float]] = {
    ("G140H", "F100LP"): (0.97, 1.82),
    ("G235H", "F170LP"): (1.66, 3.05),
    ("G395H", "F290LP"): (2.87, 5.27),
    ("PRISM", "CLEAR"): (0.60, 5.30),
}

#: MIRI MRS (channel, band) -> wavelength coverage [um] (values from the MIRI docs).
MRS_BANDS: dict[tuple[str, str], tuple[float, float]] = {
    ("1", "SHORT"): (4.90, 5.74), ("1", "MEDIUM"): (5.66, 6.63), ("1", "LONG"): (6.53, 7.65),
    ("2", "SHORT"): (7.51, 8.77), ("2", "MEDIUM"): (8.67, 10.13), ("2", "LONG"): (10.02, 11.70),
    ("3", "SHORT"): (11.55, 13.47), ("3", "MEDIUM"): (13.34, 15.57), ("3", "LONG"): (15.41, 17.98),
    ("4", "SHORT"): (17.70, 20.95), ("4", "MEDIUM"): (20.69, 24.48), ("4", "LONG"): (24.19, 27.90),
}

#: MRS detector -> the two channels it records simultaneously.
MRS_DETECTOR_CHANNELS: dict[str, tuple[str, str]] = {"MIRIFUSHORT": ("1", "2"), "MIRIFULONG": ("3", "4")}

#: MRS channel -> cube spaxel scale [arcsec] (the real cube_build defaults; the growing
#: scale is what keeps sky annuli inside the FOV out to channel 4).
MRS_PIX_ARCSEC: dict[str, float] = {"1": 0.13, "2": 0.17, "3": 0.20, "4": 0.35}


# --------------------------------------------------------------------------- the scene


@dataclass(frozen=True)
class MockScene:
    """The astrophysical content of a mock observation, in cube units (MJy/sr).

    The scene is rendered per wavelength plane: a flat ``background``, an
    edge-on disk (elongated Gaussian of peak ``disk_peak``), an optional point
    source of total flux ``point_flux_jy``, and emission ``lines`` --
    ``(wavelength_um, relative_amplitude)`` pairs that add a spatially wider
    halo (disk scenes) or a brighter point source (point-source scenes) inside
    ``line_width_um`` of the line. A ``hot_pixel`` far from the target tests
    that masks/cleaning never pick up isolated outliers. All of it round-trips
    through FITS headers (:meth:`to_header` / :meth:`from_header`) so the stub
    pipelines can re-render the identical scene at any stage.
    """

    background: float = 1.0
    disk_peak: float = 1000.0
    disk_sigma_maj: float = 6.0     # px, along the disk midplane (x)
    disk_sigma_min: float = 2.0     # px
    point_flux_jy: float = 0.0
    point_fwhm_pix: float = 2.5
    lines: tuple[tuple[float, float], ...] = ()
    line_width_um: float = 0.02
    line_halo_scale: float = 1.6    # halo sigmas = scale x disk sigmas
    #: source position offset from the field centre [arcsec along +x / +y] -- emulates a
    #: pointing error the aperture placement must absorb (hot pixel stays put)
    offset_x_arcsec: float = 0.0
    offset_y_arcsec: float = 0.0
    hot_pixel: tuple[int, int] | None = (3, 3)
    hot_value: float = 3000.0
    noise: float = 0.0
    seed: int = 0

    # -- rendering -----------------------------------------------------------
    def plane(self, size: int, pix_arcsec: float, wave_um: float) -> np.ndarray:
        """One (size, size) surface-brightness image [MJy/sr] at ``wave_um``."""
        yy, xx = np.mgrid[:size, :size]
        c_x = (size - 1) / 2.0 + self.offset_x_arcsec / pix_arcsec
        c_y = (size - 1) / 2.0 + self.offset_y_arcsec / pix_arcsec
        img = np.full((size, size), float(self.background))
        boost = 0.0
        for wave0, amp in self.lines:
            if abs(wave_um - wave0) <= self.line_width_um:
                boost += float(amp)
        if self.disk_peak > 0:
            disk = np.exp(-((xx - c_x) ** 2 / (2 * self.disk_sigma_maj**2)
                            + (yy - c_y) ** 2 / (2 * self.disk_sigma_min**2)))
            img += self.disk_peak * disk
            if boost:
                s_maj, s_min = self.line_halo_scale * self.disk_sigma_maj, self.line_halo_scale * self.disk_sigma_min
                img += boost * np.exp(-((xx - c_x) ** 2 / (2 * s_maj**2) + (yy - c_y) ** 2 / (2 * s_min**2)))
        if self.point_flux_jy > 0:
            sig = self.point_fwhm_pix / 2.3548
            area_sr = (pix_arcsec / 206265.0) ** 2
            peak = self.point_flux_jy / (2 * np.pi * sig**2) / area_sr / 1e6
            psf = np.exp(-((xx - c_x) ** 2 + (yy - c_y) ** 2) / (2 * sig**2))
            img += peak * (1.0 + boost) * psf
        if self.hot_pixel is not None:
            img[self.hot_pixel] += self.hot_value
        return img

    def cube(self, wave_um: np.ndarray, size: int, pix_arcsec: float) -> np.ndarray:
        """The full (nwave, size, size) scene cube [MJy/sr], with optional noise."""
        data = np.stack([self.plane(size, pix_arcsec, float(w)) for w in np.asarray(wave_um)])
        if self.noise > 0:
            data = data + np.random.default_rng(self.seed).normal(0.0, self.noise, data.shape)
        return data.astype("f4")

    # -- header round trip ----------------------------------------------------
    def to_header(self) -> dict[str, Any]:
        lines = ",".join(f"{w:.6g}:{a:.6g}" for w, a in self.lines)
        return {
            "JWFMKBG": (self.background, "mock scene: sky background [MJy/sr]"),
            "JWFMKDP": (self.disk_peak, "mock scene: disk peak [MJy/sr]"),
            "JWFMKDA": (self.disk_sigma_maj, "mock scene: disk major sigma [px]"),
            "JWFMKDB": (self.disk_sigma_min, "mock scene: disk minor sigma [px]"),
            "JWFMKPF": (self.point_flux_jy, "mock scene: point-source flux [Jy]"),
            "JWFMKPW": (self.point_fwhm_pix, "mock scene: point-source FWHM [px]"),
            "JWFMKLN": (lines, "mock scene: lines wave:amp,..."),
            "JWFMKLW": (self.line_width_um, "mock scene: line half-width [um]"),
            "JWFMKLS": (self.line_halo_scale, "mock scene: line halo scale"),
            "JWFMKOX": (self.offset_x_arcsec, "mock scene: source offset +x [arcsec]"),
            "JWFMKOY": (self.offset_y_arcsec, "mock scene: source offset +y [arcsec]"),
            "JWFMKHX": (-1 if self.hot_pixel is None else self.hot_pixel[0], "mock scene: hot pixel y (-1: none)"),
            "JWFMKHY": (-1 if self.hot_pixel is None else self.hot_pixel[1], "mock scene: hot pixel x"),
            "JWFMKHV": (self.hot_value, "mock scene: hot pixel value [MJy/sr]"),
            "JWFMKNS": (self.noise, "mock scene: noise sigma [MJy/sr]"),
            "JWFMKSD": (self.seed, "mock scene: noise seed"),
        }

    @classmethod
    def from_header(cls, hdr: Any) -> MockScene:
        lines: tuple[tuple[float, float], ...] = ()
        raw = str(hdr.get("JWFMKLN", "") or "")
        if raw:
            lines = tuple((float(w), float(a)) for w, a in (item.split(":") for item in raw.split(",")))
        hx, hy = int(hdr.get("JWFMKHX", -1)), int(hdr.get("JWFMKHY", -1))
        return cls(
            background=float(hdr.get("JWFMKBG", 0.0)),
            disk_peak=float(hdr.get("JWFMKDP", 0.0)),
            disk_sigma_maj=float(hdr.get("JWFMKDA", 6.0)),
            disk_sigma_min=float(hdr.get("JWFMKDB", 2.0)),
            point_flux_jy=float(hdr.get("JWFMKPF", 0.0)),
            point_fwhm_pix=float(hdr.get("JWFMKPW", 2.5)),
            lines=lines,
            line_width_um=float(hdr.get("JWFMKLW", 0.02)),
            line_halo_scale=float(hdr.get("JWFMKLS", 1.6)),
            offset_x_arcsec=float(hdr.get("JWFMKOX", 0.0)),
            offset_y_arcsec=float(hdr.get("JWFMKOY", 0.0)),
            hot_pixel=None if hx < 0 else (hx, hy),
            hot_value=float(hdr.get("JWFMKHV", 0.0)),
            noise=float(hdr.get("JWFMKNS", 0.0)),
            seed=int(hdr.get("JWFMKSD", 0)),
        )


# --------------------------------------------------------------------------- observations


@dataclass
class MockExposure:
    """One ``_uncal`` file to be written: its DMS name and full primary header."""

    filename: str
    header: dict[str, Any]


@dataclass
class MockObservation:
    """A set of exposures sharing a program/observation, plus the scene and cube geometry."""

    program: int
    observation: int
    target_id: str
    scene: MockScene
    exposures: list[MockExposure] = field(default_factory=list)

    @property
    def obs_ids(self) -> list[str]:
        """MAST-style observation ids recorded in the download log (one per optical config)."""
        seen: dict[str, None] = {}
        for exp in self.exposures:
            inst = str(exp.header["INSTRUME"]).lower()
            seen.setdefault(f"jw{self.program:05d}-o{self.observation:03d}_{self.target_id}_{inst}", None)
        return list(seen)


def _dms_uncal_name(program: int, observation: int, visit: int, visitgrp: int, seq: int, act: int,
                    exposure: int, detector: str) -> str:
    return (f"jw{program:05d}{observation:03d}{visit:03d}_{visitgrp:02d}{seq:d}{act:02d}"
            f"_{exposure:05d}_{detector.lower()}_uncal.fits")


def _base_header(*, instrument: str, detector: str, exp_type: str, program: int, observation: int,
                 visit: int, exposure: int, dither: int, ndithers: int, targprop: str, ra: float,
                 dec: float, background: bool, scene: MockScene, nwave: int, size: int,
                 pix_arcsec: float, ngroups: int, date_obs: str) -> dict[str, Any]:
    hdr: dict[str, Any] = {
        "TELESCOP": "JWST",
        "INSTRUME": instrument,
        "DETECTOR": detector,
        "EXP_TYPE": exp_type,
        "PROGRAM": f"{program:05d}",
        "OBSERVTN": f"{observation:03d}",
        "VISIT": f"{visit:03d}",
        "VISITGRP": "02",
        "SEQ_ID": "1",
        "ACT_ID": "01",
        "EXPOSURE": f"{exposure:05d}",
        "OBS_ID": f"V{program:05d}{observation:03d}{visit:03d}P0000000002101",
        "VISIT_ID": f"{program:05d}{observation:03d}{visit:03d}",
        "TARGPROP": targprop,
        "TARGNAME": targprop,
        "TARG_RA": ra,
        "TARG_DEC": dec,
        "RA_REF": ra,     # pointing reference: what wcs_offset shifts before assign_wcs
        "DEC_REF": dec,
        "BKGDTARG": background,
        "IS_IMPRT": False,
        "TSOVISIT": False,
        "PATT_NUM": dither,
        "NUMDTHPT": ndithers,
        "PATTTYPE": "CYCLING" if instrument == "NIRSPEC" else "4-POINT",
        "NINTS": 1,
        "NGROUPS": ngroups,
        "READPATT": "NRSIRS2RAPID" if instrument == "NIRSPEC" else "FASTR1",
        "DATE-OBS": date_obs,
        "TIME-OBS": "00:00:00",
        "EFFEXPTM": 300.0,
        "DATAMODL": "RampModel",
        "JWFMKNW": (nwave, "mock geometry: cube planes per band"),
        "JWFMKSZ": (size, "mock geometry: cube spatial size [px]"),
        "JWFMKPX": (pix_arcsec, "mock geometry: cube pixel scale [arcsec]"),
    }
    hdr.update(scene.to_header())
    return hdr


def nirspec_ifu_observation(
    *,
    program: int = 1751,
    observation: int = 6,
    visit: int = 1,
    target_id: str = "t005",
    targprop: str = "ESO-HA-569",
    ra: float = 167.7885,
    dec: float = -76.6960,
    gratings: Iterable[tuple[str, str]] = (("G235H", "F170LP"), ("G395H", "F290LP")),
    dithers: int = 2,
    scene: MockScene | None = None,
    nwave: int = 150,
    size: int = 31,
    pix_arcsec: float = 0.1,
    ngroups: int = 3,
) -> MockObservation:
    """A NIRSpec-IFU observation modelled on ESO-Ha 569 (jw01751 obs 6).

    The real observation is 2 gratings x 4 dithers x 2 detectors = 16
    ``_uncal`` files with ~3900-plane cubes; the default mock keeps the same
    structure at 2 dithers and 150 planes. The default scene is a bright
    edge-on disk over a faint sky, with one gas-line halo per grating
    (H2 1-0 S(1) in G235H, H I Pf-delta in G395H) and one hot pixel.
    """
    if scene is None:
        # hot pixel stays a clear outlier but below the disk's *smoothed* peak, so it
        # cannot dominate the disk_mask normalisation (in real data the many-plane
        # median and dither averaging dilute single hot detector pixels the same way)
        scene = MockScene(background=1.0, disk_peak=1000.0, point_flux_jy=0.0, hot_value=3000.0,
                          lines=((2.1218, 400.0), (3.2970, 300.0)), line_width_um=0.03)
    obs = MockObservation(program, observation, target_id, scene)
    exposure = 0
    for grating, filt in gratings:
        if (grating, filt) not in NRS_GRATINGS:
            raise ValueError(f"unknown NIRSpec configuration {grating}/{filt}; known: {sorted(NRS_GRATINGS)}")
        for dither in range(1, dithers + 1):
            exposure += 1
            for detector in ("NRS1", "NRS2"):
                hdr = _base_header(instrument="NIRSPEC", detector=detector, exp_type="NRS_IFU",
                                   program=program, observation=observation, visit=visit, exposure=exposure,
                                   dither=dither, ndithers=dithers, targprop=targprop, ra=ra, dec=dec,
                                   background=False, scene=scene, nwave=nwave, size=size,
                                   pix_arcsec=pix_arcsec, ngroups=ngroups, date_obs="2024-01-01")
                hdr["GRATING"], hdr["FILTER"] = grating, filt
                obs.exposures.append(MockExposure(
                    _dms_uncal_name(program, observation, visit, 2, 1, 1, exposure, detector), hdr))
    return obs


def miri_mrs_observation(
    *,
    program: int = 1751,
    observation: int = 10,
    visit: int = 1,
    target_id: str = "t005",
    targprop: str = "ESO-HA-569",
    ra: float = 167.7885,
    dec: float = -76.6960,
    bands: Iterable[str] = ("SHORT", "MEDIUM", "LONG"),
    dithers: int = 2,
    background: bool = False,
    scene: MockScene | None = None,
    nwave: int = 120,
    size: int = 31,
    pix_arcsec: float = 0.13,
    ngroups: int = 3,
    imager: bool = False,
    pointing_error: tuple[float, float] = (0.35, -0.22),
    n_stars: int = 12,
    imager_size: int = 200,
    imager_pix_arcsec: float = 0.11,
    star_seed: int = 3,
    scene_offset_arcsec: tuple[float, float] = (0.0, 0.0),
) -> MockObservation:
    """A MIRI MRS observation modelled on ESO-Ha 569 (jw01751 obs 10 / 12).

    Each dither of each sub-band exposes both detectors (MIRIFUSHORT carries
    channels 1+2, MIRIFULONG 3+4), so all twelve ch1..ch4 SHORT/MEDIUM/LONG
    band cubes emerge like in the real reduction -- with ~120 planes each
    instead of ~1000+. ``background=True`` makes it a dedicated sky
    observation (``BKGDTARG`` set, sources removed from the default scene).
    The default science scene is a compact source over a bright mid-IR sky
    with two gas lines (H2 0-0 S(7) in ch1-short, [Ne II] in ch3-short).

    ``imager=True`` adds the simultaneous MIRI imager frames (one per dither,
    ``EXP_TYPE=MIR_IMAGE``) carrying a deterministic star field displaced by
    ``pointing_error`` (dRA, dDec on the sky, arcsec) -- the astrometry loop's
    input: image2/image3 stubs resample them into an ``_i2d``, ``gaia_offset``
    measures the injected error against :func:`write_gaia_catalog`'s truth,
    and ``wcs_offset`` applies the correction to the MRS rates.
    """
    if scene is None:
        base = MockScene(background=30.0, disk_peak=0.0, point_flux_jy=0.05, hot_pixel=(2, 2),
                         hot_value=20000.0, lines=((5.5112, 1.5), (12.8135, 1.0)), line_width_um=0.03)
        scene = replace(base, disk_peak=0.0, point_flux_jy=0.0, lines=(), hot_pixel=None) if background else base
    if scene_offset_arcsec != (0.0, 0.0):
        # emulate a pointing error in the cubes: the source sits off the position the
        # WCS claims for it (what mrs_extract's refine_centroid must absorb)
        scene = replace(scene, offset_x_arcsec=scene_offset_arcsec[0],
                        offset_y_arcsec=scene_offset_arcsec[1])
    obs = MockObservation(program, observation, target_id, scene)
    exposure = 0
    for band in bands:
        band = band.upper()
        for dither in range(1, dithers + 1):
            exposure += 1
            for detector in ("MIRIFUSHORT", "MIRIFULONG"):
                hdr = _base_header(instrument="MIRI", detector=detector, exp_type="MIR_MRS",
                                   program=program, observation=observation, visit=visit, exposure=exposure,
                                   dither=dither, ndithers=dithers, targprop=targprop, ra=ra, dec=dec,
                                   background=background, scene=scene, nwave=nwave, size=size,
                                   pix_arcsec=pix_arcsec, ngroups=ngroups, date_obs="2024-02-01")
                hdr["CHANNEL"] = "".join(MRS_DETECTOR_CHANNELS[detector])
                hdr["BAND"] = band
                obs.exposures.append(MockExposure(
                    _dms_uncal_name(program, observation, visit, 2, 1, 1, exposure, detector), hdr))
    if imager:
        for dither in range(1, dithers + 1):
            exposure += 1
            hdr = _base_header(instrument="MIRI", detector="MIRIMAGE", exp_type="MIR_IMAGE",
                               program=program, observation=observation, visit=visit, exposure=exposure,
                               dither=dither, ndithers=dithers, targprop=targprop, ra=ra, dec=dec,
                               background=background, scene=scene, nwave=1, size=imager_size,
                               pix_arcsec=imager_pix_arcsec, ngroups=ngroups, date_obs="2024-02-01")
            hdr["FILTER"] = "F770W"
            hdr.update({
                "JWFMKNST": (n_stars, "mock star field: number of stars"),
                "JWFMKSSD": (star_seed, "mock star field: position seed"),
                "JWFMKPRA": (pointing_error[0], "mock pointing error dRA [arcsec on sky]"),
                "JWFMKPDE": (pointing_error[1], "mock pointing error dDec [arcsec]"),
            })
            obs.exposures.append(MockExposure(
                _dms_uncal_name(program, observation, visit, 2, 1, 1, exposure, "MIRIMAGE"), hdr))
    return obs


# --------------------------------------------------------------------------- the imager star field


def _imager_wcs(hdr: Any):
    """The TAN WCS of a mock imager frame / i2d (centred on the target)."""
    from astropy.wcs import WCS

    size = int(hdr["JWFMKSZ"])
    pix = float(hdr["JWFMKPX"])
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [float(hdr["TARG_RA"]), float(hdr["TARG_DEC"])]
    w.wcs.crpix = [(size + 1) / 2.0, (size + 1) / 2.0]
    w.wcs.cdelt = [-pix / 3600.0, pix / 3600.0]
    return w


def star_field(hdr: Any) -> tuple[Any, Any]:
    """True (ra, dec) arrays of the mock star field an imager header describes."""
    size = int(hdr["JWFMKSZ"])
    rng = np.random.default_rng(int(hdr["JWFMKSSD"]))
    margin = max(12, size // 10)
    xs = rng.uniform(margin, size - margin, int(hdr["JWFMKNST"]))
    ys = rng.uniform(margin, size - margin, int(hdr["JWFMKNST"]))
    ra, dec = _imager_wcs(hdr).all_pix2world(xs, ys, 0)
    return ra, dec


def write_gaia_catalog(path: Path, observation: MockObservation) -> Path:
    """The truth catalogue for the observation's imager star field (gaia_offset's
    ``catalog:`` format: ra/dec/pmra/pmdec), so the astrometry loop runs offline."""
    from astropy.table import Table

    imagers = [e for e in observation.exposures if e.header.get("EXP_TYPE") == "MIR_IMAGE"]
    if not imagers:
        raise ValueError("observation has no imager exposures (pass imager=True)")
    from astropy.io import fits

    hdr = fits.Header()
    for k, v in imagers[0].header.items():
        hdr[k] = v
    ra, dec = star_field(hdr)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Table({"ra": ra, "dec": dec, "pmra": np.zeros(len(ra)), "pmdec": np.zeros(len(ra))}).write(
        path, format="ascii.ecsv", overwrite=True)
    return path


def write_observation(raw_dir: Path, *observations: MockObservation,
                      detector_shape: tuple[int, int] = (24, 24)) -> list[Path]:
    """Write the ``_uncal`` files of one or more observations plus the MAST log.

    The MAST observation log (``mast_observations.json``) is what the runner
    parses the DMS target id (``t005``) from, exactly as after a real
    download; calling this again for another observation merges into it.
    Detector-space data is a small synthetic ramp (its content is never used
    scientifically -- the stubs re-render the scene from the headers).
    """
    from astropy.io import fits

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    log_path = raw_dir / "mast_observations.json"
    rows: list[dict[str, Any]] = []
    if log_path.exists():
        with contextlib.suppress(json.JSONDecodeError):
            rows = json.loads(log_path.read_text())
    known = {r.get("obs_id") for r in rows}
    for obs in observations:
        for exp in obs.exposures:
            hdr = fits.Header()
            for key, value in exp.header.items():
                hdr[key] = value
            ngroups = int(exp.header.get("NGROUPS", 3))
            ny, nx = detector_shape
            level = float(obs.scene.background) + float(obs.scene.disk_peak) / 100.0
            ramp = np.cumsum(np.full((ngroups, ny, nx), level, dtype="f4"), axis=0)
            ramp += np.random.default_rng(obs.scene.seed).normal(0, 0.01, ramp.shape).astype("f4")
            path = raw_dir / exp.filename
            fits.HDUList([fits.PrimaryHDU(header=hdr), fits.ImageHDU(ramp, name="SCI")]).writeto(
                path, overwrite=True)
            written.append(path)
        for obs_id in obs.obs_ids:
            if obs_id not in known:
                rows.append({"obs_id": obs_id, "target_name": obs.exposures[0].header["TARGPROP"]
                             if obs.exposures else "", "mock": True})
                known.add(obs_id)
    log_path.write_text(json.dumps(rows, indent=2))
    return written


# --------------------------------------------------------------------------- stub pipelines


def _member_paths(asn_path: Path, exptype: str = "science") -> list[Path]:
    data = json.loads(Path(asn_path).read_text())
    out: list[Path] = []
    for product in data.get("products", []):
        for member in product.get("members", []):
            if member.get("exptype") != exptype:
                continue
            p = Path(member["expname"])
            out.append(p if p.is_absolute() else Path(asn_path).parent / p)
    return out


def _product_name(asn_path: Path) -> str:
    data = json.loads(Path(asn_path).read_text())
    return str(data["products"][0]["name"])


def _strip_suffix(stem: str) -> str:
    return re.sub(r"_(uncal|rate|cal)$", "", stem)


def _skip_requested(steps_params: Any, step_name: str, default: bool) -> bool:
    """Whether ``parameters.steps.<step_name>.skip`` asks to skip (with a default)."""
    if isinstance(steps_params, dict):
        entry = steps_params.get(step_name)
        if isinstance(entry, dict) and "skip" in entry:
            return bool(entry["skip"])
    return default


class StubDetector1(Step):
    """Stand-in for calwebb_detector1: one ``_rate.fits`` per ``_uncal.fits``.

    The ramp is collapsed to a slope image; every header keyword (the scene
    included) is carried over unchanged, which is all the mock data flow
    needs. Accepts and ignores the official pipeline's ``steps:`` parameters.
    """

    name = "calwebb_detector1"
    level = 1
    batch = "per_file"
    inputs = ("*_uncal.fits",)
    writes_official_products = True
    version = "1"

    def run(self, inputs: list[Path], ctx: RunContext, **params: Any) -> list[Path]:
        from astropy.io import fits

        (src,) = inputs
        out = ctx.output_dir / f"{_strip_suffix(src.stem)}_rate.fits"
        with fits.open(src) as hdul:
            hdr = hdul[0].header.copy()
            data = np.asarray(hdul["SCI"].data, dtype="f4")
        rate = (data[-1] - data[0]) / max(1, data.shape[0] - 1) if data.ndim == 3 else data
        hdr["DATAMODL"] = "ImageModel"
        fits.HDUList([
            fits.PrimaryHDU(header=hdr),
            fits.ImageHDU(rate.astype("f4"), name="SCI"),
            fits.ImageHDU(np.full_like(rate, 0.01, dtype="f4"), name="ERR"),
            fits.ImageHDU(np.zeros(rate.shape, dtype="u4"), name="DQ"),
        ]).writeto(out, overwrite=True)
        ctx.log.info("stub detector1: %s -> %s", src.name, out.name)
        return [out]


class StubSpec2(Step):
    """Stand-in for calwebb_spec2: one ``_cal.fits`` per level-2 association (or rate file).

    Faithful to the parts of the real pipeline that shape the mock data flow:
    when the association carries ``background`` members and
    ``steps.bkg_subtract.skip`` is false, the members' sky (their scene
    background) is subtracted from the science scene, and ``JWFBKGN`` records
    how many backgrounds were used -- so a test can assert the association
    logic actually wired the dedicated sky observation in.
    """

    name = "calwebb_spec2"
    level = 2
    batch = "per_file"
    inputs = ("*_asn.json", "*_rate.fits")
    writes_official_products = True
    version = "1"

    def run(self, inputs: list[Path], ctx: RunContext, **params: Any) -> list[Path]:
        from astropy.io import fits

        (src,) = inputs
        if src.suffix == ".json":
            (sci,) = _member_paths(src, "science")
            backgrounds = _member_paths(src, "background")
        else:
            sci, backgrounds = src, []
        with fits.open(sci) as hdul:
            hdr = hdul[0].header.copy()
            data = np.asarray(hdul["SCI"].data, dtype="f4")
        subtract = backgrounds and not _skip_requested(params.get("steps"), "bkg_subtract", True)
        if subtract:
            sky = float(np.mean([fits.getheader(b)["JWFMKBG"] for b in backgrounds]))
            hdr["JWFMKBG"] = max(0.0, float(hdr["JWFMKBG"]) - sky)
        hdr["JWFBKGN"] = (len(backgrounds) if subtract else 0, "stub spec2: background members subtracted")
        hdr["DATAMODL"] = "ImageModel"
        out = ctx.output_dir / f"{_strip_suffix(sci.stem)}_cal.fits"
        fits.HDUList([
            fits.PrimaryHDU(header=hdr),
            fits.ImageHDU(data.astype("f4"), name="SCI"),
            fits.ImageHDU(np.full_like(data, 0.01, dtype="f4"), name="ERR"),
            fits.ImageHDU(np.zeros(data.shape, dtype="u4"), name="DQ"),
        ]).writeto(out, overwrite=True)
        ctx.log.info("stub spec2: %s -> %s (%d background member(s))", src.name, out.name, len(backgrounds))
        return [out]


class StubSpec3(Step):
    """Stand-in for calwebb_spec3: scene-rendered ``_s3d`` cubes per band.

    One level-3 association in, DMS-named band cubes out: NIRSpec cubes get
    the ``-<filter>`` the real ``cube_build`` appends, MIRI members expand
    into every ``_ch<N>-<band>`` they cover. When ``steps.extract_1d.skip``
    is false a matching aperture-summed ``_x1d.fits`` is written per cube,
    like the official pipeline would. Cubes are IFUCubeModels with the same
    WCS/photometry metadata the contributed steps read from real cubes.
    """

    name = "calwebb_spec3"
    level = 3
    batch = "per_file"
    inputs = ("*_asn.json",)
    writes_official_products = True
    version = "1"

    #: subtract this much sky before rendering (set by StubSpec3WithBackground)
    def run(self, inputs: list[Path], ctx: RunContext, **params: Any) -> list[Path]:
        from astropy.io import fits

        (asn,) = inputs
        members = _member_paths(asn, "science")
        if not members:
            raise ValueError(f"association {asn.name} has no science members")
        product = _product_name(asn)
        headers = [fits.getheader(m) for m in members]
        first = headers[0]
        scene = MockScene.from_header(first)
        scene, cal_steps = self._apply_background(scene, headers, ctx, **params)
        nwave = int(first["JWFMKNW"])
        size = int(first["JWFMKSZ"])
        pix = float(first["JWFMKPX"])
        outputs: list[Path] = []
        write_x1d = not _skip_requested(params.get("steps"), "extract_1d", True)
        if str(first["INSTRUME"]).upper() == "NIRSPEC":
            configs = sorted({(str(h["GRATING"]), str(h["FILTER"])) for h in headers})
            for grating, filt in configs:
                lo, hi = NRS_GRATINGS[(grating, filt)]
                name = f"{product}-{filt.lower()}" if not product.endswith(filt.lower()) else product
                outputs += self._write_band(ctx, name, scene, first, nwave, size, pix, lo, hi,
                                            instrument="NIRSPEC", grating=grating, filt=filt,
                                            n_members=len(members), x1d=write_x1d, cal_steps=cal_steps)
        else:
            covered: set[tuple[str, str]] = set()
            for h in headers:
                for channel in str(h.get("CHANNEL", "")):
                    covered.add((channel, str(h.get("BAND", "SHORT")).upper()))
            for channel, band in sorted(covered):
                lo, hi = MRS_BANDS[(channel, band)]
                name = f"{product}_ch{channel}-{band.lower()}"
                # the header pix scale is the channel-1 value; later channels grow like
                # the real cube_build grids (0.13 -> 0.35 arcsec), so PSF-scaled
                # apertures and sky annuli stay inside the (small) mock FOV
                pix_band = pix * MRS_PIX_ARCSEC[channel] / MRS_PIX_ARCSEC["1"]
                outputs += self._write_band(ctx, name, scene, first, nwave, size, pix_band, lo, hi,
                                            instrument="MIRI", channel=channel, band=band,
                                            n_members=len(members), x1d=write_x1d, cal_steps=cal_steps)
        ctx.log.info("stub spec3: %s -> %d product(s)", asn.name, len(outputs))
        return outputs

    def _apply_background(self, scene: MockScene, headers: list[Any], ctx: RunContext,
                          **params: Any) -> tuple[MockScene, dict[str, str]]:
        """The scene to render and the ``meta.cal_step`` entries to record on the products
        (empty here: plain spec3, master_background is skipped in the mock workflows)."""
        return scene, {}

    def _write_band(self, ctx: RunContext, name: str, scene: MockScene, first: Any, nwave: int,
                    size: int, pix: float, lo: float, hi: float, *, instrument: str,
                    grating: str | None = None, filt: str | None = None, channel: str | None = None,
                    band: str | None = None, n_members: int, x1d: bool,
                    cal_steps: dict[str, str] | None = None) -> list[Path]:
        from stdatamodels.jwst import datamodels

        wave = np.linspace(lo, hi, nwave)
        data = scene.cube(wave, size, pix)
        area_sr = (pix / 206265.0) ** 2
        cube = datamodels.IFUCubeModel(data=data, err=np.full_like(data, 0.01), dq=np.zeros(data.shape, "u4"))
        cube.meta.instrument.name = instrument
        cube.meta.exposure.type = "NRS_IFU" if instrument == "NIRSPEC" else "MIR_MRS"
        if instrument == "NIRSPEC":
            cube.meta.instrument.grating, cube.meta.instrument.filter = grating, filt
        else:
            cube.meta.instrument.channel, cube.meta.instrument.band = channel, band
        cube.meta.target.proposer_name = str(first["TARGPROP"])
        cube.meta.target.ra, cube.meta.target.dec = float(first["TARG_RA"]), float(first["TARG_DEC"])
        cube.meta.observation.program_number = str(first["PROGRAM"])
        cube.meta.observation.observation_number = str(first["OBSERVTN"])
        w = cube.meta.wcsinfo
        w.ctype1, w.ctype2, w.ctype3 = "RA---TAN", "DEC--TAN", "WAVE"
        w.crval1, w.crval2, w.crval3 = float(first["TARG_RA"]), float(first["TARG_DEC"]), float(wave[0])
        w.crpix1, w.crpix2, w.crpix3 = (size + 1) / 2.0, (size + 1) / 2.0, 1.0
        w.cdelt1, w.cdelt2, w.cdelt3 = -pix / 3600.0, pix / 3600.0, float(wave[1] - wave[0])
        w.cunit1 = w.cunit2 = "deg"
        cube.meta.photometry.pixelarea_steradians = area_sr
        cube.meta.photometry.pixelarea_arcsecsq = pix**2
        for step, status in (cal_steps or {}).items():   # e.g. S_MSBSUB = COMPLETE, like the real pipeline
            setattr(cube.meta.cal_step, step, status)
        cards: list[list[Any]] = [[key, first[key], ""] for key in ("PROGRAM", "OBSERVTN", "TARGPROP", "BKGDTARG")]
        cards.append(["JWFNMEMB", n_members, "stub spec3: science members combined"])
        cards += [[key, value[0], value[1]] for key, value in scene.to_header().items()]
        cube.extra_fits = {"PRIMARY": {"header": cards}}
        cube_path = ctx.output_dir / f"{name}_s3d.fits"
        cube.save(str(cube_path))
        outputs = [cube_path]
        if x1d:
            from ..spectra import Spectrum1D, write_x1d

            flux_jy = data.reshape(nwave, -1).sum(axis=1) * area_sr * 1e6
            header = {"INSTRUME": instrument, "TARGPROP": str(first["TARGPROP"]),
                      "TARG_RA": float(first["TARG_RA"]), "TARG_DEC": float(first["TARG_DEC"]),
                      "SRCNAME": str(first["TARGPROP"])}
            if instrument == "NIRSPEC":
                header["GRATING"], header["FILTER"] = grating, filt
            else:
                header["CHANNEL"], header["BAND"] = channel, band
            outputs.append(write_x1d(ctx.output_dir / f"{name}_x1d.fits",
                                     Spectrum1D(wave, flux_jy, np.full(nwave, 0.01)), header=header))
        return outputs


class StubSpec3WithBackground(StubSpec3):
    """Stand-in for the jwstflow-midas ``spec3_with_background`` subclass.

    Like the real step it locates the per-grating ``*_bkgspec.fits`` in
    ``background_dir`` (failing loudly when it is missing -- the test then
    catches a broken stage wiring) and, like ``master_background``, subtracts
    the *measured* sky spectrum's median from the scene background and
    records the subtraction on the cubes (S_MSBSUB = COMPLETE), which the
    downstream extraction requires before it documents a background.
    """

    def _apply_background(self, scene: MockScene, headers: list[Any], ctx: RunContext,
                          **params: Any) -> tuple[MockScene, dict[str, str]]:
        background_dir = str(params.get("background_dir") or "")
        if not background_dir:
            return scene, {}
        from astropy.table import Table

        grating = str(headers[0].get("GRATING", "")).lower()
        hits = sorted(Path(background_dir).glob(f"*_{grating}*_bkgspec.fits"))
        if not hits:
            raise FileNotFoundError(f"no background file for grating {grating} in {background_dir}")
        table = Table.read(hits[0], hdu="EXTRACT1D")
        measured = float(np.nanmedian(np.asarray(table["SURF_BRIGHT"], dtype=float)))
        ctx.log.info("stub spec3_with_background: subtracting measured sky %.3f MJy/sr from %s", measured, hits[0].name)
        return replace(scene, background=max(0.0, scene.background - measured)), {"master_background": "COMPLETE"}


class StubImage2(StubSpec2):
    """Stand-in for calwebb_image2: one ``_cal.fits`` per level-2 association.

    Same behaviour as :class:`StubSpec2` (imager associations carry no
    background members in the mock workflows), under the image2 identity.
    """

    name = "calwebb_image2"


class StubImage3(StubSpec3):
    """Stand-in for calwebb_image3: one star-field ``_i2d.fits`` per association.

    The stars of the first science member's ``JWFMK*`` star-field spec are
    rendered *displaced by the injected pointing error*, over a TAN WCS
    centred on the (error-free) target position -- so ``gaia_offset``
    measures exactly the injected error against
    :func:`write_gaia_catalog`'s truth, and the astrometry loop closes
    offline.
    """

    name = "calwebb_image3"

    def run(self, inputs: list[Path], ctx: RunContext, **params: Any) -> list[Path]:
        from astropy.io import fits

        (asn,) = inputs
        members = _member_paths(asn, "science")
        if not members:
            raise ValueError(f"association {asn.name} has no science members")
        first = fits.getheader(members[0])
        wcs = _imager_wcs(first)
        size = int(first["JWFMKSZ"])
        ra, dec = star_field(first)
        dra = float(first["JWFMKPRA"]) / 3600.0 / np.cos(np.radians(dec))
        ddec = float(first["JWFMKPDE"]) / 3600.0
        xs, ys = wcs.all_world2pix(ra + dra, dec + ddec, 0)
        rng = np.random.default_rng(int(first["JWFMKSSD"]) + 1)
        image = 0.1 * rng.standard_normal((size, size))
        yy, xx = np.mgrid[:size, :size]
        for x, y in zip(xs, ys):
            image += 50.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5**2))
        out = ctx.output_dir / f"{_product_name(asn)}_i2d.fits"
        primary = fits.Header({"TELESCOP": "JWST", "INSTRUME": "MIRI", "DETECTOR": "MIRIMAGE",
                               "EXP_TYPE": "MIR_IMAGE", "DATE-OBS": str(first["DATE-OBS"]),
                               "PROGRAM": str(first["PROGRAM"]), "OBSERVTN": str(first["OBSERVTN"]),
                               "TARGPROP": str(first["TARGPROP"]), "TARG_RA": float(first["TARG_RA"]),
                               "TARG_DEC": float(first["TARG_DEC"]), "DATAMODL": "ImageModel"})
        fits.HDUList([fits.PrimaryHDU(header=primary),
                      fits.ImageHDU(image.astype("f4"), header=wcs.to_header(), name="SCI")]).writeto(
            out, overwrite=True)
        ctx.log.info("stub image3: %s -> %s (%d star(s), injected offset %.3f/%.3f arcsec)",
                     asn.name, out.name, len(ra), float(first["JWFMKPRA"]), float(first["JWFMKPDE"]))
        return [out]


#: What :func:`stub_pipelines` registers, by the exact ``step:`` spec strings used in YAMLs.
DEFAULT_STUBS: dict[str, type[Step]] = {
    "detector1": StubDetector1,
    "spec2": StubSpec2,
    "spec3": StubSpec3,
    "spec3_with_background": StubSpec3WithBackground,
    "image2": StubImage2,
    "image3": StubImage3,
}


@contextlib.contextmanager
def stub_pipelines(extra: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Shadow the official pipeline aliases with the stubs, restoring on exit.

    The in-process registry wins over built-in aliases and entry points, so a
    workflow YAML runs unchanged -- stage names, directories and product names
    stay exactly as with the real pipelines. Registry entries only exist in
    this process: run with ``parallel.backend=serial`` (or ``workers=1``),
    where tasks execute in-process. For spawned workers, reference the stubs
    by dotted path in the YAML instead (``jwstflow.testing.mock:StubSpec3``).
    """
    mapping: dict[str, Any] = {**DEFAULT_STUBS, **(extra or {})}
    saved = {name: _REGISTRY.get(name) for name in mapping}
    _REGISTRY.update(mapping)
    try:
        yield mapping
    finally:
        for name, previous in saved.items():
            if previous is None:
                _REGISTRY.pop(name, None)
            else:
                _REGISTRY[name] = previous


# --------------------------------------------------------------------------- running workflows


def offline_overrides(work: Path, *, backend: str = "serial", extra: Iterable[str] = ()) -> list[str]:
    """Config overrides that make any workflow run offline inside ``work``.

    The CRDS context is pinned to a fixed name (never contacted -- the stubs
    replace every CRDS-using stage), prefetch and the workflow graph are off,
    the run tree goes to ``<work>/reductions``, and tasks run serially so the
    in-process stub registry applies.
    """
    work = Path(work)
    return [
        f"workspace={work / 'reductions'}",
        f"crds.path={work / 'crds_cache'}",
        "crds.context=jwst_1364.pmap",
        "crds.prefetch=false",
        f"parallel.backend={backend}",
        "workflow_graph=false",
        *extra,
    ]


def run_mock_workflow(
    config_path: str | Path,
    work: Path,
    *,
    overrides: Iterable[str] = (),
    stubs: dict[str, Any] | None = None,
    only: Iterable[str] | None = None,
    start: str | None = None,
    until: str | None = None,
    force: bool = False,
) -> tuple[Any, Any]:
    """Load ``config_path`` with :func:`offline_overrides`, run it with stubbed
    pipelines, and return ``(cfg, RunSummary)``.

    Raw mock data is expected in the config's raw directory (see
    :func:`write_observation`); MAST download is skipped. ``overrides`` are
    appended after the offline set, so tests can tweak any stage parameter
    (``--set`` syntax, e.g. ``"stages.extract_extended.parameters.apcorr=false"``).
    """
    from ..config.loader import load_config
    from ..engine.runner import Runner

    cfg = load_config(config_path, overrides=offline_overrides(Path(work), extra=overrides))
    with stub_pipelines(stubs):
        summary = Runner(cfg, skip_download=True, only=only, start=start, until=until, force=force).run()
    return cfg, summary
