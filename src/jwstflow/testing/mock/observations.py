"""Mock observations: the ``_uncal`` files of a program, with DMS names and headers, and the MAST log.

Also the simultaneous imager's star field and the Gaia-like truth catalogue that goes with it.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .instruments import MRS_DETECTOR_CHANNELS, NRS_GRATINGS
from .scene import MockScene


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


#: The default NIRSpec mock scene: a bright edge-on disk over a faint sky with one gas-line
#: halo per grating and a hot pixel. The hot pixel stays a clear outlier but below the disk's
#: *smoothed* peak, so it cannot dominate the disk_mask normalisation (in real data the
#: many-plane median and dither averaging dilute single hot detector pixels the same way).
#: ``dataclasses.replace(DEFAULT_NIRSPEC_SCENE, cluster_dither=2, ...)`` adds a bad cluster.
DEFAULT_NIRSPEC_SCENE = MockScene(background=1.0, disk_peak=1000.0, point_flux_jy=0.0, hot_value=3000.0,
                                  lines=((2.1218, 400.0), (3.2970, 300.0)), line_width_um=0.03)


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
        scene = DEFAULT_NIRSPEC_SCENE
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
