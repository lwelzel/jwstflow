"""Instrument tables of the mock observations: real wavelength ranges, detectors and spaxel scales."""

from __future__ import annotations

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
