"""The astrophysical scene of a mock observation; it travels in ``JWFMK*`` header keywords."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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
    #: a bad spaxel cluster in ONE dither (0: none): an elongated blob (3:1 ellipse, 30 deg)
    #: of semi-major ``cluster_radius`` px around ``cluster_center`` (y, x), adding
    #: ``cluster_value`` on the planes inside ``cluster_wave`` [um] -- what
    #: flag_spaxel_clusters must find and propagate_cluster_flags must remove
    cluster_dither: int = 0
    cluster_center: tuple[int, int] = (0, 0)
    cluster_radius: float = 0.0
    cluster_wave: tuple[float, float] = (0.0, 0.0)
    cluster_value: float = 0.0

    # -- rendering -----------------------------------------------------------
    def cluster_mask(self, size: int) -> np.ndarray:
        """The (size, size) footprint of the bad cluster: deliberately not circular."""
        yy, xx = np.mgrid[:size, :size]
        cy, cx = self.cluster_center
        pa = np.radians(30.0)
        u = (xx - cx) * np.cos(pa) + (yy - cy) * np.sin(pa)
        v = -(xx - cx) * np.sin(pa) + (yy - cy) * np.cos(pa)
        a = max(float(self.cluster_radius), 1e-6)
        return (u / a) ** 2 + (v / (a / 3.0)) ** 2 <= 1.0

    def plane(self, size: int, pix_arcsec: float, wave_um: float, dither: int | None = None) -> np.ndarray:
        """One (size, size) surface-brightness image [MJy/sr] at ``wave_um``; ``dither``
        selects dither-specific content (the bad cluster of ``cluster_dither``)."""
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
        if (self.cluster_dither and dither == self.cluster_dither
                and self.cluster_wave[0] <= wave_um <= self.cluster_wave[1]):
            img[self.cluster_mask(size)] += self.cluster_value
        return img

    def cube(self, wave_um: np.ndarray, size: int, pix_arcsec: float, dither: int | None = None) -> np.ndarray:
        """The full (nwave, size, size) scene cube [MJy/sr], with optional noise."""
        data = np.stack([self.plane(size, pix_arcsec, float(w), dither) for w in np.asarray(wave_um)])
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
            "JWFMKCD": (self.cluster_dither, "mock scene: bad cluster dither (0: none)"),
            "JWFMKCY": (self.cluster_center[0], "mock scene: bad cluster centre y [px]"),
            "JWFMKCX": (self.cluster_center[1], "mock scene: bad cluster centre x [px]"),
            "JWFMKCR": (self.cluster_radius, "mock scene: bad cluster semi-major [px]"),
            "JWFMKC1": (self.cluster_wave[0], "mock scene: bad cluster wave min [um]"),
            "JWFMKC2": (self.cluster_wave[1], "mock scene: bad cluster wave max [um]"),
            "JWFMKCV": (self.cluster_value, "mock scene: bad cluster value [MJy/sr]"),
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
            cluster_dither=int(hdr.get("JWFMKCD", 0)),
            cluster_center=(int(hdr.get("JWFMKCY", 0)), int(hdr.get("JWFMKCX", 0))),
            cluster_radius=float(hdr.get("JWFMKCR", 0.0)),
            cluster_wave=(float(hdr.get("JWFMKC1", 0.0)), float(hdr.get("JWFMKC2", 0.0))),
            cluster_value=float(hdr.get("JWFMKCV", 0.0)),
        )
