"""Stub pipelines: scene-faithful stand-ins that keep the official pipelines' interfaces.

Same stage names, same product names, the same association files in, the parameters that
shape the data flow honoured -- and :func:`stub_pipelines` to put them in place of the
official aliases for the duration of a test.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ...steps.base import _REGISTRY, RunContext, Step
from .detector import _mock_flagged_voxels, mock_cal_layout, mock_cal_wcs
from .instruments import MRS_BANDS, MRS_PIX_ARCSEC, NRS_GRATINGS
from .observations import _imager_wcs, star_field
from .scene import MockScene


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
    logic actually wired the dedicated sky observation in. NIRSpec IFU cal
    files get the mock detector layout (:func:`mock_cal_layout`: the scene
    rendered as one pixel per spaxel and plane) and its ``gwcs``
    (:func:`mock_cal_wcs`), like ``assign_wcs`` would attach one.
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
        layout = mock_cal_layout(hdr)
        if layout is not None:
            # the IFU detector: rows = spaxels, columns = this detector's planes, scene-rendered
            lo, hi = layout["planes"]
            scene = MockScene.from_header(hdr)
            planes = scene.cube(layout["wave"][lo:hi], layout["size"], layout["pix"], dither=int(hdr.get("PATT_NUM", 1)))
            data = np.ascontiguousarray(planes.reshape(hi - lo, -1).T)
        fits.HDUList([
            fits.PrimaryHDU(header=hdr),
            fits.ImageHDU(data.astype("f4"), name="SCI"),
            fits.ImageHDU(np.full_like(data, 0.01, dtype="f4"), name="ERR"),
            fits.ImageHDU(np.zeros(data.shape, dtype="u4"), name="DQ"),
        ]).writeto(out, overwrite=True)
        if layout is not None:
            from stdatamodels.jwst import datamodels

            tmp = out.with_name(out.stem + ".wcs.tmp.fits")
            with datamodels.open(out) as model:
                model.meta.wcs = mock_cal_wcs(hdr)
                model.save(str(tmp))
            tmp.replace(out)
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

    Like ``cube_build``, the scene is rendered *per dither* (the members'
    PATT_NUM) and averaged over the dithers; voxels whose detector pixel is
    OUTLIER-flagged in a member cal file (:func:`_mock_flagged_voxels`) are
    left out of that dither, so flags set by ``propagate_cluster_flags``
    remove a dither's bad cluster from the combined cube exactly as the real
    drizzle would, and a single-dither cube shows NaN there.
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
        members, product = self._members_and_product(asn)
        if not members:
            raise ValueError(f"association {asn.name} has no science members")
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
                band_members = [m for m, h in zip(members, headers)
                                if (str(h["GRATING"]), str(h["FILTER"])) == (grating, filt)]
                outputs += self._write_band(ctx, name, scene, first, nwave, size, pix, lo, hi,
                                            instrument="NIRSPEC", grating=grating, filt=filt,
                                            n_members=len(members), x1d=write_x1d, cal_steps=cal_steps,
                                            members=band_members)
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
                                            n_members=len(members), x1d=write_x1d, cal_steps=cal_steps,
                                            members=list(members))
        ctx.log.info("stub spec3: %s -> %d product(s)", asn.name, len(outputs))
        return outputs

    def _members_and_product(self, src: Path) -> tuple[list[Path], str]:
        """Science members and product base name of an association file."""
        return _member_paths(src, "science"), _product_name(src)

    @staticmethod
    def _render(scene: MockScene, wave: np.ndarray, size: int, pix: float, members: list[Path],
                ) -> tuple[np.ndarray, tuple[int, int]]:
        """The band cube: the scene per dither, minus OUTLIER-flagged voxels, averaged over
        the dithers; plus the plane range the members' detectors cover."""
        import warnings

        from astropy.io import fits

        by_dither: dict[int, list[Path]] = {}
        covered: list[tuple[int, int]] = []
        for m in members:
            hdr = fits.getheader(m)
            by_dither.setdefault(int(hdr.get("PATT_NUM", 1)), []).append(m)
            layout = mock_cal_layout(hdr)
            covered.append(layout["planes"] if layout is not None else (0, len(wave)))
        stack = []
        for dither, files in sorted(by_dither.items()):
            cube = scene.cube(wave, size, pix, dither=dither).astype(float)
            for m in files:
                voxels = _mock_flagged_voxels(m)
                if voxels is not None and voxels[0].size:
                    k, iy, ix = voxels
                    ok = k < cube.shape[0]
                    cube[k[ok], iy[ok], ix[ok]] = np.nan
            stack.append(cube)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
            data = np.nanmean(np.stack(stack), axis=0)
        lo, hi = min(c[0] for c in covered), max(c[1] for c in covered)
        return data.astype("f4"), (lo, hi)

    def _apply_background(self, scene: MockScene, headers: list[Any], ctx: RunContext,
                          **params: Any) -> tuple[MockScene, dict[str, str]]:
        """The scene to render and the ``meta.cal_step`` entries to record on the products
        (empty here: plain spec3, master_background is skipped in the mock workflows)."""
        return scene, {}

    def _write_band(self, ctx: RunContext, name: str, scene: MockScene, first: Any, nwave: int,
                    size: int, pix: float, lo: float, hi: float, *, instrument: str,
                    grating: str | None = None, filt: str | None = None, channel: str | None = None,
                    band: str | None = None, n_members: int, x1d: bool,
                    cal_steps: dict[str, str] | None = None, members: list[Path] | None = None) -> list[Path]:
        from stdatamodels.jwst import datamodels

        wave = np.linspace(lo, hi, nwave)
        if members:
            data, (k_lo, k_hi) = self._render(scene, wave, size, pix, members)
            wave, data = wave[k_lo:k_hi], data[k_lo:k_hi]   # only the planes the members' detectors cover
            nwave = len(wave)
        else:
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
            from ...spectra import Spectrum1D, write_x1d

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


class StubCubeBuild(StubSpec3):
    """Stand-in for the ``cube_build`` step on its own: one cube per association *or* cal file.

    Same rendering as :class:`StubSpec3` (per dither, OUTLIER voxels left
    out), under the ``cube_build`` identity. Per-dither associations
    (``group_by: [..., PATT_NUM]``) give the per-dither cubes the bad-cluster
    chain inspects; a single cal file gives ``<base>_<grating>-<filter>_s3d.fits``
    over that detector's half of the band, like the real step. Accepts and
    ignores ``coord_system``/``weighting``.
    """

    name = "cube_build"
    inputs = ("*_asn.json", "*_cal.fits")

    def _members_and_product(self, src: Path) -> tuple[list[Path], str]:
        if src.suffix == ".json":
            return super()._members_and_product(src)
        from astropy.io import fits

        hdr = fits.getheader(src)
        return [src], f"{_strip_suffix(src.stem)}_{str(hdr['GRATING']).lower()}"


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
    "cube_build": StubCubeBuild,
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
