"""The joint-cube product (``*_s3dcomb.fits``): writing it as an IFU cube model, reading its records.

The layout is described in :mod:`jwstflow.cubes.stitching`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .grid import SpatialGrid

#: Product suffix of joint cubes (``*_s3dcomb.fits``), the cube counterpart of ``_s1dcomb``.
JOINT_SUFFIX = "s3dcomb"

#: Columns of the SEGMENTS table: one row per segment, blue to red.
SEGMENT_COLUMNS: tuple[str, ...] = (
    "label", "file", "wave_min", "wave_max", "wave_first", "wave_last", "wave_step", "plane_first", "plane_last",
    "n_planes", "scale", "scale_error", "spaxel_arcsec", "placement", "phase_x", "phase_y", "correlation")
#: Columns of the OVERLAPS table: one row per pair of neighbouring segments.
OVERLAP_COLUMNS: tuple[str, ...] = (
    "segment_a", "segment_b", "crossover", "overlap_min", "overlap_max", "window_min", "window_max", "ratio",
    "ratio_error", "n_samples", "n_spaxels", "region", "level_a", "level_b", "dra", "ddec", "dra_error", "ddec_error",
    "x", "y", "n_window")
#: Columns of the two spectrum tables (SEGSPEC: per segment over the common field; OVLSPEC: per pair
#: and segment over the pair's measurement region).
SPECTRUM_COLUMNS: tuple[str, ...] = ("index", "segment", "wavelength", "flux", "error")


def write_joint_cube(path: Path, *, like: Path, grid: SpatialGrid, wave: np.ndarray, data: np.ndarray,
                     err: np.ndarray, dq: np.ndarray, wmap: np.ndarray, segments: list[dict[str, Any]],
                     overlaps: list[dict[str, Any]], segment_spectra: list[dict[str, Any]],
                     overlap_spectra: list[dict[str, Any]], overlap_images: np.ndarray, overlap_regions: np.ndarray,
                     overlap_cores: np.ndarray, common: np.ndarray, source: np.ndarray, instrument: dict[str, str], keys: dict[str, tuple[Any, str]]) -> Path:
    """Write a joint cube (layout in :mod:`jwstflow.cubes.stitching`).

    The cube itself is an ``IFUCubeModel`` with a tabular wavelength axis --
    what ``cube_build`` writes for a multi-band cube -- carrying the metadata
    of ``like`` (the reference segment) with the grid, the wavelength range
    and the band keywords (``instrument``: datamodel attribute -> value) of
    the joint cube. ``keys`` are PRIMARY keywords (value, comment). The
    stitch records follow as further extensions.
    """
    from astropy.io import fits
    from astropy.table import Table
    from jwst.assign_wcs import pointing
    from stdatamodels.jwst import datamodels

    nw, ny, nx = data.shape
    wavelengths = np.asarray(wave, dtype=np.float32)
    table = np.array([(wavelengths[None].T,)], dtype=[("wavelength", "<f4", (nw, 1))])
    cards = grid.header_cards()
    with datamodels.open(str(like)) as reference:
        model = datamodels.IFUCubeModel(data=data, err=err, dq=dq, weightmap=wmap, wavetable=table)
        model.update(reference)                                    # the observation; never its WCS
        scale_ratio = grid.pixel_area_arcsec2 / float(reference.meta.photometry.pixelarea_arcsecsq or grid.pixel_area_arcsec2)
        area_sr = float(reference.meta.photometry.pixelarea_steradians or grid.pixel_area_sr) * scale_ratio
    model.meta.filename = Path(path).name
    for attribute, value in instrument.items():
        setattr(model.meta.instrument, attribute, value)
    w = model.meta.wcsinfo
    w.ctype1, w.ctype2, w.cunit1, w.cunit2 = cards["CTYPE1"], cards["CTYPE2"], cards["CUNIT1"], cards["CUNIT2"]
    w.crval1, w.crval2, w.crpix1, w.crpix2 = cards["CRVAL1"], cards["CRVAL2"], cards["CRPIX1"], cards["CRPIX2"]
    w.cdelt1, w.cdelt2 = cards["CDELT1"], cards["CDELT2"]
    w.pc1_1, w.pc1_2, w.pc2_1, w.pc2_2 = cards["PC1_1"], cards["PC1_2"], cards["PC2_1"], cards["PC2_2"]
    w.pc1_3 = w.pc2_3 = w.pc3_1 = w.pc3_2 = 0
    w.pc3_3 = 1
    # the wavelength axis: a table, in the keywords cube_build uses for non-linear cubes
    w.ctype3, w.ps3_0, w.ps3_1, w.cunit3 = "WAVE-TAB", "WCS-TABLE", "wavelength", "um"
    w.crval3, w.crpix3, w.cdelt3, w.wcsaxes = 1.0, 1.0, None, 3
    model.wavedim = f"(1,{nw:d})"
    w.waverange_start, w.waverange_end = float(wavelengths[0]) * 1e-6, float(wavelengths[-1]) * 1e-6
    corners_ra, corners_dec = grid.pixel_to_sky([-0.5, -0.5, nx - 0.5, nx - 0.5], [-0.5, ny - 0.5, ny - 0.5, -0.5])
    w.s_region = "POLYGON ICRS  " + " ".join(f"{ra:.9f} {dec:.9f}" for ra, dec in zip(corners_ra, corners_dec))
    model.meta.photometry.pixelarea_arcsecsq = grid.pixel_area_arcsec2
    model.meta.photometry.pixelarea_steradians = area_sr
    model.meta.wcs = pointing.create_fitswcs(model)
    model.meta.wcs.bounding_box = ((0, nx - 1), (0, ny - 1), (0, nw - 1))
    model.extra_fits = {"PRIMARY": {"header": [[key, value, comment] for key, (value, comment) in keys.items()]}}
    model.save(str(path))
    model.close()

    image_header = fits.Header()
    for key, value in cards.items():
        image_header[key] = value
    image_header["WCSAXES"] = 2

    def spectra(rows: list[dict[str, Any]]) -> Any:
        columns = {name: np.concatenate([np.atleast_1d(r[name]) for r in rows]) if rows else np.array([])
                   for name in SPECTRUM_COLUMNS}
        out = Table(columns)
        out["wavelength"].unit, out["flux"].unit, out["error"].unit = "um", "Jy", "Jy"
        return out

    segment_table = Table({name: [row[name] for row in segments] for name in SEGMENT_COLUMNS})
    overlap_table = Table({name: [row[name] for row in overlaps] for name in OVERLAP_COLUMNS})
    with fits.open(path, mode="append") as hdul:
        hdul.append(fits.BinTableHDU(segment_table, name="SEGMENTS"))
        hdul.append(fits.BinTableHDU(overlap_table, name="OVERLAPS"))
        hdul.append(fits.BinTableHDU(spectra(segment_spectra), name="SEGSPEC"))
        hdul.append(fits.BinTableHDU(spectra(overlap_spectra), name="OVLSPEC"))
        hdul.append(fits.ImageHDU(np.asarray(overlap_images, dtype="f4"), header=image_header.copy(), name="OVLIMG"))
        hdul.append(fits.ImageHDU(np.asarray(overlap_regions, dtype="u1"), header=image_header.copy(), name="OVLREG"))
        hdul.append(fits.ImageHDU(np.asarray(overlap_cores, dtype="u1"), header=image_header.copy(), name="OVLCORE"))
        hdul.append(fits.ImageHDU(np.asarray(common, dtype="u1"), header=image_header.copy(), name="COMMON"))
        hdul.append(fits.ImageHDU(np.asarray(source, dtype="u1"), header=image_header.copy(), name="SOURCE"))
    return Path(path)


def read_joint_records(path: Path | str) -> dict[str, Any]:
    """The stitch records of a joint cube as plain tables and arrays -- everything but the cube.

    Keys: ``path``, ``header`` (PRIMARY), ``map_header`` (the celestial WCS
    of the grid, for the 2-D extensions), ``unit``, ``pixel_area_sr``,
    ``wave`` (per plane), ``segments``, ``overlaps``, ``segment_spectra``,
    ``overlap_spectra`` (tables), ``overlap_images`` and ``overlap_cores``
    ``(2 * pairs, ny, nx)`` (segment a and b of every pair: the collapsed
    image, and its bright core inside the window the pair was registered in),
    ``overlap_regions`` ``(pairs, ny, nx)``, and ``common`` and ``source``
    ``(ny, nx)`` (bool: the field all segments cover, and the region in it
    that ``segment_spectra`` are taken over).
    """
    from astropy.io import fits
    from astropy.table import Table

    from ..wcs import hdu_wavelengths

    with fits.open(path) as hdul:
        sci = hdul["SCI"].header
        return {
            "path": Path(path), "header": hdul[0].header.copy(), "map_header": hdul["COMMON"].header.copy(),
            "unit": sci.get("BUNIT"), "pixel_area_sr": sci.get("PIXAR_SR"), "shape": hdul["SCI"].shape,
            "wave": hdu_wavelengths(hdul),
            "segments": Table(hdul["SEGMENTS"].data), "overlaps": Table(hdul["OVERLAPS"].data),
            "segment_spectra": Table(hdul["SEGSPEC"].data), "overlap_spectra": Table(hdul["OVLSPEC"].data),
            "overlap_images": np.asarray(hdul["OVLIMG"].data, dtype=float),
            "overlap_regions": np.asarray(hdul["OVLREG"].data).astype(bool),
            "overlap_cores": np.asarray(hdul["OVLCORE"].data).astype(bool),
            "common": np.asarray(hdul["COMMON"].data).astype(bool),
            "source": np.asarray(hdul["SOURCE"].data).astype(bool),
        }


def region_flux(path: Path | str, region: np.ndarray, field: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(wavelength, flux [Jy])`` of a joint cube over ``region``, read plane by plane from its SCI
    extension -- the cube itself, for comparison with what its records say went into it.

    The sum over the region, NaN in planes where a spaxel of it is missing;
    with ``field`` the excess over the level of that field (its median in
    each plane), as :func:`jwstflow.cubes.stitching.measure.region_spectrum`
    measures it.
    """
    from astropy.io import fits

    from ..wcs import hdu_wavelengths
    from .measure import region_spectrum

    with fits.open(path, memmap=True) as hdul:
        wave = hdu_wavelengths(hdul)
        area = float(hdul["SCI"].header.get("PIXAR_SR") or np.nan)
        flux = np.full(len(wave), np.nan)
        sci = hdul["SCI"].data
        for start in range(0, len(wave), 512):
            planes = np.asarray(sci[start:start + 512], dtype=np.float32)
            flux[start:start + 512] = region_spectrum(planes, None, region, pixel_area_sr=area, field=field)[0]
    return wave, flux
