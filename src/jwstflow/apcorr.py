"""Aperture-correction references for IFU extraction steps (MIRI MRS, NIRSpec IFU).

Extraction steps that integrate a source over a finite aperture use the CRDS
``apcorr`` reference to account for the PSF flux falling outside the aperture.
The reference layouts differ per instrument; this module normalises both to
one plain-array table

    {"wavelength": (nw,), "radius": (nr, nw), "apcorr": (nr, nw), "radius_units": str}

(the layout jwst's own ``ApCorrRadial`` reads) and evaluates it at a given
aperture radius per wavelength. :func:`load_apcorr` resolves the reference
through CRDS for an opened cube datamodel unless an explicit file is given.

Heavy imports (jwst, stdatamodels) happen lazily and only for the CRDS lookup
and file reading, so the module stays importable without jwst. The code was
adapted from the JOYS+ MRS extraction and previously lived, duplicated, in
jwstflow-joys and jwstflow-midas.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def load_apcorr(cube: Any, apcorr_file: str | None = None) -> dict[str, np.ndarray]:
    """Aperture-correction table for this cube, via CRDS unless a file is given.

    ``cube`` is an opened IFU cube datamodel; the instrument decides the layout:
    MIRI (``MirMrsApcorrModel``) stores one table per band as an object node,
    NIRSpec (``NrsIfuApcorrModel``) one row per FILTER/GRATING combination.
    """
    path = apcorr_file
    if path is None:
        from jwst.stpipe import Step as _StpipeStep

        path = _StpipeStep().get_reference_file(cube, "apcorr")  # honours CRDS_CONTEXT / cache
        log.info("apcorr reference from CRDS: %s", path)
    instrument = str(cube.meta.instrument.name).upper()
    if instrument == "MIRI":
        from stdatamodels.jwst.datamodels import MirMrsApcorrModel

        with MirMrsApcorrModel(path) as ref:
            _warn_on_band_mismatch(ref.apcorr_table, cube, path)
            return mrs_apcorr_table(ref.apcorr_table)
    if instrument == "NIRSPEC":
        from stdatamodels.jwst.datamodels import NrsIfuApcorrModel

        filt = str(cube.meta.instrument.filter).upper()
        grat = str(cube.meta.instrument.grating).upper()
        with NrsIfuApcorrModel(path) as ref:
            return nrs_apcorr_table(ref.apcorr_table, filt, grat,
                                    sizeunit=getattr(ref, "sizeunit", None), origin=str(path))
    raise ValueError(f"no apcorr layout known for instrument {instrument!r} (supported: MIRI MRS, NIRSpec IFU)")


def mrs_apcorr_table(node: Any) -> dict[str, np.ndarray]:
    """Normalise a MIRI MRS ``apcorr_table`` node: ``wavelength`` (nw,), ``radius`` and
    ``apcorr`` (nradius x nwave; a transposed layout is tolerated) and ``radius_units``."""
    wavelength = np.asarray(node.wavelength, dtype=float).ravel()
    radius = np.atleast_2d(np.asarray(node.radius, dtype=float))
    apcorr = np.atleast_2d(np.asarray(node.apcorr, dtype=float))
    units = str(getattr(node, "radius_units", "arcsec") or "arcsec")
    if radius.shape[-1] != wavelength.size:  # tolerate a transposed (nwave x nradius) layout
        radius, apcorr = radius.T, apcorr.T
    return {"wavelength": wavelength, "radius": radius, "apcorr": apcorr, "radius_units": units}


def nrs_apcorr_table(rows: Any, filt: str, grat: str, *, sizeunit: str | None = None,
                     origin: str = "apcorr reference") -> dict[str, np.ndarray]:
    """The NIRSpec-IFU apcorr row matching FILTER/GRATING as a plain-array table.

    ``rows`` iterates mappings with ``filter``, ``grating``, ``nelem_wl``,
    ``wavelength`` (nelem_wl), ``radius`` and ``apcorr`` (nradius x nelem_wl) --
    the ``NrsIfuApcorrModel.apcorr_table`` layout.
    """
    seen = []
    for row in rows:
        row_filt, row_grat = str(row["filter"]).upper(), str(row["grating"]).upper()
        seen.append(f"{row_grat}/{row_filt}")
        if (row_filt, row_grat) != (filt.upper(), grat.upper()):
            continue
        n = int(row["nelem_wl"])
        wavelength = np.asarray(row["wavelength"], dtype=float).ravel()[:n]
        radius = np.atleast_2d(np.asarray(row["radius"], dtype=float))[:, :n]
        apcorr = np.atleast_2d(np.asarray(row["apcorr"], dtype=float))[:, :n]
        units = str(sizeunit or "arcsec")
        return {"wavelength": wavelength, "radius": radius, "apcorr": apcorr, "radius_units": units}
    raise ValueError(f"{origin} has no row for {grat}/{filt} (rows: {sorted(set(seen))})")


def apcorr_factor(table: dict[str, np.ndarray], wave: np.ndarray, radius_arcsec: np.ndarray,
                  pix_arcsec: float | None = None) -> np.ndarray:
    """Correction at each plane's aperture radius, like jwst's ApCorrRadial: every radius track
    (nradius x nwave) and its correction are interpolated in wavelength, then the correction is
    interpolated in radius at the plane's aperture. Never below 1; clamped at the grid edges."""
    ref_w = np.asarray(table["wavelength"], dtype=float)
    ref_r = np.atleast_2d(np.asarray(table["radius"], dtype=float))
    ref_c = np.atleast_2d(np.asarray(table["apcorr"], dtype=float))
    units = str(table.get("radius_units", "arcsec"))
    radius = np.asarray(radius_arcsec, dtype=float)
    if units.startswith("pix"):
        if not pix_arcsec:
            raise ValueError("apcorr reference radii are in pixels; the cube pixel scale is needed")
        radius = radius / pix_arcsec
    order = np.argsort(ref_w)
    ref_w, ref_r, ref_c = ref_w[order], ref_r[:, order], ref_c[:, order]
    out = np.ones(len(wave))
    for k, (w, r) in enumerate(zip(wave, radius)):
        radii = np.array([np.interp(w, ref_w, ref_r[i]) for i in range(ref_r.shape[0])])
        corrs = np.array([np.interp(w, ref_w, ref_c[i]) for i in range(ref_c.shape[0])])
        srt = np.argsort(radii)
        out[k] = max(1.0, float(np.interp(r, radii[srt], corrs[srt])))
    return out


def _warn_on_band_mismatch(node: Any, cube: Any, path: Any) -> None:
    """An explicit MIRI apcorr file for the wrong band silently mis-corrects; say so."""
    for attr, cube_val in (("channel", cube.meta.instrument.channel), ("band", cube.meta.instrument.band)):
        ref_val = getattr(node, attr, None)
        if ref_val not in (None, "") and cube_val not in (None, "") and str(ref_val).upper() != str(cube_val).upper():
            log.warning("apcorr reference %s is for %s=%s but the cube has %s=%s",
                        path, attr, ref_val, attr, cube_val)
