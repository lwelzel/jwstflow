"""Reading products for QA figures: locating files of other stages, the spectrum
tables of 1-D products, the planes of a cube."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .frames import SkyFrame
from .units import to_mjy

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- products
def find_file(name: str, ctx: Any, stage: str | None = None, *extra: Path) -> Path | None:
    """Locate the product ``name`` for a QA step: in ``stage``'s directory, in any stage
    directory of the run, in the ``extra`` directories, or -- last -- in the sibling runs
    of the target (``<target>/<run>/<level>/<stage>/``). None when it is nowhere."""
    dirs: list[Path] = []
    if stage:
        try:
            dirs.append(ctx.dir_of(stage))
        except KeyError:
            log.warning("stage %r is not part of this workflow; searching all stages", stage)
    dirs += [d for _, d in sorted(ctx.stage_dirs.items())] + list(extra)
    for d in dirs:
        candidate = Path(d) / name
        if candidate.exists():
            return candidate
    target_dir = getattr(ctx, "target_dir", None)
    if target_dir is not None and Path(target_dir).is_dir():
        hits = sorted(Path(target_dir).glob(f"*/*/*/{name}"))
        if hits:
            return hits[0]
    return None


def stage_of(path: Path) -> str:
    """The stage a product belongs to: the name of the directory it lies in."""
    return Path(path).parent.name


def ancestor(path: Path, directory: Path) -> Path | None:
    """The file in ``directory`` that ``path`` was derived from: same product suffix, and its
    name without that suffix the longest leading part of ``path``'s (None when there is none)."""
    stem, _, suffix = path.stem.rpartition("_")
    best: Path | None = None
    for candidate in sorted(directory.glob(f"*_{suffix}{path.suffix}")):
        core = candidate.stem.rpartition("_")[0]
        if (stem == core or stem.startswith(core + "_")) and (best is None or len(core) > len(best.stem.rpartition("_")[0])):
            best = candidate
    return best


def short_labels(names: list[str]) -> list[str]:
    """Distinct cores of a family of product names: the leading and trailing name parts
    (``_``-separated) that all of them share are stripped, whole parts only -- so
    ``..._nirspec_g235h-f170lp_s1d`` and ``..._nirspec_g395h-f290lp_s1d`` become
    ``g235h-f170lp`` and ``g395h-f290lp``, never a fragment of either."""
    if len(names) < 2:
        return list(names)
    parts = [n.split("_") for n in names]
    shortest = min(len(p) for p in parts)
    head = 0
    while head < shortest - 1 and len({p[head] for p in parts}) == 1:
        head += 1
    tail = 0
    while tail < shortest - head - 1 and len({p[-1 - tail] for p in parts}) == 1:
        tail += 1
    return ["_".join(p[head:len(p) - tail]) or n for p, n in zip(parts, names)]


SPECTRUM_EXTENSIONS = ("EXTRACT1D", "COMBINE1D")


def column_unit(hdu: Any, column: str) -> str | None:
    """A table column's TUNIT, falling back to the x1d contract's unit."""
    from ..spectra import X1D_UNITS

    return hdu.columns[column].unit or X1D_UNITS.get(column.upper())


def spectrum_name(hdu: Any, index: int) -> str:
    """What to call one of several spectra in a file: its slit, its source, else its position."""
    return str(hdu.header.get("SLTNAME") or hdu.header.get("SRCNAME") or f"ext {index + 1}")


def read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    """(wavelength, flux, flux error, flux unit) of the first EXTRACT1D/COMBINE1D table of a
    1-D product (the error all-NaN when the table carries no FLUX_ERROR column)."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        for name in SPECTRUM_EXTENSIONS:
            if name in hdul:
                tab = hdul[name].data
                wave = np.asarray(tab["WAVELENGTH"], dtype=float)
                error = np.asarray(tab["FLUX_ERROR"], dtype=float) if "FLUX_ERROR" in tab.names \
                    else np.full(len(wave), np.nan)
                return wave, np.asarray(tab["FLUX"], dtype=float), error, column_unit(hdul[name], "FLUX")
    raise ValueError(f"{path}: no EXTRACT1D/COMBINE1D extension")


def read_background(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Finite ``(wavelength, background, bkgd_error)`` [um, mJy, mJy] of the background a 1-D
    product records as subtracted, or None when it carries none (no SKYSUB header, no
    BACKGROUND column, or no finite samples)."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        if not hdul[0].header.get("SKYSUB"):
            return None
        exts = [h for h in hdul if h.name in SPECTRUM_EXTENSIONS]
        if not exts or "BACKGROUND" not in exts[0].data.names:
            return None
        tab = exts[0].data
        wave = np.asarray(tab["WAVELENGTH"], dtype=float)
        bkg, _ = to_mjy(np.asarray(tab["BACKGROUND"], dtype=float), column_unit(exts[0], "BACKGROUND"))
        err_raw = np.asarray(tab["BKGD_ERROR"], dtype=float) if "BKGD_ERROR" in tab.names else np.zeros(len(wave))
        err, _ = to_mjy(err_raw, column_unit(exts[0], "BKGD_ERROR"))
    ok = np.isfinite(wave) & np.isfinite(bkg)
    if not ok.any():
        return None
    order = np.argsort(wave[ok])
    return wave[ok][order], bkg[ok][order], np.where(np.isfinite(err[ok]), err[ok], 0.0)[order]


def read_cube(path: Path) -> dict[str, Any] | None:
    """SCI / ERR planes, wavelengths [um], unit, pixel area and sky frame of an IFU cube
    (None, with a warning, when ``path`` holds no 3-D SCI extension)."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        if "SCI" not in hdul or hdul["SCI"].data is None or hdul["SCI"].data.ndim != 3:
            log.warning("%s has no 3-D SCI extension; skipped", path.name)
            return None
        header = hdul["SCI"].header
        data = np.asarray(hdul["SCI"].data, dtype=float)
        error = np.asarray(hdul["ERR"].data, dtype=float) if "ERR" in hdul and hdul["ERR"].data is not None \
            and hdul["ERR"].data.shape == data.shape else None
        wave = None
        if "WCS-TABLE" in hdul and "wavelength" in [n.lower() for n in hdul["WCS-TABLE"].data.names]:
            wave = np.asarray(hdul["WCS-TABLE"].data["wavelength"], dtype=float).ravel()
        if wave is None or wave.size != data.shape[0]:
            n = data.shape[0]
            wave = float(header.get("CRVAL3", 0.0)) + (np.arange(n) + 1 - float(header.get("CRPIX3", 1.0))) \
                * float(header.get("CDELT3", 1.0))
        pixar_sr = header.get("PIXAR_SR")
        if pixar_sr is None and header.get("PIXAR_A2") is not None:
            pixar_sr = float(header["PIXAR_A2"]) / (np.degrees(1.0) * 3600.0) ** 2
        return {"data": data, "error": error, "wave": wave, "unit": header.get("BUNIT"),
                "pixar_sr": None if pixar_sr is None else float(pixar_sr),
                "sky": SkyFrame.from_header(header, data.shape, primary=hdul[0].header)}


__all__ = [
    "SPECTRUM_EXTENSIONS",
    "ancestor",
    "column_unit",
    "find_file",
    "read_background",
    "read_cube",
    "read_spectrum",
    "short_labels",
    "spectrum_name",
    "stage_of",
]
