"""File naming rules shared by all steps.

Official pipeline products keep the names the pipeline gives them; jwstflow
never renames them. Custom steps that *derive* products follow the same
grammar (``<basename>_<suffix>.fits``) but must not use a suffix that the
pipeline reserves for its own product types (``_cal``, ``_s3d``, ``_x1d``, ...),
so that a file's suffix always says which software produced it.

Level-3 product names follow the DMS convention
``jw<PPPPP>-o<OOO>_<tTTT|sSSSSS>_<instrument>[_<optelem>]``; the target id
comes from the MAST observation id recorded at download time (it is not in
the FITS headers).
"""

from __future__ import annotations

import re
from pathlib import Path

# Product-type suffixes of the jwst pipeline (subset of jwst.lib.suffix.KNOW_SUFFIXES that
# denote data products; kept static so the check works without jwst installed).
JWST_PRODUCT_SUFFIXES: frozenset[str] = frozenset({
    "uncal", "rate", "rateints", "ramp", "trapsfilled", "dark", "cal", "calints", "crf", "crfints", "bsub",
    "bsubints", "s2d", "s3d", "x1d", "x1dints", "c1d", "i2d", "cat", "segm", "phot", "whtlt", "psfstack",
    "psfalign", "psfsub", "ami", "ami-oi", "aminorm", "aminorm-oi", "masterbg", "wfscmb", "median", "blot",
    "outlier_i2d", "outlier_s2d", "assign_wcs", "flat_field", "srctype", "photom", "extract_2d", "resample",
    "bkgsub", "imprint",
})

# Product types produced by jwstflow / contributed steps (never colliding with the above).
CUSTOM_SUFFIXES: set[str] = {
    "s1d",       # 1-D spectrum extracted by a custom step (x1d table format)
    "bkgspec",   # 1-D background spectrum in x1d format (input for master_background)
    "lsr",       # copy of a cube/spectrum with an LSRK wavelength axis
    "s1dcomb",   # stitched / combined 1-D spectrum
    "psfcube",   # spectrally sub-sampled instrument PSF matching an s3d cube (jwstflow.psf)
    "clustermask",  # bad spaxel clusters found in a per-dither cube (jwstflow.clusters)
}

_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<suffix>[a-z0-9]+(?:-[a-z0-9]+)?)$")

# DMS level-3 product name; `TARGID` is filled by the runner from the MAST log ('t010'),
# falling back to a slug of TARGPROP. cube_build appends the band itself.
DMS_L3_TEMPLATE = "jw{PROGRAM}-o{OBSERVTN}_{TARGID}_{INSTRUME}"


def split_suffix(stem: str) -> tuple[str, str | None]:
    """``'jw..._nrs1_cal' -> ('jw..._nrs1', 'cal')``; unknown trailing tokens are not split."""
    m = _SUFFIX_RE.match(stem)
    if m and (m.group("suffix") in JWST_PRODUCT_SUFFIXES or m.group("suffix") in CUSTOM_SUFFIXES):
        return m.group("base"), m.group("suffix")
    return stem, None


def check_custom_suffix(suffix: str) -> str:
    """Raise if a custom step tries to write an official product type; register new custom suffixes."""
    if suffix in JWST_PRODUCT_SUFFIXES:
        raise ValueError(
            f"suffix {suffix!r} is reserved for jwst pipeline products; custom steps must use "
            "their own suffix (e.g. 's1d' for extracted spectra)"
        )
    if not re.fullmatch(r"[a-z0-9]+", suffix):
        raise ValueError(f"suffix {suffix!r} must be lowercase alphanumeric")
    CUSTOM_SUFFIXES.add(suffix)
    return suffix


def derived_name(source: str | Path, suffix: str, *, descriptor: str | None = None, ext: str = ".fits") -> str:
    """Name of a product derived from ``source`` by a custom step.

    The official suffix of the source is stripped, an optional descriptor
    (e.g. source + aperture) inserted, and the custom suffix appended:
    ``derived_name('jw..._miri_ch1-short_s3d.fits', 's1d', descriptor='eso-ha-569_circle1')``
    -> ``'jw..._miri_ch1-short_eso-ha-569_circle1_s1d.fits'``.
    """
    check_custom_suffix(suffix)
    base, _ = split_suffix(Path(source).stem)
    parts = [base] + ([slug(descriptor)] if descriptor else []) + [suffix]
    return "_".join(parts) + ext


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+_-]+", "-", str(text)).strip("-")


def is_official_product(path: str | Path) -> bool:
    return split_suffix(Path(path).stem)[1] in JWST_PRODUCT_SUFFIXES
