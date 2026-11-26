"""The ``*_astrom.fits`` product: the registration of a run's images, in four HDUs.

The layout is described in :mod:`jwstflow.images.astrometry`; every table column carries its unit
(TUNIT) and the primary header the parameters and conventions (JWFAS* keys and COMMENT cards), so a
figure or a later step reads the product without this package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

ASTROM_SUFFIX = "astrom"   # <base>_astrom.fits: the registration of a run's images

#: Columns of the tables: name -> unit (None: dimensionless / text).
IMAGE_COLUMNS: dict[str, str | None] = {
    "file": None, "catalog": None, "instrument": None, "band": None, "pivot": "um", "fwhm": "arcsec",
    "pixscale": "arcsec", "mjd": "d", "epoch": "yr", "n_catalog": None, "n_finite": None, "n_compact": None,
    "n_snr": None, "n_isolated": None, "n_point": None, "snr_column": None, "reference": None,
}
SUMMARY_COLUMNS: dict[str, str | None] = {
    "file": None, "band": None, "pivot": "um", "against": None, "against_name": None, "radius": "arcsec",
    "shift_dra": "mas", "shift_ddec": "mas", "n_matched": None, "n_used": None, "dra": "mas", "ddec": "mas",
    "dra_scatter": "mas", "ddec_scatter": "mas", "dra_std": "mas", "ddec_std": "mas", "dra_error": "mas",
    "ddec_error": "mas", "pixscale": "arcsec",
}
MATCH_COLUMNS: dict[str, str | None] = {
    "file": None, "band": None, "against": None, "id": None, "ref_id": None, "ra": "deg", "dec": "deg",
    "dra": "mas", "ddec": "mas", "snr": None, "ref_has_pm": None, "used": None,
}
#: How the tables are documented in the primary header.
CONVENTIONS: tuple[str, ...] = (
    "Offsets are image minus reference: another image of the run (against = 'reference') or Gaia DR3",
    "propagated to the image's mid-exposure epoch (against = 'gaia'). dra = offset in right ascension",
    "times cos(dec), east positive; ddec = offset in declination, north positive; both in mas.",
    "dra, ddec in SUMMARY: median of the used matches, clipped at JWFASCLP scatters (with 5 or more);",
    "scatter = 1.4826 x MAD; std = standard deviation; error of the median = 1.2533 x max(scatter, std)",
    "/ sqrt(n_used); NaN below JWFASMIN used matches.",
    "shift_dra/ddec: the bulk shift found within JWFASSRC arcsec before matching within radius.",
    "Point sources: catalogue is_extended false, S/N >= JWFASSNR, no catalogue neighbour within",
    "JWFASISO FWHM, data in every pixel within JWFASCOR FWHM; matched within JWFASMRD FWHM of the",
    "coarser band. IMAGES n_*: sources left after each cut, in that order.",
)


def write_astrometry_product(path: Path, *, images: list[dict[str, Any]], summary: list[dict[str, Any]],
                             matches: list[dict[str, Any]], keys: dict[str, Any], primary: Any = None) -> Path:
    """Write ``*_astrom.fits``: PRIMARY (``keys``, observation keys of ``primary``, the conventions),
    IMAGES, SUMMARY and MATCHES (rows as dicts with the columns of :data:`IMAGE_COLUMNS`,
    :data:`SUMMARY_COLUMNS`, :data:`MATCH_COLUMNS`)."""
    from astropy.io import fits

    prim = fits.PrimaryHDU()
    for card in ("TELESCOP", "TARGPROP", "TARGNAME", "TARG_RA", "TARG_DEC", "PROGRAM", "OBSERVTN", "CAL_VER", "CRDS_CTX"):
        if primary is not None and card in primary:
            prim.header[card] = primary[card]
    for k, v in keys.items():
        prim.header[k] = v
    for line in CONVENTIONS:
        prim.header.add_comment(line)
    hdus = [prim, _table_hdu("IMAGES", images, IMAGE_COLUMNS), _table_hdu("SUMMARY", summary, SUMMARY_COLUMNS),
            _table_hdu("MATCHES", matches, MATCH_COLUMNS)]
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return Path(path)


def _table_hdu(name: str, rows: list[dict[str, Any]], columns: dict[str, str | None]) -> Any:
    import numpy as np
    from astropy.io import fits
    from astropy.table import Table

    empty: dict[str, Any] = {"file": "", "band": "", "catalog": "", "instrument": "", "snr_column": "", "against": "",
                             "against_name": ""}
    table = Table()
    for col, unit in columns.items():
        values = [row.get(col) for row in rows]
        if col in empty:
            table[col] = np.array([str(v) for v in values], dtype=str) if values else np.zeros(0, dtype="U1")
        elif col in ("reference", "used", "ref_has_pm"):
            table[col] = np.array(values, dtype=bool)
        elif col.startswith("n_") or col in ("id", "ref_id"):
            table[col] = np.array(values, dtype=np.int64)
        else:
            table[col] = np.array([np.nan if v is None else v for v in values], dtype=float)
        if unit:
            table[col].unit = unit
    return fits.BinTableHDU(table, name=name)
