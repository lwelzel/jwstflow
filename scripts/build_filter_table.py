"""Rebuild the table of JWST imaging filters from the filter throughput curves of stpsf.

Run it from the repository root with the stpsf data files at hand::

    STPSF_PATH=/path/to/stpsf-data python scripts/build_filter_table.py            # report only
    STPSF_PATH=/path/to/stpsf-data python scripts/build_filter_table.py --write    # update the ECSV

For every ``<INSTRUMENT>/filters/<FILTER>_throughput.fits`` of NIRCam, NIRISS and MIRI (the
imaging and coronagraphic filters; NIRCam's WLP4 weak lens, NIRISS' CLEAR and MIRI's neutral
density FND are not filters a science image is named by and are skipped) it records

* ``pivot_um`` -- the pivot wavelength, sqrt(int T lambda dlambda / int T / lambda dlambda): the
  wavelength at which a flux density per unit frequency and per unit wavelength convert into each
  other exactly for the band (Tokunaga & Vacca 2005, PASP 117, 421);
* ``width_um`` -- the equivalent width int T dlambda / max T;
* ``wave_min_um`` / ``wave_max_um`` -- where the throughput crosses half its maximum, outermost;

and, in ``jwst_imaging_throughputs.ecsv``, the curve itself (relative throughput against wavelength,
200 samples over where it exceeds 0.1 % of its peak) for band averages of spectra
(``jwstflow.images.filters.band_average``).

The curves are the relative system throughputs stpsf ships (normalised to a peak of one), so the
table carries wavelengths and widths, not absolute throughputs. Nothing is written without
``--write``; the script prints every row.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

TABLE = Path(__file__).resolve().parents[1] / "src" / "jwstflow" / "refdata" / "filters" / "jwst_imaging_filters.ecsv"
CURVES = TABLE.with_name("jwst_imaging_throughputs.ecsv")
#: The curves are kept at this many samples per filter (linear interpolation of stpsf's, over the range
#: where the throughput exceeds CURVE_FLOOR of its peak): enough for band averages to 1e-4.
CURVE_SAMPLES, CURVE_FLOOR = 200, 1e-3
INSTRUMENTS = {"NIRCam": "NIRCAM", "NIRISS": "NIRISS", "MIRI": "MIRI"}
SKIP = {"WLP4", "CLEAR", "FND"}


def band_numbers(wave_um: np.ndarray, throughput: np.ndarray) -> tuple[float, float, float, float]:
    """(pivot, equivalent width, lower and upper half-maximum crossing) of one curve [um]."""
    order = np.argsort(wave_um)
    w, t = wave_um[order], np.clip(throughput[order], 0.0, None)
    pivot = float(np.sqrt(np.trapezoid(t * w, w) / np.trapezoid(t / w, w)))
    width = float(np.trapezoid(t, w) / t.max())
    above = np.nonzero(t >= 0.5 * t.max())[0]
    lo, hi = above[0], above[-1]
    # linear interpolation of the half-maximum crossings
    w_lo = float(np.interp(0.5 * t.max(), [t[lo - 1], t[lo]], [w[lo - 1], w[lo]])) if lo > 0 else float(w[lo])
    w_hi = float(np.interp(0.5 * t.max(), [t[hi + 1], t[hi]], [w[hi + 1], w[hi]])) if hi + 1 < w.size else float(w[hi])
    return pivot, width, w_lo, w_hi


def main(argv: list[str] | None = None) -> int:
    from astropy.io import fits
    from astropy.table import Table

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="write the ECSV table")
    args = parser.parse_args(argv)
    root = Path(os.environ.get("STPSF_PATH", "~/data/stpsf-data")).expanduser()
    version = (root / "version.txt").read_text().strip() if (root / "version.txt").exists() else "unknown"
    rows, curves = [], []
    for folder, instrument in INSTRUMENTS.items():
        for path in sorted((root / folder / "filters").glob("*_throughput.fits")):
            name = path.name.split("_")[0].upper()
            if name in SKIP:
                continue
            data = fits.getdata(path, 1)
            wave = np.asarray(data["WAVELENGTH"], dtype=float) * 1e-4        # Angstrom -> um
            pivot, width, lo, hi = band_numbers(wave, np.asarray(data["THROUGHPUT"], dtype=float))
            rows.append((instrument, name, round(pivot, 4), round(width, 4), round(lo, 4), round(hi, 4)))
            t = np.clip(np.asarray(data["THROUGHPUT"], dtype=float), 0.0, None)
            order = np.argsort(wave)
            inside = np.nonzero(t[order] >= CURVE_FLOOR * t.max())[0]
            grid = np.linspace(wave[order][inside[0]], wave[order][inside[-1]], CURVE_SAMPLES)
            for w, v in zip(grid, np.interp(grid, wave[order], t[order]) / t.max()):
                curves.append((instrument, name, round(float(w), 5), round(float(v), 5)))
            print(f"{instrument:7s} {name:7s} pivot {pivot:8.4f} um  width {width:7.4f} um  "
                  f"half-max {lo:8.4f}-{hi:8.4f} um")
    table = Table(rows=rows, names=("instrument", "filter", "pivot_um", "width_um", "wave_min_um", "wave_max_um"))
    for col, desc in (("pivot_um", "pivot wavelength sqrt(int T l dl / int T / l dl)"),
                      ("width_um", "equivalent width int T dl / max T"),
                      ("wave_min_um", "lower half-maximum crossing"),
                      ("wave_max_um", "upper half-maximum crossing")):
        table[col].unit = "um"
        table[col].description = desc
    table.meta.update({
        "dataset": "jwst_imaging_filters",
        "description": "JWST NIRCam, NIRISS and MIRI imaging filters: pivot wavelength, equivalent width, half-maximum edges",
        "provenance": f"computed by scripts/build_filter_table.py from the relative system throughput curves of stpsf "
                      f"(data version {version}); pivot wavelength after Tokunaga & Vacca 2005, PASP 117, 421",
        "usage": "jwstflow.images.filters.band_of(instrument, filter, pupil)",
    })
    curve_table = Table(rows=curves, names=("instrument", "filter", "wavelength_um", "throughput"))
    curve_table["wavelength_um"].unit = "um"
    curve_table["throughput"].description = "relative system throughput (peak 1)"
    curve_table.meta.update({
        "dataset": "jwst_imaging_throughputs",
        "description": f"the curves of jwst_imaging_filters: {CURVE_SAMPLES} samples per filter where the throughput "
                       f"exceeds {CURVE_FLOOR:g} of its peak",
        "provenance": table.meta["provenance"],
        "usage": "jwstflow.images.filters.throughput(band), band_average(wavelength, flux, band)",
    })
    if args.write:
        TABLE.parent.mkdir(parents=True, exist_ok=True)
        table.write(TABLE, format="ascii.ecsv", overwrite=True)
        curve_table.write(CURVES, format="ascii.ecsv", overwrite=True)
        print(f"wrote {len(table)} filters to {TABLE} and their curves to {CURVES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
