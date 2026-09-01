"""The stitching contract: splice N wavelength-ordered 1-D spectra into one.

Every instrument mode ends with the same move -- several overlapping spectral
segments (NIRSpec gratings, MRS bands, ...) combined into one spectrum. This
module provides :class:`StitchSegments`, a concrete, YAML-usable step that
implements the generic mechanics once:

* segments are read from the EXTRACT1D/COMBINE1D table of each input and
  ordered by wavelength;
* in each overlap the flux ratio between neighbours is measured as the ratio
  of the two overlap medians, and only when both are significantly positive
  (and recorded in the output metadata, whether or not it is applied);
* with ``rescale``, segments are multiplicatively chained onto a ``reference``
  segment (default: the reddest);
* the spectrum switches segments at ``crossovers`` (default: the midpoint of
  each overlap, or of the gap when neighbours do not overlap);
* the result is one ECSV (WAVELENGTH/FLUX/FLUX_ERROR/SEGMENT + provenance
  metadata); the ``plot_stitch`` QA step draws the comparison figure from it
  (data steps never plot -- see ``docs/qa_figures.md``).

Contributed packages subclass it to add mode-specific behaviour -- override
:meth:`segment_label` for naming, :meth:`overlap_ratio` for the measurement
(e.g. masking emission lines out of it), or wrap :meth:`run` for validation
and grouping -- while different plugins keep producing byte-compatible
stitched products.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from .naming import derived_name
from .steps.base import RunContext, Step, StepParams

log = logging.getLogger(__name__)


class StitchSegments(Step):
    """Splice overlapping 1-D spectra (``batch=all``: one task receives every segment).

    Combines the wavelength-ordered segments of one target (NIRSpec gratings,
    MRS bands, ...) into a single spectrum:

    1. each input's EXTRACT1D/COMBINE1D table is read and sorted by
       wavelength; segments are labelled from their headers (grating-filter /
       channel-band) and ordered blue to red;
    2. in every overlap between neighbours the flux ratio is measured -- the
       ratio of the two segments' overlap medians, and only when both stand
       significantly above zero (``min_overlap_snr``), because an overlap
       whose flux is consistent with zero cannot anchor a multiplicative
       rescaling (always recorded in the output metadata; applied only with
       ``rescale``, which multiplicatively chains all segments onto the
       ``reference`` segment -- default the reddest; an unmeasurable ratio
       leaves its link unscaled instead of amplifying noise down the chain);
    3. the combination switches from one segment to the next at the
       ``crossovers`` wavelengths (default: the midpoint of each overlap, or
       of the gap when neighbours do not overlap) -- no averaging across
       segments, each wavelength comes from exactly one;
    4. the result is one ECSV table (WAVELENGTH/FLUX/FLUX_ERROR/SEGMENT plus
       scales, ratios and crossovers as metadata) named ``*_s1dcomb.ecsv``;
       the ``plot_stitch`` QA step draws the comparison figure from it.

    Contributed packages subclass it for mode-specific behaviour (labelling,
    ratio measurement, validation) while producing byte-compatible products.
    """

    inputs = ("*_s1d.fits", "*_x1d.fits", "*_c1d.fits")
    outputs = ("s1dcomb",)
    batch = "all"
    version = "3"   # 2 -> 3: overlap ratio = ratio of overlap medians, guarded by min_overlap_snr

    class Params(StepParams):
        crossovers: list[float] | None = Field(None, description="wavelengths [um] where the combination switches "
                                               "segments (one fewer than segments); default: overlap midpoints")
        rescale: bool = Field(False, description="chain-scale segments onto the reference using the overlap ratios")
        reference: str | None = Field(None, description="segment kept fixed when rescaling: a label, "
                                      "'shortest' or 'longest' (default: the reddest segment)")
        min_overlap_points: int = Field(5, ge=1, description="overlap samples needed to measure a ratio")
        min_overlap_snr: float = Field(3.0, ge=0, description="both overlap medians must sit this many robust "
                                       "sigmas above zero for the ratio to count (a near-zero overlap cannot "
                                       "anchor a rescaling; its link stays at 1). 0 disables the guard")

    def segment_label(self, path: Path) -> str:
        """Short label of a segment (override for mode-specific naming)."""
        from astropy.io import fits

        hdr = fits.getheader(path)
        for keys in (("GRATING", "FILTER"), ("CHANNEL", "BAND")):
            vals = [str(hdr.get(k, "")).strip() for k in keys]
            if all(vals):
                return "-".join(v.lower() for v in vals)
        return Path(path).stem

    def run(self, inputs: list[Path], ctx: RunContext, *, crossovers: list[float] | None = None,
            rescale: bool = False, reference: str | None = None, min_overlap_points: int = 5,
            min_overlap_snr: float = 3.0, **params) -> list[Path]:
        from astropy.table import Table

        if len(inputs) < 2:
            raise ValueError(f"stitching needs at least two segments, got {len(inputs)}")
        segments = []
        for path in sorted(inputs):
            tab = Table.read(path, hdu=self._table_hdu(path))
            arrays = {c: np.asarray(np.ma.filled(tab[c], np.nan), dtype=float)
                      for c in ("WAVELENGTH", "FLUX", "FLUX_ERROR")}
            ok = np.isfinite(arrays["FLUX"]) & np.isfinite(arrays["WAVELENGTH"])
            if not ok.any():
                raise ValueError(f"{path.name}: no finite flux samples")
            order = np.argsort(arrays["WAVELENGTH"][ok])
            segments.append({"path": path, "label": self.segment_label(path),
                             **{c: v[ok][order] for c, v in arrays.items()}})
        segments.sort(key=lambda s: float(s["WAVELENGTH"].min()))
        labels = [s["label"] for s in segments]
        if len(set(labels)) != len(labels):
            raise ValueError(f"segment labels are not unique: {labels}")

        # overlap ratios between neighbours (always measured, applied only with rescale)
        ratios = [self.overlap_ratio(a, b, min_overlap_points=min_overlap_points,
                                     min_overlap_snr=min_overlap_snr)
                  for a, b in zip(segments, segments[1:])]
        scales = self._scales(labels, ratios, reference) if rescale else [1.0] * len(segments)

        if crossovers is not None:
            if len(crossovers) != len(segments) - 1:
                raise ValueError(f"{len(segments)} segments need {len(segments) - 1} crossovers, got {len(crossovers)}")
            bounds = [float(c) for c in crossovers]
        else:  # midpoint of each overlap (or of the gap, when there is none)
            bounds = [0.5 * (float(b["WAVELENGTH"].min()) + float(a["WAVELENGTH"].max()))
                      for a, b in zip(segments, segments[1:])]
        edges = [-np.inf, *bounds, np.inf]
        pieces = []
        for seg, scale, lo, hi in zip(segments, scales, edges[:-1], edges[1:]):
            keep = (seg["WAVELENGTH"] > lo) & (seg["WAVELENGTH"] <= hi)
            pieces.append({"WAVELENGTH": seg["WAVELENGTH"][keep], "FLUX": seg["FLUX"][keep] * scale,
                           "FLUX_ERROR": seg["FLUX_ERROR"][keep] * scale,
                           "SEGMENT": np.full(int(keep.sum()), seg["label"], dtype=object)})
        combined = Table({c: np.concatenate([p[c] for p in pieces]) for c in ("WAVELENGTH", "FLUX", "FLUX_ERROR", "SEGMENT")})
        combined["SEGMENT"] = combined["SEGMENT"].astype(str)
        combined["WAVELENGTH"].unit, combined["FLUX"].unit, combined["FLUX_ERROR"].unit = "um", "Jy", "Jy"
        combined.meta.update({
            "segments": labels, "crossovers_um": [float(b) for b in bounds],
            "neighbour_ratios": [None if not np.isfinite(r) else float(r) for r in ratios],
            "scales_applied": [float(s) for s in scales], "rescale": bool(rescale),
            "inputs": [s["path"].name for s in segments],
        })
        log.info("stitched %s at %s (ratios %s, scales %s)", labels,
                 [f"{b:.3f}" for b in bounds], [f"{r:.4f}" for r in ratios], [f"{s:.4f}" for s in scales])
        out = ctx.output_dir / derived_name(self._product_stem(segments), "s1dcomb", ext=".ecsv")
        combined.write(out, format="ascii.ecsv", overwrite=True)
        return [out]

    # ------------------------------------------------------------------ hooks / helpers
    def overlap_ratio(self, a: dict[str, Any], b: dict[str, Any], *, min_overlap_points: int = 5,
                      min_overlap_snr: float = 3.0) -> float:
        """Flux ratio ``b/a`` where the two segments overlap (NaN when not measurable).

        The ratio of the two segments' median fluxes over the shared overlap
        samples -- robust against spikes and, unlike a median of per-sample
        ratios, against noise crossing zero. With ``min_overlap_snr`` both
        medians must also sit that many robust sigmas above zero: an overlap
        whose flux is consistent with zero cannot anchor a multiplicative
        rescaling (dividing by it amplifies the redder segments by huge or
        even negative factors), so such a ratio comes back NaN -- the chain
        then leaves that link unscaled, with a warning here.

        ``a`` and ``b`` are segment dicts (``label`` plus finite, wavelength-ordered
        ``WAVELENGTH``/``FLUX``/``FLUX_ERROR`` arrays), ``a`` the bluer one. Subclasses
        override this to mask mode-specific wavelength ranges (emission lines, band
        edges) out of the measurement.
        """
        lo, hi = float(b["WAVELENGTH"].min()), float(a["WAVELENGTH"].max())
        inside = (a["WAVELENGTH"] > lo) & (a["WAVELENGTH"] < hi)
        if hi <= lo or int(inside.sum()) < min_overlap_points:
            log.warning("no usable overlap between %s and %s (%.3f-%.3f um)", a["label"], b["label"], lo, hi)
            return float("nan")
        b_on_a = np.interp(a["WAVELENGTH"][inside], b["WAVELENGTH"], b["FLUX"])
        a_flux = a["FLUX"][inside]
        ok = np.isfinite(a_flux) & np.isfinite(b_on_a)
        if int(ok.sum()) < min_overlap_points:
            log.warning("overlap between %s and %s holds %d finite sample(s) (< %d); ratio not measured",
                        a["label"], b["label"], int(ok.sum()), min_overlap_points)
            return float("nan")
        med_a, err_a = _median_and_error(a_flux[ok])
        med_b, err_b = _median_and_error(b_on_a[ok])
        if min_overlap_snr > 0 and not (med_a > min_overlap_snr * err_a and med_b > min_overlap_snr * err_b):
            log.warning("overlap between %s and %s (%.3f-%.3f um): flux consistent with zero "
                        "(%.3g +- %.3g vs %.3g +- %.3g); ratio not measurable, that link is not rescaled",
                        a["label"], b["label"], lo, hi, med_a, err_a, med_b, err_b)
            return float("nan")
        with np.errstate(all="ignore"):
            return med_b / med_a

    @staticmethod
    def _table_hdu(path: Path) -> str:
        from astropy.io import fits

        with fits.open(path) as hdul:
            for name in ("EXTRACT1D", "COMBINE1D"):
                if name in hdul:
                    return name
        raise ValueError(f"{path}: no EXTRACT1D/COMBINE1D extension")

    @staticmethod
    def _scales(labels: list[str], ratios: list[float], reference: str | None) -> list[float]:
        """Chain the neighbour ratios so the reference segment keeps scale 1."""
        if reference in (None, "longest"):
            ref = len(labels) - 1
        elif reference == "shortest":
            ref = 0
        elif reference in labels:
            ref = labels.index(reference)
        else:
            raise ValueError(f"reference segment {reference!r} is not among {labels} (nor 'shortest'/'longest')")
        scales = [1.0] * len(labels)
        for i in range(ref - 1, -1, -1):  # bluewards of the reference
            r = ratios[i]
            scales[i] = scales[i + 1] * (r if np.isfinite(r) else 1.0)
        for i in range(ref, len(labels) - 1):  # redwards of the reference
            r = ratios[i]
            scales[i + 1] = scales[i] / (r if np.isfinite(r) and r != 0 else 1.0)
        return scales

    @staticmethod
    def _product_stem(segments: list[dict[str, Any]]) -> str:
        """Longest common prefix of the segment file names, cut back to a whole ``_`` token."""
        import os

        stems = [s["path"].stem for s in segments]
        prefix = os.path.commonprefix(stems)
        if prefix and len(prefix) < len(stems[0]) and stems[0][len(prefix)] not in "_-":
            prefix = prefix.rsplit("_", 1)[0]  # drop the partial token (g235h... vs g395h... -> ..._nirspec)
        prefix = prefix.rstrip("-_")
        return (prefix or stems[0]) + ".fits"  # derived_name strips the extension again


def _median_and_error(values: np.ndarray) -> tuple[float, float]:
    """Median of ``values`` and the robust (MAD-based) uncertainty of that median."""
    med = float(np.median(values))
    mad_sigma = 1.4826 * float(np.median(np.abs(values - med)))
    return med, 1.2533 * mad_sigma / np.sqrt(len(values))
