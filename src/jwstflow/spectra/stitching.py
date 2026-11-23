"""The stitching contract: splice N wavelength-ordered 1-D spectra into one.

Every instrument mode ends with the same move -- several overlapping spectral
segments (NIRSpec gratings, MRS bands, ...) combined into one spectrum. This
module provides :class:`StitchSegments`, a concrete, YAML-usable step that
implements the generic mechanics once:

* segments are read from the EXTRACT1D/COMBINE1D table of each input and
  ordered by wavelength; samples with a non-zero DQ (DO_NOT_USE, or e.g.
  FLUX_ESTIMATED planes next to a detector gap) are left out, as
  :attr:`~jwstflow.spectra.Spectrum1D.good` does;
* the spectrum switches segments at ``crossovers`` (default: the midpoint of
  each overlap, or of the gap when neighbours do not overlap);
* around each crossover the flux ratio between neighbours is measured inside
  a small wavelength window (its full width a fraction
  ``ratio_window_frac`` of the crossover wavelength) as the ratio of the two
  windowed medians, with a 1-sigma uncertainty propagated from the segments'
  FLUX_ERROR samples in that window -- and only when both medians are
  significantly positive (recorded in the output metadata, whether or not it
  is applied);
* with ``rescale``, segments are multiplicatively chained onto a
  ``reference`` segment (default: the reddest); the scale uncertainties
  accumulate in quadrature along the chain and are folded into the output
  FLUX_ERROR, so the scaling uncertainty propagates into the spectrum and
  through any later stitch that consumes it. ``combine: add`` chains
  additive offsets instead (flux differences around the crossovers), for
  mismatches that are additive -- a residual background over an extended
  aperture -- which a factor would carry into every feature's depth;
* ``fixed_scales`` multiplies named segments by a given factor (with its
  uncertainty) -- a calibration or epoch factor measured elsewhere, e.g. the
  flux change of a variable source between two instruments' epochs measured
  in identical apertures on the cubes -- on top of any measured rescaling;
* inputs may also be stitched spectra of earlier runs (``*_s1dcomb.ecsv``),
  each one segment (labelled ``nirspec`` / ``miri`` from its name), so a
  combine can join two instruments once instead of re-chaining all bands;
* the result is one ECSV (WAVELENGTH/FLUX/FLUX_ERROR/SEGMENT + provenance
  metadata); the ``plot_stitch`` QA step draws the comparison figure and
  ``plot_stitch_overlaps`` the per-overlap measurement figure from it
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

from ..steps import RunContext, Step, StepParams
from ..steps.naming import derived_name
from .overlaps import (
    OverlapRatio,
    chain_offsets,
    chain_scales,
    measure_overlap_offset,
    measure_overlap_ratio,
    overlap_midpoints,
)

log = logging.getLogger(__name__)


class StitchSegments(Step):
    """Splice overlapping 1-D spectra (``batch=all``: one task receives every segment).

    Combines the wavelength-ordered segments of one target (NIRSpec gratings,
    MRS bands, ...) into a single spectrum:

    1. each input's EXTRACT1D/COMBINE1D table is read and sorted by
       wavelength; segments are labelled from their headers (grating-filter /
       channel-band) and ordered blue to red;
    2. the combination switches from one segment to the next at the
       ``crossovers`` wavelengths (default: the midpoint of each overlap, or
       of the gap when neighbours do not overlap) -- no averaging across
       segments, each wavelength comes from exactly one;
    3. in every overlap the flux ratio of the neighbours is measured *around
       the crossover*: inside a wavelength window of full width
       ``ratio_window_frac`` times the crossover wavelength, clipped to the
       overlap (a whole-overlap median would mix in regions far from the
       splice, where the segments' calibrations differ most -- the window
       measures the factor where it is actually applied). The ratio of the
       two windowed medians carries a 1-sigma uncertainty propagated from
       the FLUX_ERROR samples in the window, and counts only when both
       medians stand significantly above zero (``min_overlap_snr``), because
       an overlap whose flux is consistent with zero cannot anchor a
       multiplicative rescaling. Ratios, uncertainties and the windows used
       are always recorded in the output metadata; they are applied only
       with ``rescale``, which multiplicatively chains all segments onto the
       ``reference`` segment -- default the reddest; an unmeasurable ratio
       leaves its link unscaled instead of amplifying noise down the chain.
       The chained scale uncertainties (relative errors added in quadrature
       link by link) are folded into the output FLUX_ERROR, so the scaling
       uncertainty propagates into the spectrum and through any later stitch
       that consumes it;
    4. the result is one ECSV table (WAVELENGTH/FLUX/FLUX_ERROR/SEGMENT plus
       scales, ratios, uncertainties, windows and crossovers as metadata)
       named ``*_s1dcomb.ecsv``; the ``plot_stitch`` QA step draws the
       comparison figure and ``plot_stitch_overlaps`` the per-overlap
       measurement figure from it.

    Contributed packages subclass it for mode-specific behaviour (labelling,
    ratio measurement, validation) while producing byte-compatible products.
    """

    inputs = ("*_s1d.fits", "*_x1d.fits", "*_c1d.fits", "*_s1dcomb.ecsv")
    outputs = ("s1dcomb",)
    batch = "all"
    version = "7"   # 6 -> 7: min/max_wavelength; 5 -> 6: combine add|multiply, fixed_scales, stitched ECSV inputs (4 -> 5: DQ != 0 not spliced)

    class Params(StepParams):
        crossovers: list[float] | None = Field(None, description="wavelengths [um] where the combination switches "
                                               "segments (one fewer than segments); default: overlap midpoints")
        rescale: bool = Field(False, description="chain-scale segments onto the reference using the overlap ratios")
        reference: str | None = Field(None, description="segment kept fixed when rescaling: a label, "
                                      "'shortest' or 'longest' (default: the reddest segment)")
        ratio_window_frac: float = Field(0.05, gt=0, description="full width of the wavelength window around each "
                                         "crossover from which the multiply factor (and its uncertainty) is "
                                         "measured, as a fraction of the crossover wavelength; the window is "
                                         "clipped to the overlap, and widened back to the whole overlap when it "
                                         "holds fewer than min_overlap_points samples")
        min_overlap_points: int = Field(5, ge=1, description="samples needed in the ratio window to measure a ratio")
        combine: str = Field("multiply", pattern="^(multiply|add)$", description="how rescale joins segments: "
                             "multiply (factors) or add (offsets, for additive mismatches such as a residual "
                             "background over an extended aperture)")
        fixed_scales: dict[str, list[float]] = Field(default_factory=dict, description="segment label -> [factor, "
                                                     "1-sigma error]: a factor measured elsewhere, applied on top of "
                                                     "any rescaling, its error folded into FLUX_ERROR")
        min_wavelength: float | None = Field(None, gt=0, description="samples bluer than this [um] are left out "
                                             "of every segment (a noisy band end)")
        max_wavelength: float | None = Field(None, gt=0, description="samples redder than this [um] are left out "
                                             "of every segment (e.g. the end of MRS ch4-long, S/N < 2 past 27.5 um)")
        min_overlap_snr: float = Field(3.0, ge=0, description="both windowed medians must sit this many sigmas "
                                       "above zero for the ratio to count (a near-zero overlap cannot anchor a "
                                       "rescaling; its link stays at 1). 0 disables the guard")

    def segment_label(self, path: Path) -> str:
        """Short label of a segment (override for mode-specific naming)."""
        from astropy.io import fits

        if Path(path).suffix == ".ecsv":       # a stitched spectrum of an earlier run: one segment
            name = Path(path).name.lower()
            for token in ("nirspec", "miri", "nircam", "niriss"):
                if f"_{token}" in name:
                    return token
            return Path(path).stem

        hdr = fits.getheader(path)
        for keys in (("GRATING", "FILTER"), ("CHANNEL", "BAND")):
            vals = [str(hdr.get(k, "")).strip() for k in keys]
            if all(vals):
                return "-".join(v.lower() for v in vals)
        return Path(path).stem

    def run(self, inputs: list[Path], ctx: RunContext, *, crossovers: list[float] | None = None,
            rescale: bool = False, reference: str | None = None, ratio_window_frac: float = 0.05,
            min_overlap_points: int = 5, min_overlap_snr: float = 3.0, combine: str = "multiply",
            fixed_scales: dict[str, list[float]] | None = None, min_wavelength: float | None = None,
            max_wavelength: float | None = None, **params) -> list[Path]:
        from astropy.table import Table

        if len(inputs) < 2:
            raise ValueError(f"stitching needs at least two segments, got {len(inputs)}")
        segments = []
        for path in sorted(inputs):
            tab = Table.read(path, format="ascii.ecsv") if path.suffix == ".ecsv" else \
                Table.read(path, hdu=self._table_hdu(path))
            arrays = {c: np.asarray(np.ma.filled(tab[c], np.nan), dtype=float)
                      for c in ("WAVELENGTH", "FLUX", "FLUX_ERROR")}
            ok = np.isfinite(arrays["FLUX"]) & np.isfinite(arrays["WAVELENGTH"])
            if min_wavelength is not None:
                ok &= arrays["WAVELENGTH"] >= float(min_wavelength)
            if max_wavelength is not None:
                ok &= arrays["WAVELENGTH"] <= float(max_wavelength)
            if "DQ" in tab.colnames:  # flagged samples (DO_NOT_USE, FLUX_ESTIMATED, ...) are not spliced
                ok &= np.asarray(np.ma.filled(tab["DQ"], 1)).astype(np.int64) == 0
            if not ok.any():
                raise ValueError(f"{path.name}: no finite flux samples")
            order = np.argsort(arrays["WAVELENGTH"][ok])
            segments.append({"path": path, "label": self.segment_label(path),
                             **{c: v[ok][order] for c, v in arrays.items()}})
        segments.sort(key=lambda s: float(s["WAVELENGTH"].min()))
        labels = [s["label"] for s in segments]
        if len(set(labels)) != len(labels):
            raise ValueError(f"segment labels are not unique: {labels}")

        # crossovers first: the ratio of each neighbour pair is measured around its crossover
        if crossovers is not None:
            if len(crossovers) != len(segments) - 1:
                raise ValueError(f"{len(segments)} segments need {len(segments) - 1} crossovers, got {len(crossovers)}")
            bounds = [float(c) for c in crossovers]
        else:  # midpoint of each overlap (or of the gap, when there is none)
            bounds = overlap_midpoints([(float(s["WAVELENGTH"].min()), float(s["WAVELENGTH"].max())) for s in segments])

        # ratio +- uncertainty per neighbour pair (always measured, applied only with rescale)
        ratios = [self.overlap_ratio(a, b, crossover=c, ratio_window_frac=ratio_window_frac,
                                     min_overlap_points=min_overlap_points, min_overlap_snr=min_overlap_snr)
                  for a, b, c in zip(segments, segments[1:], bounds)]
        offsets = [measure_overlap_offset(a, b, crossover=c, ratio_window_frac=ratio_window_frac,
                                          min_overlap_points=min_overlap_points)
                   for a, b, c in zip(segments, segments[1:], bounds)] if combine == "add" else []
        scales, scale_errors = (chain_scales(labels, ratios, reference) if rescale and combine == "multiply"
                                else ([1.0] * len(segments), [0.0] * len(segments)))
        shifts, shift_errors = (chain_offsets(labels, offsets, reference) if rescale and combine == "add"
                                else ([0.0] * len(segments), [0.0] * len(segments)))
        for label, value in (fixed_scales or {}).items():
            if label not in labels:
                raise ValueError(f"fixed_scales: segment {label!r} is not among {labels}")
            factor, ferr = float(value[0]), float(value[1]) if len(value) > 1 else 0.0
            i = labels.index(label)
            rel = np.hypot(scale_errors[i] / scales[i] if scales[i] else 0.0, ferr / factor if factor else 0.0)
            scales[i] = scales[i] * factor
            scale_errors[i] = abs(scales[i]) * float(rel)

        edges = [-np.inf, *bounds, np.inf]
        pieces = []
        for seg, scale, scale_err, shift, shift_err, lo, hi in zip(segments, scales, scale_errors, shifts,
                                                                   shift_errors, edges[:-1], edges[1:]):
            keep = (seg["WAVELENGTH"] > lo) & (seg["WAVELENGTH"] <= hi)
            # the scale (offset) uncertainty is part of the stitched error budget: it adds in
            # quadrature to the measurement error of every rescaled sample
            error = np.hypot(np.hypot(seg["FLUX_ERROR"][keep] * scale, seg["FLUX"][keep] * scale_err), shift_err)
            pieces.append({"WAVELENGTH": seg["WAVELENGTH"][keep], "FLUX": seg["FLUX"][keep] * scale + shift,
                           "FLUX_ERROR": error,
                           "SEGMENT": np.full(int(keep.sum()), seg["label"], dtype=object)})
        combined = Table({c: np.concatenate([p[c] for p in pieces]) for c in ("WAVELENGTH", "FLUX", "FLUX_ERROR", "SEGMENT")})
        combined["SEGMENT"] = combined["SEGMENT"].astype(str)
        combined["WAVELENGTH"].unit, combined["FLUX"].unit, combined["FLUX_ERROR"].unit = "um", "Jy", "Jy"
        combined.meta.update({
            "segments": labels, "crossovers_um": [float(b) for b in bounds],
            "neighbour_ratios": [None if not np.isfinite(r.ratio) else float(r.ratio) for r in ratios],
            "neighbour_ratio_errors": [None if not np.isfinite(r.error) else float(r.error) for r in ratios],
            "ratio_windows_um": [None if r.window is None else [float(r.window[0]), float(r.window[1])]
                                 for r in ratios],
            "ratio_window_frac": float(ratio_window_frac),
            "scales_applied": [float(s) for s in scales],
            "scale_errors": [float(e) for e in scale_errors],
            "rescale": bool(rescale),
            "combine": combine,
            "neighbour_offsets": [None if not np.isfinite(o.ratio) else float(o.ratio) for o in offsets],
            "neighbour_offset_errors": [None if not np.isfinite(o.error) else float(o.error) for o in offsets],
            "offsets_applied": [float(x) for x in shifts],
            "offset_errors": [float(e) for e in shift_errors],
            "fixed_scales": {k: [float(x) for x in v] for k, v in (fixed_scales or {}).items()},
            "inputs": [s["path"].name for s in segments],
        })
        log.info("stitched %s at %s (ratios %s, scales %s)", labels, [f"{b:.3f}" for b in bounds],
                 [f"{r.ratio:.4f}+-{r.error:.4f}" for r in ratios],
                 [f"{s:.4f}+-{e:.4f}" for s, e in zip(scales, scale_errors)])
        out = ctx.output_dir / derived_name(self._product_stem(segments), "s1dcomb", ext=".ecsv")
        combined.write(out, format="ascii.ecsv", overwrite=True)
        return [out]

    # ------------------------------------------------------------------ hooks / helpers
    def overlap_ratio(self, a: dict[str, Any], b: dict[str, Any], *, crossover: float,
                      ratio_window_frac: float = 0.05, min_overlap_points: int = 5,
                      min_overlap_snr: float = 3.0) -> OverlapRatio:
        """Flux ratio ``b/a`` around the crossover, with its uncertainty (:class:`OverlapRatio`).

        The measurement itself is :func:`jwstflow.spectra.overlaps.measure_overlap_ratio`
        (the windowed ratio of medians, its propagated uncertainty, the
        significance guard) -- shared with the cube stitcher. ``a`` and ``b``
        are segment dicts (``label`` plus finite, wavelength-ordered
        ``WAVELENGTH``/``FLUX``/``FLUX_ERROR`` arrays), ``a`` the bluer one.
        Subclasses override this to mask mode-specific wavelength ranges
        (emission lines, band edges) out of the measurement.
        """
        return measure_overlap_ratio(a, b, crossover=crossover, ratio_window_frac=ratio_window_frac,
                                     min_overlap_points=min_overlap_points, min_overlap_snr=min_overlap_snr)

    @staticmethod
    def _table_hdu(path: Path) -> str:
        from astropy.io import fits

        with fits.open(path) as hdul:
            for name in ("EXTRACT1D", "COMBINE1D"):
                if name in hdul:
                    return name
        raise ValueError(f"{path}: no EXTRACT1D/COMBINE1D extension")

    @staticmethod
    def _product_stem(segments: list[dict[str, Any]]) -> str:
        """Longest common prefix of the segment file names, cut back to a whole ``_`` token."""
        import os

        stems = [s["path"].stem for s in segments]
        prefix = os.path.commonprefix(stems)
        if prefix and len(prefix) < len(stems[0]) and stems[0][len(prefix)] not in "_-":
            # drop the partial token (g235h... vs g395h... -> ..._nirspec); without an underscore in the
            # prefix (two observations: jw01751-o006... vs jw01751-o010...) cut at the last hyphen
            prefix = prefix.rsplit("_", 1)[0] if "_" in prefix else prefix.rsplit("-", 1)[0]
        prefix = prefix.rstrip("-_")
        return (prefix or stems[0]) + ".fits"  # derived_name strips the extension again
