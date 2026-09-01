"""The stitching contract: splice N wavelength-ordered 1-D spectra into one.

Every instrument mode ends with the same move -- several overlapping spectral
segments (NIRSpec gratings, MRS bands, ...) combined into one spectrum. This
module provides :class:`StitchSegments`, a concrete, YAML-usable step that
implements the generic mechanics once:

* segments are read from the EXTRACT1D/COMBINE1D table of each input and
  ordered by wavelength;
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
  through any later stitch that consumes it;
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
from typing import Any, NamedTuple

import numpy as np
from pydantic import Field

from .naming import derived_name
from .steps.base import RunContext, Step, StepParams

log = logging.getLogger(__name__)


class OverlapRatio(NamedTuple):
    """One neighbour-pair flux-ratio measurement (the result of :meth:`StitchSegments.overlap_ratio`).

    ``ratio`` is the flux ratio ``b/a`` (NaN when not measurable) and
    ``error`` its 1-sigma uncertainty, propagated from the FLUX_ERROR samples
    of the measurement region (NaN when the ratio is; falls back to the
    scatter of the samples when neither segment carries usable errors).
    ``window`` is the wavelength region [um] the measurement actually used --
    the ``ratio_window_frac`` window around the crossover, clipped to the
    overlap (widened back to the whole overlap when the window holds too few
    samples) -- or None when the segments do not overlap. ``n`` counts the
    samples measured.
    """

    ratio: float
    error: float
    window: tuple[float, float] | None
    n: int

    @classmethod
    def unmeasurable(cls, window: tuple[float, float] | None = None, n: int = 0) -> OverlapRatio:
        return cls(float("nan"), float("nan"), window, n)


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

    inputs = ("*_s1d.fits", "*_x1d.fits", "*_c1d.fits")
    outputs = ("s1dcomb",)
    batch = "all"
    version = "4"   # 3 -> 4: ratio from a window around the crossover; uncertainty into FLUX_ERROR

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
        min_overlap_snr: float = Field(3.0, ge=0, description="both windowed medians must sit this many sigmas "
                                       "above zero for the ratio to count (a near-zero overlap cannot anchor a "
                                       "rescaling; its link stays at 1). 0 disables the guard")

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
            rescale: bool = False, reference: str | None = None, ratio_window_frac: float = 0.05,
            min_overlap_points: int = 5, min_overlap_snr: float = 3.0, **params) -> list[Path]:
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

        # crossovers first: the ratio of each neighbour pair is measured around its crossover
        if crossovers is not None:
            if len(crossovers) != len(segments) - 1:
                raise ValueError(f"{len(segments)} segments need {len(segments) - 1} crossovers, got {len(crossovers)}")
            bounds = [float(c) for c in crossovers]
        else:  # midpoint of each overlap (or of the gap, when there is none)
            bounds = [0.5 * (float(b["WAVELENGTH"].min()) + float(a["WAVELENGTH"].max()))
                      for a, b in zip(segments, segments[1:])]

        # ratio +- uncertainty per neighbour pair (always measured, applied only with rescale)
        ratios = [self.overlap_ratio(a, b, crossover=c, ratio_window_frac=ratio_window_frac,
                                     min_overlap_points=min_overlap_points, min_overlap_snr=min_overlap_snr)
                  for a, b, c in zip(segments, segments[1:], bounds)]
        scales, scale_errors = (self._scales(labels, ratios, reference) if rescale
                                else ([1.0] * len(segments), [0.0] * len(segments)))

        edges = [-np.inf, *bounds, np.inf]
        pieces = []
        for seg, scale, scale_err, lo, hi in zip(segments, scales, scale_errors, edges[:-1], edges[1:]):
            keep = (seg["WAVELENGTH"] > lo) & (seg["WAVELENGTH"] <= hi)
            # the scale uncertainty is part of the stitched error budget: it adds in
            # quadrature to the measurement error of every rescaled sample
            error = np.hypot(seg["FLUX_ERROR"][keep] * scale, seg["FLUX"][keep] * scale_err)
            pieces.append({"WAVELENGTH": seg["WAVELENGTH"][keep], "FLUX": seg["FLUX"][keep] * scale,
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

        The measurement region is the wavelength window of full width
        ``ratio_window_frac * crossover`` centred on the crossover, clipped
        to the overlap -- the factor is estimated where it is applied,
        instead of over the whole overlap whose far ends carry the largest
        calibration differences. When the window holds fewer than
        ``min_overlap_points`` finite samples it is widened back to the
        whole overlap (with a warning); when even that holds too few, the
        ratio is unmeasurable (NaN).

        The ratio is the ratio of the two segments' median fluxes over the
        window samples -- robust against spikes and, unlike a median of
        per-sample ratios, against noise crossing zero. Its 1-sigma
        uncertainty is propagated from the segments' FLUX_ERROR samples in
        the window (the error of each median, combined in quadrature as
        relative errors); segments without usable errors fall back to the
        robust scatter of their samples. With ``min_overlap_snr`` both
        medians must also sit that many sigmas above zero -- judged against
        the larger of the propagated and the scatter-based error, so
        underestimated FLUX_ERRORs cannot sneak a near-zero overlap past
        the guard: an overlap whose flux is consistent with zero cannot
        anchor a multiplicative rescaling (dividing by it amplifies the
        redder segments by huge or even negative factors), so such a ratio
        comes back NaN -- the chain then leaves that link unscaled, with a
        warning here.

        ``a`` and ``b`` are segment dicts (``label`` plus finite, wavelength-ordered
        ``WAVELENGTH``/``FLUX``/``FLUX_ERROR`` arrays), ``a`` the bluer one. Subclasses
        override this to mask mode-specific wavelength ranges (emission lines, band
        edges) out of the measurement.
        """
        lo, hi = float(b["WAVELENGTH"].min()), float(a["WAVELENGTH"].max())
        if hi <= lo:
            log.warning("no overlap between %s and %s (%.3f-%.3f um); ratio not measured",
                        a["label"], b["label"], lo, hi)
            return OverlapRatio.unmeasurable()
        half = 0.5 * ratio_window_frac * float(crossover)
        window = (max(lo, float(crossover) - half), min(hi, float(crossover) + half))

        def finite_pairs(wlo: float, whi: float) -> tuple[np.ndarray, ...]:
            inside = (a["WAVELENGTH"] >= wlo) & (a["WAVELENGTH"] <= whi)
            wave = a["WAVELENGTH"][inside]
            a_flux, a_err = a["FLUX"][inside], a["FLUX_ERROR"][inside]
            b_flux = np.interp(wave, b["WAVELENGTH"], b["FLUX"])
            b_err = np.interp(wave, b["WAVELENGTH"], b["FLUX_ERROR"])
            ok = np.isfinite(a_flux) & np.isfinite(b_flux)
            return a_flux[ok], a_err[ok], b_flux[ok], b_err[ok]

        samples = finite_pairs(*window)
        if len(samples[0]) < min_overlap_points and window != (lo, hi):
            log.warning("ratio window %.3f-%.3f um around the %s/%s crossover holds %d finite sample(s) "
                        "(< %d); widening to the whole overlap %.3f-%.3f um",
                        *window, a["label"], b["label"], len(samples[0]), min_overlap_points, lo, hi)
            window = (lo, hi)
            samples = finite_pairs(*window)
        a_flux, a_err, b_flux, b_err = samples
        n = len(a_flux)
        if n < min_overlap_points:
            log.warning("overlap between %s and %s (%.3f-%.3f um) holds %d finite sample(s) (< %d); "
                        "ratio not measured", a["label"], b["label"], lo, hi, n, min_overlap_points)
            return OverlapRatio.unmeasurable(window, n)
        med_a, scatter_a = _median_and_error(a_flux)
        med_b, scatter_b = _median_and_error(b_flux)
        prop_a = _median_error_from_flux_errors(a_err)
        prop_b = _median_error_from_flux_errors(b_err)
        err_a = prop_a if prop_a is not None else scatter_a
        err_b = prop_b if prop_b is not None else scatter_b
        guard_a, guard_b = max(err_a, scatter_a), max(err_b, scatter_b)
        if min_overlap_snr > 0 and not (med_a > min_overlap_snr * guard_a and med_b > min_overlap_snr * guard_b):
            log.warning("ratio window %.3f-%.3f um between %s and %s: flux consistent with zero "
                        "(%.3g +- %.3g vs %.3g +- %.3g); ratio not measurable, that link is not rescaled",
                        *window, a["label"], b["label"], med_a, guard_a, med_b, guard_b)
            return OverlapRatio.unmeasurable(window, n)
        with np.errstate(all="ignore"):
            ratio = med_b / med_a
            error = abs(ratio) * float(np.hypot(err_a / med_a, err_b / med_b))
        return OverlapRatio(float(ratio), float(error), window, n)

    @staticmethod
    def _table_hdu(path: Path) -> str:
        from astropy.io import fits

        with fits.open(path) as hdul:
            for name in ("EXTRACT1D", "COMBINE1D"):
                if name in hdul:
                    return name
        raise ValueError(f"{path}: no EXTRACT1D/COMBINE1D extension")

    @staticmethod
    def _scales(labels: list[str], ratios: list[OverlapRatio], reference: str | None) -> tuple[list[float], list[float]]:
        """Chain the neighbour ratios so the reference segment keeps scale 1 (error 0).

        Each link multiplies (bluewards) or divides (redwards) by its ratio;
        the relative uncertainties of the ratios crossed on the way add in
        quadrature, so a segment far from the reference carries the combined
        uncertainty of every measurement between them. Unmeasurable links
        stay at factor 1 and contribute no uncertainty (the warning was
        already given where the ratio was measured).
        """
        if reference in (None, "longest"):
            ref = len(labels) - 1
        elif reference == "shortest":
            ref = 0
        elif reference in labels:
            ref = labels.index(reference)
        else:
            raise ValueError(f"reference segment {reference!r} is not among {labels} (nor 'shortest'/'longest')")
        scales = [1.0] * len(labels)
        rel_var = [0.0] * len(labels)

        def link(r: OverlapRatio) -> tuple[float, float]:
            if not (np.isfinite(r.ratio) and r.ratio != 0):
                return 1.0, 0.0
            return r.ratio, (r.error / r.ratio) ** 2 if np.isfinite(r.error) else 0.0

        for i in range(ref - 1, -1, -1):  # bluewards of the reference
            factor, var = link(ratios[i])
            scales[i] = scales[i + 1] * factor
            rel_var[i] = rel_var[i + 1] + var
        for i in range(ref, len(labels) - 1):  # redwards of the reference
            factor, var = link(ratios[i])
            scales[i + 1] = scales[i] / factor
            rel_var[i + 1] = rel_var[i] + var
        return scales, [abs(s) * float(np.sqrt(v)) for s, v in zip(scales, rel_var)]

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


def _median_error_from_flux_errors(errors: np.ndarray) -> float | None:
    """Uncertainty of the median of N samples, propagated from their per-sample errors.

    The error of the mean is ``sqrt(mean(sigma_i^2) / N)``; the median of a
    Gaussian sample is ``sqrt(pi/2) = 1.2533`` times noisier. Samples without
    a usable error are left out of the typical sigma but still count toward
    N (they are part of the median); None when no sample carries one.
    """
    errors = np.asarray(errors, dtype=float)
    usable = errors[np.isfinite(errors) & (errors > 0)]
    if not len(usable):
        return None
    return 1.2533 * float(np.sqrt(np.mean(usable**2) / len(errors)))
