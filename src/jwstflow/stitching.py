"""The stitching contract: splice N wavelength-ordered 1-D spectra into one.

Every instrument mode ends with the same move -- several overlapping spectral
segments (NIRSpec gratings, MRS bands, ...) combined into one spectrum. This
module provides :class:`StitchSegments`, a concrete, YAML-usable step that
implements the generic mechanics once:

* segments are read from the EXTRACT1D/COMBINE1D table of each input and
  ordered by wavelength;
* in each overlap the median flux ratio between neighbours is measured (and
  recorded in the output metadata, whether or not it is applied);
* with ``rescale``, segments are multiplicatively chained onto a ``reference``
  segment (default: the reddest);
* the spectrum switches segments at ``crossovers`` (default: the midpoint of
  each overlap, or of the gap when neighbours do not overlap);
* the result is one ECSV (WAVELENGTH/FLUX/FLUX_ERROR/SEGMENT + provenance
  metadata) and a comparison plot.

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
    """Splice overlapping 1-D spectra (``batch=all``: one task receives every segment)."""

    inputs = ("*_s1d.fits", "*_x1d.fits", "*_c1d.fits")
    outputs = ("s1dcomb",)
    batch = "all"
    version = "1"

    class Params(StepParams):
        crossovers: list[float] | None = Field(None, description="wavelengths [um] where the combination switches "
                                               "segments (one fewer than segments); default: overlap midpoints")
        rescale: bool = Field(False, description="chain-scale segments onto the reference using the overlap ratios")
        reference: str | None = Field(None, description="segment kept fixed when rescaling: a label, "
                                      "'shortest' or 'longest' (default: the reddest segment)")
        min_overlap_points: int = Field(5, ge=1, description="overlap samples needed to measure a ratio")

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
            **params) -> list[Path]:
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
        ratios = [self.overlap_ratio(a, b, min_overlap_points=min_overlap_points)
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
        outputs = [out]
        png = _plot(out.with_suffix(".png"), segments, scales, combined)
        if png:
            outputs.append(png)
        return outputs

    # ------------------------------------------------------------------ hooks / helpers
    def overlap_ratio(self, a: dict[str, Any], b: dict[str, Any], *, min_overlap_points: int = 5) -> float:
        """Median flux ratio ``b/a`` where the two segments overlap (NaN when not measurable).

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
        with np.errstate(all="ignore"):
            return float(np.nanmedian(b_on_a / a["FLUX"][inside]))

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


def _plot(path: Path, segments: list[dict[str, Any]], scales: list[float], combined: Any) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(11, 4))
    for seg, scale in zip(segments, scales):
        ax.plot(seg["WAVELENGTH"], seg["FLUX"] * scale, lw=0.6, alpha=0.5,
                label=f"{seg['label']}" + (f" x {scale:.3f}" if scale != 1.0 else ""))
    ax.plot(combined["WAVELENGTH"], combined["FLUX"], "k", lw=0.6, label="stitched")
    ax.set(xlabel="wavelength [um]", ylabel="flux [Jy]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
