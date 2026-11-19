"""Annotation of spectral features on a spectrum axes: :func:`annotate_features` records
what to mark; :class:`~jwstflow.qa.layout.FeatureLayout` places it when the figure is saved.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .layout import FEATURE_LABEL_OFFSET, FeatureLayout
from .style import _defer

log = logging.getLogger(__name__)

#: Bands of the gas catalogue narrower than this fraction of their wavelength (Q-branches,
#: bandheads) are annotated like lines; broader ones are rectangles in the band rows.
FEATURE_NARROW_BAND = 0.02
#: The opt-in catalogue of stellar photospheric absorption lines (:mod:`jwstflow.spectra.features`).
PHOTOSPHERIC_DATASET = "photospheric_lines"
#: The opt-in table of further gas lines, annotated together with ``gas_lines``.
EXTENDED_DATASET = "gas_lines_extended"


def annotate_features(ax: Any, features: Any, *, wave_min: float, wave_max: float,
                      line_width_um: float = 0.02, resolving_power: float | None = None,
                      max_line_labels: int = 250, label_offset: float = FEATURE_LABEL_OFFSET,
                      under: tuple[Any, Any] | None = None, photospheric: bool = True,
                      extended: bool = True, detected_only: bool = False, series: str = "labels") -> None:
    """Mark the selected spectral features on a spectrum axes (x = wavelength [um]).

    ``features`` is a jwstflow feature selection (the ``features:`` grammar of
    :mod:`jwstflow.spectra.features`; ``"all"`` takes every default dataset), restricted
    to ``wave_min``-``wave_max`` (the plotted wavelength range). The
    annotation is laid out when the figure is saved (:func:`save`); the
    spectrum stays the main element of the figure:

    * **Every line in range is marked, whether the spectrum shows it or
      not** -- the figure tells where lines *would* be.
    * **Gas lines** are labelled *in the plot, right above their own line*,
      always with a stem to it: the label starts a constant factor
      (``label_offset``) above the spectrum there, in the **color of the
      species** (:data:`SPECIES_COLORS`, sampled from the standard
      colormap). When the selection contains gas lines, the opt-in
      ``gas_lines_extended`` table is annotated with them
      (``extended=False`` keeps to the selection). Narrow bands of the gas
      catalogue (Q-branches) count as lines.
    * **Each label stands at its own line**: a constant factor
      (``label_offset``) above the spectrum there, so the labels follow the
      spectrum instead of forming a row. A line that towers over its
      neighbours keeps its label on top of itself and the labels of the
      neighbours **dodge left or right** of it; labels that would touch are
      moved apart, and only labels whose connectors would otherwise cross
      share a level. Where lines crowd so much that a single row of labels
      would have to move far from them, the labels alternate between two
      tiers, the stem of an upper label rising between its neighbours.
    * **Series** (the H I series, the branches of the H2 bands) are
      labelled line by line like everything else (``series="labels"``).
      Only where consecutive members of one series lie closer than a label
      is wide do they share one label for the range (``H I Pf 17-25``),
      with a stem from every member -- no bracket, no comb. With
      ``series="combs"`` every series with :data:`COMB_MIN` or more members
      in view is instead one named row above the plot -- a tick and a
      number per member: the compact inventory, for a figure of its own.
    * **Broad gas emission** (the CO fundamental, OH and water forests) is
      marked above the plot as well: outlined rectangles over the wavelength
      range, in the species color.
    * **Photospheric lines** of the star (the opt-in ``photospheric_lines``
      dataset; ``photospheric=False`` leaves them out) are the mirror image
      of the gas lines: dashed, labelled right *below* their absorption
      line, in one family of greens (:data:`PHOTOSPHERE_COLORS`). Lines of
      one species that stand side by side are one multiplet: a single label
      (``Na I doublet``, ``Ca I triplet``, ``Ca I (x4)``) with a stem from
      every line.
    * **Solid-state bands** (ices, silicates, PAH) are rectangles over their
      wavelength range in compact rows at the foot of the plot area, colored
      by group (:data:`BAND_COLORS`), all bands of a species in one row, the
      typeset formula beside the rectangle. The rows share the room below
      the spectrum with the photospheric labels: those stand clear of the
      bands, and a photospheric line that has no room beside them is
      labelled above the spectrum instead.
    * The spectrum is searched for peaks that stand out of its running
      median (:func:`detect_lines`): a strong one that no catalogue line
      explains gets a grey ``?`` over it (inside a broad gas band only when
      it towers over the band's own forest), and ``detected_only=True``
      restricts the marks to the lines found.
    * Labels are typeset: formulae with sub- and superscripts, Greek
      letters for the hydrogen series; the label of an ice band is the
      formula of its species.
    * **Nothing crosses and nothing overlaps.** Neighbouring labels that
      would touch are moved apart as a group, and each is joined to its line
      by one smooth curve; the labels of a group keep the order of their
      lines, so the curves cannot cross. No label is dropped while the axes
      has room for all of them.
    * The y range gains only the room the labels need; the strips above and
      below are a few rows high.

    The annotation follows everything plotted in the axes, or the one curve
    given as ``under`` (``(wavelength, flux)``). Empty selections draw
    nothing. The annotation adds no legend entries: every mark carries its
    own label, in the color of its class.
    """
    from ..spectra.features import select_features

    if features in (None, False) or not np.isfinite([wave_min, wave_max]).all() or wave_max <= wave_min:
        return
    selected = select_features(features, wave_min=wave_min, wave_max=wave_max)
    stellar: list[Any] = []
    if photospheric:
        try:
            stellar = select_features({PHOTOSPHERIC_DATASET: "all"}, wave_min=wave_min, wave_max=wave_max)
        except (FileNotFoundError, ValueError) as exc:
            log.debug("photospheric lines not annotated: %s", exc)
    if extended and any(f.dataset == "gas_lines" and f.kind == "line" for f in selected):
        try:
            further = select_features({EXTENDED_DATASET: "all"}, wave_min=wave_min, wave_max=wave_max)
        except (FileNotFoundError, ValueError) as exc:
            further = []
            log.debug("extended gas lines not annotated: %s", exc)
        selected += [f for f in further if f not in selected]
    stellar += [f for f in selected if f.dataset == PHOTOSPHERIC_DATASET and f not in stellar]
    selected = [f for f in selected if f.dataset != PHOTOSPHERIC_DATASET]
    log_axis = ax.get_xscale() == "log"

    def centre(lo: float, hi: float) -> float:
        return float(np.sqrt(lo * hi)) if log_axis and lo > 0 else 0.5 * (lo + hi)

    record = ax.__dict__.setdefault("_qafig_annotation", {"lines": [], "bands": [], "legend": set()})
    first = not (record["lines"] or record["bands"])
    record.update(offset=float(label_offset), max_line_labels=int(max_line_labels))
    if under is not None:
        record["under"] = under
    for group, cls in ((selected, None), (stellar, "photosphere")):
        for f in group:
            lo, hi = f.window(line_width_um, resolving_power)
            narrow = f.kind != "line" and f.dataset in ("gas_lines", EXTENDED_DATASET) and (hi - lo) <= FEATURE_NARROW_BAND * centre(lo, hi)
            if f.kind == "line" or narrow:
                w = f.wavelength if f.wavelength is not None else centre(lo, hi)
                if wave_min <= w <= wave_max:
                    record["lines"].append({"x": float(w), "label": f.label, "species": f.species, "cls": cls or "line",
                                            "notes": f.notes})
            else:
                lo, hi = max(lo, wave_min), min(hi, wave_max)
                if hi > lo:
                    record["bands"].append({"lo": float(lo), "hi": float(hi), "label": f.label, "species": f.species,
                                            "cls": cls or ("ice" if f.dataset == "ice_bands" else "emission")})
    record["detected_only"] = bool(detected_only)
    record["series"] = str(series)
    if first and (record["lines"] or record["bands"]):
        _defer(ax.figure, lambda: FeatureLayout(ax).run())


__all__ = ["EXTENDED_DATASET", "FEATURE_NARROW_BAND", "PHOTOSPHERIC_DATASET", "annotate_features"]
