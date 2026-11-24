"""The search for spectral lines in a 1-D spectrum, and which catalogue lines it shows.

* :func:`detect_lines` finds the emission peaks and absorption dips that stand out of
  the running median of a spectrum; :func:`merge_peaks` joins the peaks of several curves.
* :func:`identify_lines` decides what they are: which catalogue lines are found, which
  broad gas bands (a forest of lines the catalogue names as one band) the spectrum
  shows, and which strong peaks no catalogue line explains.
* :func:`search_features` does both for :class:`~jwstflow.spectra.features.Feature`
  objects, sorted into lines and bands the way the QA figures sort them
  (:func:`line_wavelength`).

These are the decisions the QA figures label (``jwstflow.qa``: the ``? [wavelength]``
labels, the ``*_detected.png`` figures), made available to data steps. Plain numpy.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .features import Feature

#: The opt-in table of further gas lines (QA figures annotate it together with ``gas_lines``).
EXTENDED_DATASET = "gas_lines_extended"
#: The catalogues of gas emission: their bands are forests of lines or, when narrow, Q-branches and bandheads.
GAS_DATASETS = ("gas_lines", EXTENDED_DATASET)
#: The opt-in catalogue of stellar photospheric absorption lines (:mod:`jwstflow.spectra.features`).
PHOTOSPHERIC_DATASET = "photospheric_lines"
#: Bands of the gas catalogue narrower than this fraction of their wavelength (Q-branches,
#: bandheads) are lines, searched and annotated at their wavelength; broader ones are bands
#: (in QA figures, rectangles in the band rows).
FEATURE_NARROW_BAND = 0.02

#: A feature counts as detected at this many times the local scatter (emission, absorption),
#: and an emission peak this strong without a catalogue line next to it is unidentified
#: (marked "?" in QA figures).
DETECT_SIGMA = (5.0, 4.0)
UNIDENTIFIED_SIGMA = 12.0
#: ... and inside a broad gas band (a forest of lines the band already names) this many times stronger.
UNIDENTIFIED_IN_BAND = 3.0
#: A peak is a catalogue line when it lies within one sample of it or this velocity [km/s],
#: whichever is larger: the half sample by which the highest sample misses the line centre, plus
#: the wavelength calibration and the velocity of the source. Measured on NIRSpec and MIRI
#: spectra of edge-on disks, every line seen lies inside it; further out, the matches are those a
#: catalogue shifted by a percent finds as well -- chance coincidences with other peaks.
IDENTIFY_VELOCITY = 50.0
#: A step between samples this many times wider than the steps beside it is a gap in the spectrum;
#: the first and last this many samples of every stretch between gaps are not searched for lines.
GAP_STEP = 4.0
EDGE_SAMPLES = 3
#: A broad gas band (a forest the catalogue names as a band, not line by line) is seen in the
#: spectrum when at least this many emission peaks that are no catalogue line stand within this
#: many samples of one another inside it. Measured on edge-on disks: a forest of CO, OH or water
#: lines puts 5 to 20 there, noise and stray peaks at most 4. Only a band that is seen explains
#: the peaks inside it (and where QA figures mark only the detected features, only such bands are drawn).
FOREST_MIN_PEAKS = 5
FOREST_WINDOW = 300

#: The keys of a :func:`detect_lines` result.
_PEAK_KEYS = ("x_em", "s_em", "tol_em", "step_em", "x_ab", "s_ab", "tol_ab", "step_ab")


def detect_lines(x: Any, y: Any, *, window: int = 50) -> dict[str, np.ndarray] | None:
    """Emission peaks and absorption dips of a spectrum, for deciding which lines it shows.

    The spectrum is compared with its running median over ``2 * window + 1``
    samples; the local scatter is the running median of the absolute
    residual (x 1.4826). An emission peak is a local maximum at least
    ``DETECT_SIGMA[0]`` scatters above the median whose neighbour stands
    out too (one sample alone is a spike, not a line); an absorption dip
    the same downwards with ``DETECT_SIGMA[1]``. The :data:`EDGE_SAMPLES`
    samples at either end of the spectrum and of every stretch between gaps
    are left out (an edge rises or falls, it is no line). Non-finite samples
    are dropped. Returns the positions
    ``x_em`` / ``x_ab``, their significances ``s_em`` / ``s_ab``, the
    sample step there ``step_em`` / ``step_ab`` and the
    tolerances ``tol_em`` / ``tol_ab`` within which a peak is the line of a
    catalogue (one sample or :data:`IDENTIFY_VELOCITY`, whichever is
    larger; the sample is the step to the nearer neighbour, so a gap in the
    spectrum never widens it); None for a spectrum too short to search.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if x.size < 4 * window:
        return None

    def running(values: np.ndarray) -> np.ndarray:
        return np.median(sliding_window_view(np.pad(values, window, mode="reflect"), 2 * window + 1), axis=1)

    residual = y - running(y)
    scatter = 1.4826 * running(np.abs(residual))
    positive = scatter[scatter > 0]
    if not positive.size:
        return None
    significance = residual / np.maximum(scatter, 0.05 * float(np.median(positive)))
    step = np.abs(np.diff(x))
    sample = np.minimum(np.append(step[:1], step), np.append(step, step[-1:]))     # never across a gap
    tolerance = np.maximum(sample, IDENTIFY_VELOCITY / 299792.458 * np.abs(x))
    # a gap is a step far wider than the steps on either side of it (a change of sampling is not);
    # the samples next to a gap or an end of the spectrum are no lines: they rise and fall with the edge
    around = np.maximum(np.append(step[1:], step[-1]), np.append(step[0], step[:-1]))
    starts = np.concatenate([[0], np.nonzero(step > GAP_STEP * around)[0] + 1, [x.size]])
    index = np.arange(x.size)
    run = np.searchsorted(starts, index, side="right") - 1
    inner = np.minimum(index - starts[run], starts[run + 1] - 1 - index) >= EDGE_SAMPLES
    core, left, right = significance[1:-1], significance[:-2], significance[2:]
    out: dict[str, np.ndarray] = {}
    for key, sign, level in (("em", 1.0, DETECT_SIGMA[0]), ("ab", -1.0, DETECT_SIGMA[1])):
        hit = (sign * core >= level) & (sign * core >= sign * left) & (sign * core >= sign * right) \
            & (np.maximum(sign * left, sign * right) >= 0.4 * level) & inner[1:-1]
        out[f"x_{key}"], out[f"s_{key}"], out[f"tol_{key}"] = x[1:-1][hit], sign * core[hit], tolerance[1:-1][hit]
        out[f"step_{key}"] = sample[1:-1][hit]
    return out


def merge_peaks(found: Iterable[dict[str, np.ndarray] | None]) -> dict[str, np.ndarray] | None:
    """The peaks and dips of several curves (several :func:`detect_lines` results) as one: their
    arrays concatenated in the order given, the None of a curve too short to search skipped. None
    when nothing could be searched."""
    found = [f for f in found if f is not None]
    if not found:
        return None
    return {key: np.concatenate([f[key] for f in found]) for key in found[0]}


@dataclass(frozen=True, eq=False)
class LineIdentification:
    """What :func:`identify_lines` decided, in the order of its inputs.

    ``strength`` (per catalogue line): the significance of the strongest peak (dip) that is the
    line, 0.0 where none is -- or, for an emission line inside a seen forest, where its peak does
    not stand above the forest around it. ``forest_seen`` (per band): the spectrum shows the
    band's forest. Per emission peak (``peaks["x_em"]``): ``explained`` (within its tolerance of
    an emission line of the catalogue), ``in_forest`` (inside a band whose forest is seen) and
    ``unidentified`` (strong, and no catalogue line explains it). ``peaks``: the peaks searched
    (None when the spectrum was too short to search).
    """

    strength: np.ndarray
    forest_seen: np.ndarray
    explained: np.ndarray
    in_forest: np.ndarray
    unidentified: np.ndarray
    peaks: dict[str, np.ndarray] | None

    @property
    def found(self) -> np.ndarray:
        """Per catalogue line: the spectrum shows it (``strength > 0``)."""
        return self.strength > 0

    @property
    def unidentified_x(self) -> np.ndarray:
        """The wavelengths of the unidentified peaks."""
        return self.peaks["x_em"][self.unidentified] if self.peaks is not None else np.empty(0)


def identify_lines(peaks: dict[str, np.ndarray] | None, line_x: Any, *, absorption: Any = None,
                   band_lo: Any = (), band_hi: Any = (), gas_band: Any = None) -> LineIdentification:
    """Which catalogue lines and broad gas bands the ``peaks`` of a spectrum show, and which strong
    peaks no catalogue line explains.

    ``peaks`` is a :func:`detect_lines` result (:func:`merge_peaks` for several curves; None, a
    spectrum too short to search, shows nothing). ``line_x`` are the wavelengths of the catalogue
    lines; ``absorption`` tells per line whether it is an absorption line (photospheric: matched to
    the dips) rather than an emission line (matched to the peaks; the default for all).
    ``band_lo`` / ``band_hi`` are the edges of the broad bands, ``gas_band`` tells per band whether
    it is a band of gas emission (default: all) -- only those can show a forest.

    * A catalogue line is **found** when a peak (a dip) stands within its tolerance of it
      (:func:`detect_lines`; one sample or :data:`IDENTIFY_VELOCITY`); its strength is the
      significance of the strongest such peak. A Q-branch too is found at its wavelength only.
    * The peaks no emission line of the catalogue explains are **loose**. A gas band's **forest
      is seen** when at least :data:`FOREST_MIN_PEAKS` loose peaks inside it stand within
      :data:`FOREST_WINDOW` samples of one another.
    * Inside a seen forest an emission line counts as found only where its peak stands above
      the forest around it: its strength is set to 0 unless it exceeds the strongest loose peak
      within ``FOREST_WINDOW / 2`` samples of it (a forest peak that happens to fall on a
      catalogue line is no detection of that line).
    * A loose peak is **unidentified** from :data:`UNIDENTIFIED_SIGMA` on, inside a seen forest
      (which explains the ordinary peaks in it) from :data:`UNIDENTIFIED_IN_BAND` times that.
    """
    searched = peaks if peaks is not None else {key: np.empty(0) for key in _PEAK_KEYS}
    line_x = np.asarray(line_x, dtype=float).reshape(-1)
    absorption = np.zeros(line_x.size, dtype=bool) if absorption is None \
        else np.asarray(absorption, dtype=bool).reshape(-1)
    band_lo = np.asarray(band_lo, dtype=float).reshape(-1)
    band_hi = np.asarray(band_hi, dtype=float).reshape(-1)
    gas_band = np.ones(band_lo.size, dtype=bool) if gas_band is None else np.asarray(gas_band, dtype=bool).reshape(-1)
    if absorption.size != line_x.size or band_hi.size != band_lo.size or gas_band.size != band_lo.size:
        raise ValueError(f"identify_lines: {line_x.size} lines with {absorption.size} absorption flags, "
                         f"{band_lo.size} / {band_hi.size} band edges with {gas_band.size} gas flags")
    x_em, s_em, step_em = searched["x_em"], searched["s_em"], searched["step_em"]

    # the peaks (dips) that are a line: those within their tolerance of it. A Q-branch too is found
    # at its wavelength only -- its catalogue window is a band's width and would take the peaks of
    # any forest around it
    strength = np.zeros(line_x.size)
    explained = np.zeros(x_em.size, dtype=bool)
    for k, (x, dip) in enumerate(zip(line_x, absorption)):
        key = "ab" if dip else "em"
        mine = np.abs(searched[f"x_{key}"] - x) <= searched[f"tol_{key}"]
        if mine.any():
            strength[k] = float(searched[f"s_{key}"][mine].max())
        if not dip:
            explained |= mine
    loose, loose_s, step = x_em[~explained], s_em[~explained], step_em[~explained]

    forest_seen = np.zeros(band_lo.size, dtype=bool)
    for b, (lo, hi, gas) in enumerate(zip(band_lo, band_hi, gas_band)):
        if gas:
            inside = (loose >= lo) & (loose <= hi)
            xs, reach = loose[inside], FOREST_WINDOW * step[inside]
            forest_seen[b] = any(int(np.sum((xs >= a) & (xs <= a + r))) >= FOREST_MIN_PEAKS for a, r in zip(xs, reach))
    covered = list(zip(band_lo[forest_seen], band_hi[forest_seen]))

    # inside a forest a catalogue line is found only where it stands above the forest around it
    for k, (x, dip) in enumerate(zip(line_x, absorption)):
        if not dip and strength[k] > 0 and any(a <= x <= b for a, b in covered):
            near = np.abs(loose - x) <= 0.5 * FOREST_WINDOW * step
            if near.any() and strength[k] <= float(loose_s[near].max()):
                strength[k] = 0.0
    in_forest = np.zeros(x_em.size, dtype=bool)
    for a, b in covered:
        in_forest |= (x_em >= a) & (x_em <= b)
    # inside a broad gas band its forest explains ordinary peaks; one that towers over it is unidentified too
    need = UNIDENTIFIED_SIGMA * np.where(in_forest, UNIDENTIFIED_IN_BAND, 1.0)
    unidentified = ~explained & (s_em >= need)
    return LineIdentification(strength=strength, forest_seen=forest_seen, explained=explained,
                              in_forest=in_forest, unidentified=unidentified, peaks=peaks)


def line_wavelength(feature: Feature, lo: float, hi: float, centre: float | None = None) -> float | None:
    """Where ``feature`` is searched and marked as a line; None for a band.

    A line (``kind == "line"``) is a line at its wavelength; so is a band of a gas catalogue
    (:data:`GAS_DATASETS`) no wider than :data:`FEATURE_NARROW_BAND` of its centre -- a
    Q-branch, a bandhead -- at its wavelength, or at ``centre`` when it has none. ``lo`` - ``hi``
    is the feature's window (:meth:`~jwstflow.spectra.features.Feature.window`), ``centre`` the
    middle of it: by default the geometric mean (the middle on the logarithmic wavelength axis of
    the QA figures), their arithmetic mean on a linear axis. Every other band is a band.
    """
    if centre is None:
        centre = float(np.sqrt(lo * hi)) if lo > 0 else 0.5 * (lo + hi)
    narrow = feature.kind != "line" and feature.dataset in GAS_DATASETS and (hi - lo) <= FEATURE_NARROW_BAND * centre
    if feature.kind == "line" or narrow:
        return feature.wavelength if feature.wavelength is not None else centre
    return None


def _gas_band(feature: Feature) -> bool:
    """A band of gas emission (a forest named as one band), as the QA figures tell it: not an ice
    band, not one of the photosphere, no PAH."""
    return feature.dataset not in ("ice_bands", PHOTOSPHERIC_DATASET) and feature.species != "PAH"


@dataclass(frozen=True, eq=False)
class LineSearch:
    """What :func:`search_features` found: the line features with their wavelengths ``line_x`` and
    ``strength`` (0.0: not shown), the band features with ``forest_seen``, and per emission peak of
    ``peaks`` (None: the spectrum was too short to search) ``explained``, ``in_forest`` and
    ``unidentified`` -- as :class:`LineIdentification`."""

    lines: tuple[Feature, ...]
    line_x: np.ndarray
    strength: np.ndarray
    bands: tuple[Feature, ...]
    forest_seen: np.ndarray
    peaks: dict[str, np.ndarray] | None
    explained: np.ndarray
    in_forest: np.ndarray
    unidentified: np.ndarray

    @property
    def present(self) -> tuple[Feature, ...]:
        """The line features the spectrum shows (``strength > 0``)."""
        return tuple(f for f, s in zip(self.lines, self.strength) if s > 0)

    @property
    def seen_bands(self) -> tuple[Feature, ...]:
        """The gas bands whose forest the spectrum shows."""
        return tuple(f for f, seen in zip(self.bands, self.forest_seen) if seen)

    @property
    def unidentified_x(self) -> np.ndarray:
        """The wavelengths of the unidentified peaks."""
        return self.peaks["x_em"][self.unidentified] if self.peaks is not None else np.empty(0)


def search_features(wavelength: Any, flux: Any, features: Iterable[Feature], *, window: int = 50) -> LineSearch:
    """Search a spectrum for the given spectral ``features`` (from
    :func:`~jwstflow.spectra.features.select_features`): which lines it shows, which gas bands show
    their forest, which strong peaks no line explains -- the decisions of the QA figures.

    The features are sorted as the QA figures sort them: lines and the narrow bands of the gas
    catalogues (:func:`line_wavelength`, windows at their defaults) are lines -- those of the
    ``photospheric_lines`` dataset absorption lines, the others emission lines -- and the other
    bands are bands, of which only those of gas emission (not ice, PAH or the photosphere) can
    show a forest. Unlike the QA annotation nothing is added to the selection: pass the
    ``gas_lines_extended`` and ``photospheric_lines`` features to search for them. Non-finite
    samples are left out (:func:`detect_lines`, ``window`` its running median); a spectrum too
    short to search shows nothing (``peaks`` None, every strength 0, no forest).
    """
    lines: list[Feature] = []
    line_x: list[float] = []
    bands: list[Feature] = []
    edges: list[tuple[float, float]] = []
    for f in features:
        lo, hi = f.window()
        x = line_wavelength(f, lo, hi)
        if x is not None:
            lines.append(f)
            line_x.append(float(x))
        else:
            bands.append(f)
            edges.append((lo, hi))
    found = identify_lines(detect_lines(wavelength, flux, window=window), line_x,
                           absorption=[f.dataset == PHOTOSPHERIC_DATASET for f in lines],
                           band_lo=[e[0] for e in edges], band_hi=[e[1] for e in edges],
                           gas_band=[_gas_band(f) for f in bands])
    return LineSearch(lines=tuple(lines), line_x=np.asarray(line_x, dtype=float), strength=found.strength,
                      bands=tuple(bands), forest_seen=found.forest_seen, peaks=found.peaks,
                      explained=found.explained, in_forest=found.in_forest, unidentified=found.unidentified)


__all__ = [
    "DETECT_SIGMA",
    "EDGE_SAMPLES",
    "EXTENDED_DATASET",
    "FEATURE_NARROW_BAND",
    "FOREST_MIN_PEAKS",
    "FOREST_WINDOW",
    "GAP_STEP",
    "GAS_DATASETS",
    "IDENTIFY_VELOCITY",
    "PHOTOSPHERIC_DATASET",
    "UNIDENTIFIED_IN_BAND",
    "UNIDENTIFIED_SIGMA",
    "LineIdentification",
    "LineSearch",
    "detect_lines",
    "identify_lines",
    "line_wavelength",
    "merge_peaks",
    "search_features",
]
