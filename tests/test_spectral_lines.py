"""The search for lines in a spectrum (jwstflow.spectra.lines): which catalogue lines it shows,
which broad gas bands show their forest, which strong peaks no catalogue line explains."""

from __future__ import annotations

import numpy as np
import pytest

from jwstflow.spectra import lines as spectral_lines
from jwstflow.spectra.features import select_features
from jwstflow.spectra.lines import (
    FOREST_MIN_PEAKS,
    FOREST_WINDOW,
    UNIDENTIFIED_IN_BAND,
    UNIDENTIFIED_SIGMA,
    detect_lines,
    identify_lines,
    line_wavelength,
    merge_peaks,
    search_features,
)


def spectrum(lines=(), dips=(), lo=1.7, hi=14.0, n=6000, seed=1):
    """A flat spectrum with 0.4 % noise, Gaussian emission lines ``(centre, amplitude)`` and absorption
    dips of depth 0.3 at the given wavelengths (sigma: 0.06 % of the wavelength, about 1.7 samples)."""
    wave = np.geomspace(lo, hi, n)
    flux = 1.0 + 0.004 * np.random.default_rng(seed).standard_normal(n)
    for centre, amplitude in [*lines, *((c, -0.3) for c in dips)]:
        flux += amplitude * np.exp(-0.5 * ((wave - centre) / (6e-4 * centre)) ** 2)
    return wave, flux


def peaks(x_em=(), s_em=(), *, x_ab=(), s_ab=(), step=0.001, tol=0.0005):
    """A hand-made search result: peaks (dips) at ``x`` with significance ``s``, all with the same
    sample step and tolerance."""
    out = {}
    for key, x, s in (("em", x_em, s_em), ("ab", x_ab, s_ab)):
        x = np.asarray(x, dtype=float)
        out[f"x_{key}"], out[f"s_{key}"] = x, np.asarray(s, dtype=float)
        out[f"tol_{key}"], out[f"step_{key}"] = np.full(x.size, tol), np.full(x.size, step)
    return out


def test_isolated_lines_are_found_absent_ones_not_and_a_strong_stranger_is_unidentified():
    wave, flux = spectrum([(3.0, 3.0), (4.0, 3.0), (4.6, 3.0)], dips=(2.2,), lo=1.7, hi=5.3)
    # catalogue: two of the emission lines, a line the spectrum lacks, the dip as an absorption line
    # and the same wavelength as an emission line (a dip is no emission line)
    found = identify_lines(detect_lines(wave, flux), [3.0, 4.0, 3.5, 2.2, 2.2],
                           absorption=[False, False, False, True, False])
    assert found.found.tolist() == [True, True, False, True, False]
    assert found.strength[2] == 0.0 and found.strength[4] == 0.0
    assert min(found.strength[[0, 1]]) > 100 and found.strength[3] > 10       # amplitude / noise: hundreds of sigma
    x = found.peaks["x_em"]
    assert found.explained.tolist() == ((np.abs(x - 3.0) < 0.01) | (np.abs(x - 4.0) < 0.01)).tolist()
    # the line at 4.6 um is in no catalogue: unidentified; there is no band, so no forest
    assert found.unidentified_x == pytest.approx([4.6], abs=0.002)
    assert found.forest_seen.size == 0 and not found.in_forest.any()


def test_a_forest_is_seen_from_enough_loose_peaks_within_the_forest_window():
    step = 0.001                                          # FOREST_WINDOW samples: 0.3 um
    span = FOREST_WINDOW * step
    comb = 5.1 + np.arange(FOREST_MIN_PEAKS) * 0.9 * span / (FOREST_MIN_PEAKS - 1)     # all within 0.27 um
    sparse = 5.1 + np.arange(FOREST_MIN_PEAKS) * 1.1 * span / (FOREST_MIN_PEAKS - 1)   # spread over 0.33 um

    def seen(x, *, lines=(), gas=True, band=(5.0, 6.0)):
        result = identify_lines(peaks(x, np.full(len(x), 8.0), step=step), lines,
                                band_lo=[band[0]], band_hi=[band[1]], gas_band=[gas])
        return bool(result.forest_seen[0])

    assert seen(comb)
    assert not seen(comb[:-1])                            # one peak short
    assert not seen(sparse)                               # as many, but never all within the window
    assert not seen(comb, gas=False)                      # an ice or PAH band shows no forest
    assert not seen(comb, lines=[comb[2]])                # a peak that is a catalogue line is no forest peak
    assert not seen(comb, band=(6.0, 7.0))                # the peaks lie outside the band
    assert seen(comb, lines=[comb[2] + 0.002])            # ... but a line beside it explains nothing
    # gas_band defaults to every band
    assert identify_lines(peaks(comb, np.full(comb.size, 8.0), step=step), [],
                          band_lo=[5.0], band_hi=[6.0]).forest_seen.tolist() == [True]


def test_inside_a_seen_forest_a_line_counts_only_where_it_stands_above_the_forest():
    comb = list(5.10 + 0.04 * np.arange(6))               # six loose peaks of 8 sigma within 0.2 um
    # emission lines at 5.12 (8 sigma, as strong as the forest), 5.16 (20 sigma), 5.24 (7.9 sigma) and
    # 5.5 (6 sigma; the forest is 0.2 um away, no peak of it within FOREST_WINDOW / 2 samples = 0.15 um);
    # a dip at 5.20
    line_x = [5.12, 5.16, 5.24, 5.5, 5.20]
    absorption = [False, False, False, False, True]
    found = peaks([*comb, 5.12, 5.16, 5.24, 5.5], [8.0] * 6 + [8.0, 20.0, 7.9, 6.0], x_ab=[5.20], s_ab=[6.0])
    in_forest = identify_lines(found, line_x, absorption=absorption, band_lo=[5.0], band_hi=[6.0])
    assert in_forest.forest_seen.tolist() == [True]
    assert in_forest.strength.tolist() == [0.0, 20.0, 0.0, 6.0, 6.0]
    # the same band without a forest (not gas emission): every line counts at its own strength
    no_forest = identify_lines(found, line_x, absorption=absorption, band_lo=[5.0], band_hi=[6.0], gas_band=[False])
    assert no_forest.strength.tolist() == [8.0, 20.0, 7.9, 6.0, 6.0]
    assert in_forest.explained.tolist() == [False] * 6 + [True] * 4
    assert in_forest.in_forest.tolist() == [True] * 10


def test_strong_unexplained_peaks_are_unidentified_inside_a_seen_forest_only_where_they_tower():
    tower = UNIDENTIFIED_SIGMA * UNIDENTIFIED_IN_BAND
    comb = list(5.10 + 0.04 * np.arange(6))
    x = [*comb, 5.32, 5.34, 7.0, 7.2, 7.4]
    s = [8.0] * 6 + [tower, tower - 0.1, UNIDENTIFIED_SIGMA, UNIDENTIFIED_SIGMA - 0.1, 50.0]
    # the catalogue: an emission line at 7.4 um, a photospheric line at 7.0 um (it explains no emission peak)
    found = identify_lines(peaks(x, s), [7.4, 7.0], absorption=[False, True], band_lo=[5.0], band_hi=[6.0])
    assert found.forest_seen.tolist() == [True] and found.strength.tolist() == [50.0, 0.0]
    assert found.in_forest.tolist() == [True] * 8 + [False] * 3
    assert found.explained.tolist() == [False] * 10 + [True]
    assert found.unidentified.tolist() == [False] * 6 + [True, False, True, False, False]
    assert found.unidentified_x.tolist() == [5.32, 7.0]
    # without the forest nothing explains the peak below the tower
    plain = identify_lines(peaks(x, s), [7.4], band_lo=[5.0], band_hi=[6.0], gas_band=[False])
    assert plain.unidentified.tolist() == [False] * 6 + [True, True, True, False, False]


def test_merge_peaks_joins_the_searches_of_several_curves():
    a, b = peaks([1.0], [6.0]), peaks([2.0, 3.0], [7.0, 8.0], x_ab=[2.5], s_ab=[5.0])
    merged = merge_peaks([a, None, b])
    assert merged["x_em"].tolist() == [1.0, 2.0, 3.0] and merged["s_em"].tolist() == [6.0, 7.0, 8.0]
    assert merged["x_ab"].tolist() == [2.5] and set(merged) == set(a)
    assert merge_peaks([None, None]) is None and merge_peaks([]) is None


def test_nothing_searched_shows_nothing():
    found = identify_lines(None, [3.0, 4.0], absorption=[False, True], band_lo=[2.0], band_hi=[5.0])
    assert found.strength.tolist() == [0.0, 0.0] and found.forest_seen.tolist() == [False]
    assert found.peaks is None and found.unidentified.size == found.explained.size == found.in_forest.size == 0
    assert found.unidentified_x.size == 0
    with pytest.raises(ValueError):
        identify_lines(None, [3.0, 4.0], absorption=[False])


def test_narrow_bands_of_the_gas_catalogue_are_lines():
    catalogue = {f.id: f for f in select_features({"gas_lines": "all", "pah_bands": "all", "ice_bands": "all"})}

    def position(name, *centre):
        feature = catalogue[name]
        return line_wavelength(feature, *feature.window(), *centre)

    assert position("neii_12.814") == 12.81355                 # a line
    assert position("c2h2_13.7") == 13.71                      # a Q-branch (0.8 % wide): a line at its wavelength
    assert position("co_fund") is None                         # the CO fundamental: a band
    assert position("pah_11.0") is None                        # as narrow as a Q-branch, but no gas: a band
    assert position("co_4.67") is None                         # an ice band
    # a narrow band without a wavelength: a line in the middle of its window (log axis: geometric)
    assert position("h2o_cold_23.9") == pytest.approx(np.sqrt(23.7 * 24.0))
    assert position("h2o_cold_23.9", 23.85) == 23.85


def _co_forest_spectrum():
    """[Ne II], H2 0-0 S(3), an unknown line at 8.3 um, Na I 2.2062 in absorption, and inside the CO
    fundamental band (4.4-5.2 um) a forest of six unknown lines of about 20 sigma with H I Pf-beta
    (4.654 um) weaker than it and H2 0-0 S(9) (4.695 um) towering over it."""
    forest = [(x, 0.1) for x in (4.45, 4.51, 4.595, 4.755, 4.815, 4.86)]
    return spectrum([(12.81355, 3.0), (9.66491, 3.0), (8.3, 3.0), (4.65379, 0.05), (4.69461, 3.0), *forest],
                    dips=(2.20625,))


def test_search_features_on_the_catalogue():
    wave, flux = _co_forest_spectrum()
    features = select_features({"gas_lines": "all", "photospheric_lines": "all", "pah_bands": "all",
                                "ice_bands": "all"}, wave_min=1.7, wave_max=14.0)
    found = search_features(wave, flux, features)
    assert {f.id for f in found.present} == {"neii_12.814", "h2_0-0_s3", "h2_0-0_s9", "nai_2.2062"}
    assert found.line_x.tolist() == [f.wavelength for f in found.lines]
    # the Q-branches are lines, the forests, PAH and ice bands bands
    assert {"c2h2_13.7", "hcn_14.0"} <= {f.id for f in found.lines}
    assert {"co_fund", "h2o_gas_6um", "pah_6.2", "h2o_3.0"} <= {f.id for f in found.bands}
    assert [f.id for f in found.seen_bands] == ["co_fund"]
    # Pf-beta stands out of the spectrum but not out of the forest around it
    pfb = [f.id for f in found.lines].index("hi_pfb")
    assert np.min(np.abs(found.peaks["x_em"] - found.line_x[pfb])) < 0.002 and found.strength[pfb] == 0.0
    assert found.strength[[f.id for f in found.lines].index("feii_5.340")] == 0.0       # absent
    # the forest peaks are above UNIDENTIFIED_SIGMA, but the seen forest explains them; 8.3 um is unidentified
    forest = found.in_forest & ~found.explained
    assert forest.sum() == 6 and np.all(found.peaks["s_em"][forest] > UNIDENTIFIED_SIGMA)
    assert found.unidentified_x == pytest.approx([8.3], abs=0.005)
    # NaNs are dropped before the search
    holes = flux.copy()
    holes[np.searchsorted(wave, [3.5, 6.0, 7.0, 11.0])] = np.nan
    again = search_features(wave, holes, features)
    assert [f.id for f in again.present] == [f.id for f in found.present] and again.seen_bands == found.seen_bands


def test_a_spectrum_too_short_to_search_shows_nothing():
    wave, flux = _co_forest_spectrum()
    features = select_features({"gas_lines": "all", "ice_bands": "all"}, wave_min=1.7, wave_max=14.0)
    found = search_features(wave[:50], flux[:50], features)
    assert found.peaks is None and found.present == () and found.seen_bands == ()
    assert found.strength.shape == (len(found.lines),) and not found.strength.any()
    assert found.forest_seen.shape == (len(found.bands),) and not found.forest_seen.any()
    assert found.unidentified.size == 0 and found.unidentified_x.size == 0


def test_search_features_decides_like_the_qa_figures():
    """The same spectrum and features: the lines found, the forests seen and the unidentified peaks
    of search_features are those of the annotation of a QA figure."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from jwstflow import qa
    from jwstflow.qa.layout import FeatureLayout

    wave, flux = _co_forest_spectrum()
    lo, hi = 1.7, 14.0
    fig, ax = qa.subplots()
    qa.step(ax, wave, flux, color=qa.MAIN_COLOR)
    qa.set_wave_scale(ax)
    qa.annotate_features(ax, "all", wave_min=lo, wave_max=hi, under=(wave, flux), detected_only=True)
    layout = FeatureLayout(ax)
    layout.detect()
    plt.close(fig)
    # what annotate_features selects for "all": the default datasets, the further gas lines, the photosphere
    features = select_features("all", wave_min=lo, wave_max=hi)
    features += [f for f in select_features({"gas_lines_extended": "all"}, wave_min=lo, wave_max=hi) if f not in features]
    features += select_features({"photospheric_lines": "all"}, wave_min=lo, wave_max=hi)
    found = search_features(wave, flux, features)
    in_view = (found.line_x >= layout.x_lo) & (found.line_x <= layout.x_hi) & (found.line_x >= lo) & (found.line_x <= hi)
    assert sorted((e["x"], e["strength"]) for e in layout.lines) == \
        sorted((float(x), float(s)) for x, s, view in zip(found.line_x, found.strength, in_view) if view and s > 0)
    assert {e["label"] for e in layout.bands if FeatureLayout._gas_band(e)} == {f.label for f in found.seen_bands}
    assert layout.unidentified == [float(x) for x in found.unidentified_x if layout.x_lo <= x <= layout.x_hi]
    assert len(layout.lines) >= 4 and layout.unidentified and found.seen_bands


def test_the_qa_package_exports_the_search_it_was_moved_from():
    from jwstflow import qa

    for name in ("detect_lines", "DETECT_SIGMA", "UNIDENTIFIED_SIGMA", "UNIDENTIFIED_IN_BAND", "IDENTIFY_VELOCITY",
                 "GAP_STEP", "EDGE_SAMPLES", "FOREST_MIN_PEAKS", "FOREST_WINDOW", "FEATURE_NARROW_BAND",
                 "PHOTOSPHERIC_DATASET", "EXTENDED_DATASET"):
        assert getattr(qa, name) is getattr(spectral_lines, name), name
    assert qa.layout.FOREST_WINDOW is spectral_lines.FOREST_WINDOW
    assert qa.annotation.FEATURE_NARROW_BAND is spectral_lines.FEATURE_NARROW_BAND
