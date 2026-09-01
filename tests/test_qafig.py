"""Tests of the QA figure standard's robust image scaling (jwstflow.qafig)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from jwstflow import qafig
from jwstflow.qafig import _robust_max


def field(rng, hot: bool = True) -> np.ndarray:
    """A ch4-like collapse: faint blob over a bright noisy background, NaN
    corners, and (optionally) isolated hot pixels."""
    yy, xx = np.mgrid[:32, :33]
    img = 2.8 + 0.05 * rng.standard_normal((32, 33))
    img += 2.3 * np.exp(-((xx - 16.0) ** 2 + (yy - 15.0) ** 2) / (2 * 1.0**2))
    img[(xx + yy < 10) | (xx + yy > 54)] = np.nan
    if hot:
        img[25, 4] = 40.0
        img[3, 27] = 25.0
    return img


def test_robust_max_ignores_isolated_hot_pixels():
    rng = np.random.default_rng(2)
    img = field(rng)
    assert np.nanmax(img) == 40.0
    cap = _robust_max(img)
    assert cap is not None and cap < 6.0        # the blob's neighbourhood, not the hot pixel
    assert cap > 3.5                            # ... but the blob still stands out of the background


def test_robust_max_keeps_extended_bright_sources():
    img = np.ones((20, 20))
    img[8:13, 8:13] = 50.0                      # a real, several-pixel source
    assert _robust_max(img) == 50.0
    assert _robust_max(np.full((2, 2), np.nan)) is None    # unusable input
    assert _robust_max(np.ones(9)) is None                 # not an image


def test_imshow_log_limits_span_the_real_structure(tmp_path):
    # the ch4 aperture-QA regression: percentiles hit the hot pixels and the
    # old one-decade floor forced vmax to 10 x vmin -- background-dominated
    # images came out black with the source invisible
    rng = np.random.default_rng(3)
    fig, ax = qafig.subplots()
    im = qafig.imshow(ax, field(rng), stretch="log", percentiles=(5.0, 99.9))
    assert im.norm.vmax < 6.0                   # capped at the blob, not the hot pixel
    assert im.norm.vmax > 3.5                   # ... and not squashed below the blob
    assert 2.0 < im.norm.vmin < 3.0             # the background sits at the bottom
    qafig.save(fig, tmp_path / "log.png")


def test_imshow_constant_image_keeps_a_well_formed_norm(tmp_path):
    fig, ax = qafig.subplots()
    im = qafig.imshow(ax, np.full((10, 10), 5.0), stretch="log")
    assert im.norm.vmax == pytest.approx(10 * im.norm.vmin)   # degenerate fallback
    fig2, ax2 = qafig.subplots()
    im2 = qafig.imshow(ax2, np.full((10, 10), 5.0))           # linear
    assert im2.norm.vmax > im2.norm.vmin
    qafig.save(fig, tmp_path / "const_log.png")
    qafig.save(fig2, tmp_path / "const_lin.png")


def test_imshow_linear_limits_are_capped_too(tmp_path):
    img = np.zeros((30, 30))
    img[14:17, 14:17] = 10.0
    img[2, 2] = 1e4
    fig, ax = qafig.subplots()
    im = qafig.imshow(ax, img, percentiles=(0.0, 100.0))
    assert im.norm.vmax == pytest.approx(10.0)  # the source, not the hot pixel
    qafig.save(fig, tmp_path / "lin.png")


def test_imshow_explicit_norm_wins(tmp_path):
    from matplotlib.colors import LogNorm

    fig, ax = qafig.subplots()
    im = qafig.imshow(ax, field(np.random.default_rng(4)), stretch="log",
                      norm=LogNorm(vmin=1.0, vmax=100.0))
    assert im.norm.vmin == 1.0 and im.norm.vmax == 100.0
    qafig.save(fig, tmp_path / "norm.png")


def test_collapse_min_coverage_blanks_edge_spaxels():
    cube = np.ones((20, 8, 8))
    cube[:, 0, 0] = np.nan
    cube[1:, 0, 1] = np.nan      # 1/20 planes finite -> nanmedian of noise
    cube[:10, 0, 2] = np.nan     # exactly half covered
    out = qafig.collapse(cube, min_coverage=0.5)
    assert np.isnan(out[0, 0]) and np.isnan(out[0, 1])
    assert out[0, 2] == 1.0 and out[5, 5] == 1.0
    full = qafig.collapse(cube)  # default: nothing blanked beyond all-NaN
    assert full[0, 1] == 1.0


def test_annotate_features_draws_the_three_lanes(tmp_path):
    fig, ax = qafig.subplots()
    ax.plot([3.0, 12.0], [1.0, 1.0])
    ax.set_xscale("log")
    qafig.annotate_features(ax, "all", wave_min=3.0, wave_max=12.0)
    _, labels = ax.get_legend_handles_labels()
    assert {"gas lines", "emission bands", "ice bands"} <= set(labels)
    assert len(ax.texts) > 10        # rotated line/band labels
    assert len(ax.collections) > 2   # the shaded band lanes
    qafig.save(fig, tmp_path / "annot.png")


def test_annotate_features_respects_the_range_and_none(tmp_path):
    fig, ax = qafig.subplots()
    ax.plot([0.5, 0.9], [1.0, 1.0])
    qafig.annotate_features(ax, "all", wave_min=0.5, wave_max=0.9)  # nothing lives here
    assert not ax.texts
    qafig.annotate_features(ax, None, wave_min=1.0, wave_max=30.0)  # no selection: no-op
    assert not ax.texts
    qafig.save(fig, tmp_path / "annot_empty.png")
