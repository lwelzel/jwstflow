"""The ``stitch_cubes`` step: the band cubes of one instrument joined along wavelength.

The method and the product are described in :mod:`jwstflow.cubes.stitching`; the figures of a
joint cube, ``qa_cube_stitch``, are in :mod:`jwstflow.qa.steps.cube_stitch`.
"""

from __future__ import annotations

import logging
import os
import warnings
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

from ...spectra.features import feature_windows
from ...spectra.overlaps import (
    OverlapRatio,
    chain_scales,
    mask_windows,
    measure_overlap_ratio,
    overlap_midpoints,
)
from ...steps import RunContext, Step, StepParams
from ...steps.naming import derived_name
from .grid import data_bounds, joint_grid, reference_index
from .measure import (
    collapse,
    footprint,
    levels,
    pair_shift,
    region_spectrum,
    smooth3,
    source_region,
)
from .product import JOINT_SUFFIX, write_joint_cube
from .regrid import IDENTITY, Placement
from .segments import CubeSegment, read_segment

log = logging.getLogger(__name__)

#: How far beyond an overlap the region spectra of a pair are recorded, in overlap widths (for QA).
OVERLAP_PAD = 0.75
#: Wavelengths closer than this (relative) to a crossover count as lying on it. The wavelength table
#: of a cube is float32 (6e-8), and two segments rarely name one wavelength to the last bit.
CROSSOVER_TOLERANCE = 1e-6


class StitchCubes(Step):
    """Join the band cubes of one instrument (NIRSpec gratings, MRS bands) into one cube along wavelength.
    In development: the joint cubes are not science-ready products.

    **Status: in development.** This step and its product are still being
    worked on. A joint cube is not a science-ready product, and no reduction
    recipe builds one; for science use the band cubes (``*_s3d.fits``) and
    the stitched spectra.

    One task receives every cube (``batch = "all"``); the cubes are grouped
    by ``group_by`` (default: the instrument) and each group becomes one
    ``*_s3dcomb.fits`` -- an IFU cube like its inputs, with a tabular
    wavelength axis, plus the records of the stitch:

    1. **One spatial grid.** The joint grid has the finest spaxel scale among
       the segments and covers the union of their fields; it extends the
       pixel lattice of the finest segment, so that segment is never
       resampled. A segment that already lies on the joint lattice -- every
       cube from ``cube_build_matched`` does -- is copied spaxel for spaxel.
       Any other segment is resampled once, with the overlap areas between
       its spaxels and the joint spaxels as weights: surface brightness is
       averaged by area, flux is conserved, nothing is interpolated between
       spaxels. That is exact but costs resolution (a spaxel is spread over
       those it straddles, coarser spaxels remain visible as blocks); the
       SEGMENTS table says for every segment how it got onto the grid, and
       the step warns when it had to resample. A joint spaxel is filled when
       valid input covers at least ``min_coverage`` of its area.
    2. **No spectral resampling.** Segments are ordered blue to red and the
       cube switches from one to the next at the ``crossovers`` (default: the
       midpoint of each overlap, or of the gap between segments that do not
       overlap). Every plane of the joint cube is one plane of one segment,
       at that segment's own wavelength; the wavelength axis is the table of
       those wavelengths.
    3. **Overlap ratios, as for spectra.** For each neighbouring pair the
       flux of both segments is measured over one region of sky and the
       ratio of the two region spectra is taken like ``stitch_segments``
       takes it: in the wavelength window of ``ratio_window_frac`` times the
       crossover wavelength around the crossover, as the ratio of the
       windowed medians with its propagated uncertainty, counted only when
       both stand ``min_overlap_snr`` sigmas above zero, with the selected
       spectral ``features`` masked out. The region (``ratio_region:
       source``) is the part of the field both segments cover completely
       where the source stands out -- above ``source_fraction`` of the peak
       excess over the field level, whole sources only: a bright piece that
       reaches the rim of the field is cut by it, each band holds another
       part of it, and it is left out -- and the flux is the source's *excess
       over the level of the field*, taken off plane by plane as the median
       of the common field: the cube counterpart of an on/off extraction.
       Two bands rarely agree in that level (the sky, or what a sky
       subtraction left behind: an additive difference that no factor
       removes), and a ratio of plain sums would take it for a calibration
       difference; the ratio of the excesses is the factor between the two
       bands, whatever their levels. ``ratio_region: footprint`` sums the
       whole common field as it is, for fields the source fills. Ratios,
       uncertainties, windows, regions and the field level of both segments
       (``level_a``, ``level_b``) are always recorded; with ``rescale``
       every segment is multiplied by one number -- the chain of ratios onto
       the ``reference`` segment (default: the reddest) -- and the scale
       uncertainty is folded into ERR. One factor per segment leaves the
       spatial distribution of each band untouched. Such a factor is well
       defined for a bright, compact source: it is then the factor of a
       stitched aperture spectrum of the same data, to about a percent. For
       a faint or extended source the ratio depends on where it is measured
       (apertures of one data set give different factors); there
       ``scales_from`` adopts the scales of one stitched spectrum instead
       (the ``*_s1dcomb.ecsv`` of that stage), so cube and spectrum carry
       the same factors; the cube's own ratios are still recorded.
    4. **Positions are checked, not adjusted.** Each segment is placed by its
       own WCS. For every pair the two segments are registered against each
       other at the same wavelengths (the planes of the ratio window): their
       collapsed images, over one window around the bright core they share,
       fitted for a shift with the scale and the level between them left
       free. The shift, with an uncertainty from the two halves of the
       planes, is recorded in the OVERLAPS table and drawn by
       ``qa_cube_stitch`` -- it is the disagreement between the two cubes
       about where the source is (their astrometry, and across MRS channels
       also their different sampling of the PSF), which no resampling
       removes.

    The segments must follow one another in wavelength (each reaching further
    red than the one before; one lying inside another's range is refused),
    and a crossover lies inside the overlap of its pair, so the joint cube
    leaves out no wavelength a segment holds. In ERR the uncertainty of a
    scale is common to every spaxel of its segment -- not independent from
    spaxel to spaxel like the rest; SEGMENTS records it (``scale_error``) so
    it can be told apart.
    """

    name = "stitch_cubes"
    level = 4
    batch = "all"
    inputs = ("*_s3d.fits",)
    outputs = (JOINT_SUFFIX,)
    version = "1"

    class Params(StepParams):
        crossovers: list[float] | None = Field(None, description="wavelengths [um] where the cube switches segments "
                                               "(one fewer than segments); default: overlap midpoints")
        rescale: bool = Field(False, description="multiply every segment by its chained overlap ratio onto the reference")
        reference: str | None = Field(None, description="segment kept fixed when rescaling: a label ('g395h-f290lp', "
                                      "'1-short'), 'shortest' or 'longest' (default: the reddest segment)")
        scales_from: str | None = Field(None, description="adopt the scales of a stitched spectrum instead of the cube's "
                                        "own ratios: the stage holding its *_s1dcomb.ecsv (segments matched by label)")
        scales_pattern: str = Field("*_s1dcomb.ecsv", description="scales_from: which stitched spectrum of that stage "
                                    "(must match exactly one file)")
        ratio_window_frac: float = Field(0.05, gt=0, description="full width of the wavelength window around each "
                                         "crossover in which the ratio is measured, as a fraction of the crossover "
                                         "wavelength (clipped to the overlap)")
        min_overlap_points: int = Field(5, ge=1, description="planes needed in the ratio window to measure a ratio")
        min_overlap_snr: float = Field(3.0, ge=0, description="both windowed medians must stand this many sigmas above "
                                       "zero for the ratio to count; 0 disables the guard")
        features: Any = Field(None, description="jwstflow feature selection masked out of the ratio measurement "
                              "(e.g. {gas_lines: all, pah_bands: all})")
        line_width_um: float = Field(0.02, gt=0, description="half-width [um] masked around narrow lines of the features")
        ratio_region: Literal["source", "footprint"] = Field("source", description="what the ratio compares: the excess "
                                                             "over the field level where the source stands out, or "
                                                             "the plain sum over the whole common field")
        source_fraction: float = Field(0.1, gt=0, lt=1, description="ratio_region source: spaxels above this fraction of "
                                       "the peak excess over the field level")
        min_coverage: float = Field(0.5, gt=0, le=1, description="resampled segments: fraction of a joint spaxel that "
                                    "valid input must cover for the spaxel to be filled")
        group_by: list[str] = Field(default_factory=lambda: ["INSTRUME"], description="header keywords; cubes agreeing "
                                    "in all of them are joined into one cube")

    def run(self, inputs: list[Path], ctx: RunContext, *, crossovers: list[float] | None = None, rescale: bool = False,
            reference: str | None = None, scales_from: str | None = None, scales_pattern: str = "*_s1dcomb.ecsv",
            ratio_window_frac: float = 0.05, min_overlap_points: int = 5,
            min_overlap_snr: float = 3.0, features: Any = None, line_width_um: float = 0.02,
            ratio_region: str = "source", source_fraction: float = 0.1, min_coverage: float = 0.5,
            group_by: list[str] | None = None, **params: Any) -> list[Path]:
        from astropy.io import fits

        if rescale and scales_from:
            raise ValueError("rescale (the cube's own overlap ratios) and scales_from (the scales of a stitched spectrum) "
                             "are two ways to scale the segments; give one")
        adopted = _adopted_scales(ctx, scales_from, scales_pattern) if scales_from else None
        keys = ["INSTRUME"] if group_by is None else list(group_by)
        groups: dict[tuple[str, ...], list[Path]] = {}
        for path in sorted(Path(p) for p in inputs):
            header = fits.getheader(path)
            groups.setdefault(tuple(str(header.get(k, "")).strip().upper() for k in keys), []).append(path)
        if crossovers is not None and len(groups) > 1:
            raise ValueError(f"crossovers were given, but the inputs form {len(groups)} groups ({sorted(groups)}); "
                             "stitch them in separate stages")
        windows = feature_windows(features, line_width_um=line_width_um) if features not in (None, False) else []
        outputs: list[Path] = []
        for key, files in sorted(groups.items()):
            if len(files) < 2:
                log.warning("group %s has only one cube (%s); nothing to stitch", "/".join(key) or "(all)", files[0].name)
                continue
            outputs.append(self._stitch(files, ctx, crossovers=crossovers, rescale=rescale, reference=reference,
                                        adopted=adopted, scales_stage=scales_from,
                                        ratio_window_frac=ratio_window_frac, min_overlap_points=min_overlap_points,
                                        min_overlap_snr=min_overlap_snr, windows=windows, ratio_region=ratio_region,
                                        source_fraction=source_fraction, min_coverage=min_coverage))
        if not outputs:
            raise ValueError(f"no group with at least two cubes among {len(inputs)} input(s) (group_by={keys})")
        return outputs

    # ------------------------------------------------------------------ one joint cube
    def _stitch(self, files: list[Path], ctx: RunContext, *, crossovers: list[float] | None, rescale: bool,
                reference: str | None, adopted: dict[str, tuple[float, float]] | None, scales_stage: str | None,
                ratio_window_frac: float, min_overlap_points: int, min_overlap_snr: float,
                windows: list[tuple[float, float]], ratio_region: str, source_fraction: float,
                min_coverage: float) -> Path:
        segments = sorted((read_segment(p) for p in files), key=lambda s: s.wave_range[0])
        labels = [s.label for s in segments]
        if len(set(labels)) != len(labels):
            raise ValueError(f"segment labels are not unique: {labels}")
        units = {(s.unit or "").strip() for s in segments}
        if len(units) > 1:
            raise ValueError(f"the cubes carry different units {sorted(units)}; they cannot be joined")
        ranges = [s.wave_range for s in segments]
        # a chain: every segment reaches further red than the one before it. A segment that lies inside
        # another's range (a medium grating next to a high-resolution one, a prism) has no place in it --
        # switching to it and back would leave planes out that the wider segment holds
        for (label_a, range_a), (label_b, range_b) in pairwise(zip(labels, ranges)):
            if range_b[1] <= range_a[1]:
                raise ValueError(f"segment {label_b} ({range_b[0]:.4f}-{range_b[1]:.4f} um) lies inside the range of "
                                 f"{label_a} ({range_a[0]:.4f}-{range_a[1]:.4f} um): the segments of a joint cube follow "
                                 "one another in wavelength. Join them in separate groups (group_by) or stages.")
        if crossovers is None:
            bounds = overlap_midpoints(ranges)
        else:
            if len(crossovers) != len(segments) - 1:
                raise ValueError(f"{len(segments)} segments need {len(segments) - 1} crossovers, got {len(crossovers)}")
            bounds = [float(c) for c in crossovers]
        if any(b <= a for a, b in pairwise(bounds)):
            raise ValueError(f"crossovers must increase: {bounds}")
        for bound, (label_a, range_a), (label_b, range_b) in zip(bounds, zip(labels, ranges), zip(labels[1:], ranges[1:])):
            # inside the overlap (or the gap): beyond it one segment would be cut where the other has no planes
            lo, hi = sorted((range_b[0], range_a[1]))
            if not lo - 1e-9 <= bound <= hi + 1e-9:
                raise ValueError(f"crossover {bound:g} um lies outside the {'overlap' if range_a[1] > range_b[0] else 'gap'} "
                                 f"of {label_a} and {label_b} ({lo:.4f}-{hi:.4f} um): planes one of them holds there "
                                 "would be left out of the joint cube")

        # 1. one spatial grid; every segment onto it
        grid = joint_grid([s.grid for s in segments], [data_bounds(s.valid) for s in segments])
        finest = reference_index([s.grid for s in segments])
        log.info("joint grid: %s (finest segment: %s)", grid.describe(), labels[finest])
        placements: list[Placement] = []
        cubes: list[dict[str, np.ndarray]] = []
        for seg in segments:
            placement = Placement.plan(seg.grid, grid, pixels=seg.valid.any(axis=0))
            placements.append(placement)
            cubes.append(placement.cube(seg.data, seg.err, seg.dq, seg.wmap, min_coverage=min_coverage))
            if placement.mode != IDENTITY:
                log.warning("%s (%.3f\" spaxels, lattice offset %+.2f, %+.2f joint spaxels) is not on the joint grid: "
                            "resampled by exact pixel overlap -- flux is conserved, resolution is not (each of its "
                            "spaxels is shared between %.1f joint spaxels). Build the cubes on the joint grid with "
                            "cube_build_matched to avoid this.", seg.label, seg.grid.scale, *placement.phase,
                            placement.correlation)
            seg.data = seg.err = seg.dq = seg.wmap = None          # type: ignore[assignment]  (the copy on the grid is used from here)
        footprints = [footprint(np.isfinite(c["data"])) for c in cubes]
        area_sr = grid.pixel_area_sr

        # 2. the neighbour pairs: region, spectra, ratio, positions
        ratios: list[OverlapRatio] = []
        overlaps: list[dict[str, Any]] = []
        overlap_spectra: list[dict[str, Any]] = []
        ny, nx = grid.shape
        images = np.full((2 * (len(segments) - 1), ny, nx), np.nan, dtype=np.float32)
        regions = np.zeros((len(segments) - 1, ny, nx), dtype=bool)
        cores = np.zeros((2 * (len(segments) - 1), ny, nx), dtype=bool)
        for i, crossover in enumerate(bounds):
            a, b = segments[i], segments[i + 1]
            lo, hi = ranges[i + 1][0], ranges[i][1]
            row: dict[str, Any] = {"segment_a": a.label, "segment_b": b.label, "crossover": float(crossover),
                                   "overlap_min": float(lo), "overlap_max": float(hi), "window_min": np.nan,
                                   "window_max": np.nan, "ratio": np.nan, "ratio_error": np.nan, "n_samples": 0,
                                   "n_spaxels": 0, "region": ratio_region, "dra": np.nan, "ddec": np.nan,
                                   "dra_error": np.nan, "ddec_error": np.nan, "x": np.nan, "y": np.nan,
                                   "n_window": 0, "level_a": np.nan, "level_b": np.nan}
            overlaps.append(row)
            if hi <= lo:
                log.warning("no overlap between %s and %s (%.4f-%.4f um); ratio and positions not measured",
                            a.label, b.label, hi, lo)
                ratios.append(OverlapRatio.unmeasurable())
                continue
            half = 0.5 * ratio_window_frac * float(crossover)
            window = (max(lo, float(crossover) - half), min(hi, float(crossover) + half))
            in_a, in_b = (a.wave >= window[0]) & (a.wave <= window[1]), (b.wave >= window[0]) & (b.wave <= window[1])
            if in_a.sum() < min_overlap_points or in_b.sum() < min_overlap_points:   # as the ratio does: the whole overlap
                in_a, in_b = (a.wave >= lo) & (a.wave <= hi), (b.wave >= lo) & (b.wave <= hi)
            planes_a, planes_b = cubes[i]["data"][in_a], cubes[i + 1]["data"][in_b]
            image_a, image_b = collapse(planes_a), collapse(planes_b)
            images[2 * i], images[2 * i + 1] = image_a, image_b
            common = footprints[i] & footprints[i + 1]
            field = common & np.isfinite(image_a) & np.isfinite(image_b)      # what both segments cover completely
            region = source_region(image_a, image_b, common, source_fraction) if ratio_region == "source" else field
            regions[i] = region
            row["n_spaxels"] = int(region.sum())
            if not region.any():
                log.warning("%s and %s share no completely covered spaxel; ratio and positions not measured",
                            a.label, b.label)
                ratios.append(OverlapRatio.unmeasurable())
                continue

            pad = OVERLAP_PAD * (hi - lo)
            pair: list[dict[str, Any]] = []
            for k, seg in ((i, a), (i + 1, b)):
                near = (seg.wave >= max(lo - pad, ranges[k][0])) & (seg.wave <= min(hi + pad, ranges[k][1]))
                errors = cubes[k]["err"][near] if "err" in cubes[k] else None
                flux, error = region_spectrum(cubes[k]["data"][near], errors, region, pixel_area_sr=area_sr,
                                              correlation=placements[k].correlation,
                                              field=field if ratio_region == "source" else None)
                ok = np.isfinite(flux)
                pair.append({"label": seg.label, "WAVELENGTH": seg.wave[near][ok], "FLUX": flux[ok],
                             "FLUX_ERROR": np.where(np.isfinite(error[ok]), error[ok], np.nan)})
                # every plane is recorded, NaN where the region is incomplete: a figure breaks its line there
                overlap_spectra.append({"index": np.full(len(flux), i), "segment": np.full(len(flux), seg.label),
                                        "wavelength": seg.wave[near], "flux": flux, "error": error})
            if min(len(pair[0]["FLUX"]), len(pair[1]["FLUX"])) == 0:
                log.warning("%s / %s: no plane covers the whole measurement region; ratio not measured", a.label, b.label)
                ratios.append(OverlapRatio.unmeasurable())
            else:
                first = dict(pair[0])
                first["FLUX"] = mask_windows(first["WAVELENGTH"], first["FLUX"], windows, within=(lo, hi))
                ratios.append(measure_overlap_ratio(first, pair[1], crossover=float(crossover),
                                                    ratio_window_frac=ratio_window_frac,
                                                    min_overlap_points=min_overlap_points, min_overlap_snr=min_overlap_snr))
            ratio = ratios[-1]
            row.update(ratio=float(ratio.ratio), ratio_error=float(ratio.error), n_samples=int(ratio.n),
                       window_min=float(ratio.window[0]) if ratio.window else np.nan,
                       window_max=float(ratio.window[1]) if ratio.window else np.nan)
            # the level of the field in both segments: a difference here is additive, no factor removes it
            row["level_a"], row["level_b"] = levels(smooth3(image_a, field))[0], levels(smooth3(image_b, field))[0]
            # where both put the source: one registration of the two images (the planes of the window in
            # wavelength the images were collapsed over, which is the ratio's unless that had to be widened)
            shift = pair_shift(planes_a, planes_b, field, within=region)
            cores[2 * i], cores[2 * i + 1] = shift["core_a"], shift["core_b"]
            # pixel shifts -> tangent-plane offsets (east, north) with the pixel matrix of the grid
            (cd11, cd12), (cd21, cd22) = grid.cd
            row.update(x=shift["x"], y=shift["y"], n_window=int(shift["window"].sum()),
                       dra=(cd11 * shift["dx"] + cd12 * shift["dy"]) * 3600.0,
                       ddec=(cd21 * shift["dx"] + cd22 * shift["dy"]) * 3600.0,
                       dra_error=float(np.hypot(cd11 * shift["ex"], cd12 * shift["ey"])) * 3600.0,
                       ddec_error=float(np.hypot(cd21 * shift["ex"], cd22 * shift["ey"])) * 3600.0)
            log.info("%s | %s at %.4f um: ratio %.4f +- %.4f over %d spaxel(s), %d plane(s); field level %.4g | %.4g %s; "
                     "position b - a = (%+.1f, %+.1f) +- (%.1f, %.1f) mas", a.label, b.label, crossover, ratio.ratio,
                     ratio.error, row["n_spaxels"], ratio.n, row["level_a"], row["level_b"], a.unit or "",
                     row["dra"] * 1e3, row["ddec"] * 1e3, row["dra_error"] * 1e3, row["ddec_error"] * 1e3)

        # 3. scales: the cube's own chain of ratios, those of a stitched spectrum, or none
        if rescale:
            scales, scale_errors = chain_scales(labels, ratios, reference)
        elif adopted is not None:
            missing = [label for label in labels if label not in adopted]
            if missing:
                raise ValueError(f"the stitched spectrum of {scales_stage!r} has no scale for {missing} "
                                 f"(it knows {sorted(adopted)})")
            scales, scale_errors = [adopted[label][0] for label in labels], [adopted[label][1] for label in labels]
        else:
            if reference not in (None, "shortest", "longest") and reference not in labels:
                raise ValueError(f"reference segment {reference!r} is not among {labels} (nor 'shortest'/'longest')")
            scales, scale_errors = [1.0] * len(segments), [0.0] * len(segments)

        # 4. the joint cube: each segment's planes between its crossovers
        edges = [-np.inf, *bounds, np.inf]
        parts: dict[str, list[np.ndarray]] = {"data": [], "err": [], "dq": [], "wmap": [], "wave": []}
        rows: list[dict[str, Any]] = []
        first_plane = 0
        for k, (seg, cube, placement) in enumerate(zip(segments, cubes, placements)):
            # a plane at a crossover belongs to the bluer segment -- also when the two segments name that
            # wavelength with a rounding error between them, which would let both planes through
            keep = (seg.wave > edges[k] * (1.0 + CROSSOVER_TOLERANCE)) & (seg.wave <= edges[k + 1] * (1.0 + CROSSOVER_TOLERANCE)) \
                & (seg.wave >= ranges[k][0]) & (seg.wave <= ranges[k][1])
            n = int(keep.sum())
            if n == 0:
                raise ValueError(f"segment {seg.label} ({ranges[k][0]:.4f}-{ranges[k][1]:.4f} um) has no plane between "
                                 f"its crossovers ({edges[k]:.4f}, {edges[k + 1]:.4f}]")
            data = cube["data"][keep] * np.float32(scales[k])
            error = cube["err"][keep] if "err" in cube else np.full(data.shape, np.nan, dtype=np.float32)
            # the scale uncertainty is part of the error budget of every rescaled spaxel
            error = np.hypot(error * np.float32(scales[k]), cube["data"][keep] * np.float32(scale_errors[k])).astype(np.float32)
            parts["data"].append(data)
            parts["err"].append(error)
            parts["dq"].append(cube["dq"][keep])
            parts["wmap"].append(cube["wmap"][keep] if "wmap" in cube else np.isfinite(data).astype(np.float32))
            parts["wave"].append(seg.wave[keep])
            used = seg.wave[keep]
            rows.append({"label": seg.label, "file": seg.path.name, "wave_min": float(ranges[k][0]),
                         "wave_max": float(ranges[k][1]), "wave_first": float(used[0]), "wave_last": float(used[-1]),
                         "wave_step": float(np.median(np.diff(seg.wave))) if len(seg.wave) > 1 else np.nan,
                         "plane_first": first_plane, "plane_last": first_plane + n - 1, "n_planes": n,
                         "scale": float(scales[k]), "scale_error": float(scale_errors[k]),
                         "spaxel_arcsec": float(seg.grid.scale), "placement": placement.mode,
                         "phase_x": float(placement.phase[0]), "phase_y": float(placement.phase[1]),
                         "correlation": float(placement.correlation)})
            first_plane += n
        wave = np.concatenate(parts["wave"])
        steps = np.diff(wave.astype(np.float32))                # as they are written: the wavelength table is float32
        if np.any(steps <= 0):
            where = float(wave[int(np.argmax(steps <= 0))])
            raise ValueError(f"the wavelengths of the joint cube do not increase at {where:.6f} um: two planes of "
                             "neighbouring segments coincide in the precision of the wavelength table; move the crossover")

        # 5. every segment over one region of the field all of them cover, for the overview figure: the
        #    quantity of the ratios (the source's excess over the field level, or the plain sum of the field)
        common_all = np.logical_and.reduce(footprints)
        source_all = np.zeros(grid.shape, dtype=bool)
        segment_spectra: list[dict[str, Any]] = []
        if common_all.any():
            if ratio_region == "source":
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*(Mean of empty slice|All-NaN).*")
                    overall = np.nanmean(np.stack([collapse(c["data"]) for c in cubes]), axis=0)
                source_all = source_region(overall, overall, common_all, source_fraction)
            else:
                source_all = common_all
            for k, (seg, cube, placement) in enumerate(zip(segments, cubes, placements)):
                flux, error = region_spectrum(cube["data"], cube.get("err"), source_all, pixel_area_sr=area_sr,
                                              correlation=placement.correlation,
                                              field=common_all if ratio_region == "source" else None)
                segment_spectra.append({"index": np.full(len(flux), k), "segment": np.full(len(flux), seg.label),
                                        "wavelength": seg.wave, "flux": flux, "error": error})
        else:
            log.warning("no spaxel is covered by all %d segments; no common-field spectra recorded", len(segments))

        resampled = [s.label for s, p in zip(segments, placements) if p.mode != IDENTITY]
        reference_label = (labels[-1] if reference in (None, "longest") else labels[0] if reference == "shortest"
                           else str(reference)) if rescale else "none"
        scaling = "measured" if rescale else "adopted" if adopted is not None else "none"
        keys: dict[str, tuple[Any, str]] = {
            "JWFJCVER": (self.version, "stitch_cubes step version"),
            "JWFJCNSG": (len(segments), "segments joined"),
            "JWFJCNRS": (len(resampled), "segments resampled onto the joint grid"),
            "JWFJCGRD": (labels[finest], "segment whose pixel lattice the grid extends"),
            "JWFJCSCL": (scaling, "scales: none | measured (own ratios) | adopted"),
            "JWFJCREF": (reference_label, "measured scales: the reference segment"),
            "JWFJCSFM": (str(scales_stage) if adopted is not None else "none", "adopted scales: stage of the spectrum"),
            "JWFJCWFR": (float(ratio_window_frac), "ratio window / crossover wavelength"),
            "JWFJCREG": (ratio_region, "what the ratios compare"),
            "JWFJCSFR": (float(source_fraction), "source region: fraction of the peak excess"),
            "JWFJCMCV": (float(min_coverage), "resampling: min covered fraction of a spaxel"),
        }
        out = ctx.output_dir / derived_name(_product_stem([s.path for s in segments]), JOINT_SUFFIX)
        write_joint_cube(out, like=segments[finest].path, grid=grid, wave=wave,
                         data=np.concatenate(parts["data"]), err=np.concatenate(parts["err"]),
                         dq=np.concatenate(parts["dq"]), wmap=np.concatenate(parts["wmap"]), segments=rows,
                         overlaps=overlaps, segment_spectra=segment_spectra, overlap_spectra=overlap_spectra,
                         overlap_images=images, overlap_regions=regions, overlap_cores=cores, common=common_all,
                         source=source_all,
                         instrument=_instrument_keys(segments), keys=keys)
        log.info("%s: %d segment(s) %s -> %d planes %.4f-%.4f um on %d x %d spaxels; crossovers %s; scales %s%s",
                 out.name, len(segments), labels, len(wave), wave[0], wave[-1], grid.shape[1], grid.shape[0],
                 [f"{b:.4f}" for b in bounds], [f"{s:.4f}+-{e:.4f}" for s, e in zip(scales, scale_errors)],
                 f"; resampled: {resampled}" if resampled else "; no segment resampled")
        return out


def _adopted_scales(ctx: RunContext, stage: str, pattern: str) -> dict[str, tuple[float, float]]:
    """``label -> (scale, scale error)`` of the stitched spectrum (``*_s1dcomb.ecsv``, the stitching
    contract of :mod:`jwstflow.spectra.stitching`) that ``pattern`` names in the directory of ``stage``."""
    from astropy.table import Table

    try:
        directory = ctx.dir_of(stage)
    except KeyError as exc:
        raise KeyError(f"scales_from: {stage!r} is not a stage of this workflow") from exc
    found = sorted(Path(directory).glob(pattern))
    if len(found) != 1:
        raise ValueError(f"scales_from: {pattern!r} matches {len(found)} file(s) in {directory} "
                         f"({', '.join(f.name for f in found[:4]) or 'none'}); narrow scales_pattern to exactly one "
                         "(and declare depends_on on that stage)")
    meta = Table.read(found[0]).meta
    labels, scales = [str(v) for v in meta.get("segments", [])], [float(v) for v in meta.get("scales_applied", [])]
    errors = [float(v) for v in meta.get("scale_errors", [0.0] * len(scales))]
    if not labels or len(labels) != len(scales):
        raise ValueError(f"{found[0].name} carries no scales (segments / scales_applied in its metadata)")
    log.info("scales adopted from %s: %s", found[0].name, {k: round(v, 4) for k, v in zip(labels, scales)})
    return {label: (scale, error) for label, scale, error in zip(labels, scales, errors)}


def _product_stem(paths: list[Path]) -> str:
    """Longest common prefix of the segment file names, cut back to a whole ``_`` token
    (``jw01751-o006_t010_nirspec_g235h...`` + ``..._g395h...`` -> ``jw01751-o006_t010_nirspec``)."""
    stems = [p.stem for p in paths]
    prefix = os.path.commonprefix(stems)
    if prefix and len(prefix) < len(stems[0]) and stems[0][len(prefix)] not in "_-":
        prefix = prefix.rsplit("_", 1)[0]
    prefix = prefix.rstrip("-_")
    return (prefix or stems[0]) + ".fits"     # derived_name strips the extension again


def _instrument_keys(segments: list[CubeSegment]) -> dict[str, str]:
    """Band keywords of the joint cube, as datamodel attributes of ``meta.instrument``: what all
    segments share stays, what differs becomes MULTIPLE (channels: their digits, as cube_build
    writes for a multi-channel cube)."""
    out: dict[str, str] = {}
    for key, attribute in (("GRATING", "grating"), ("FILTER", "filter"), ("BAND", "band"), ("DETECTOR", "detector")):
        values = {str(s.primary.get(key) or "").strip().upper() for s in segments} - {""}
        if len(values) > 1:
            out[attribute] = "MULTIPLE"
    channels = "".join(sorted({c for s in segments for c in str(s.primary.get("CHANNEL") or "") if c.isdigit()}))
    if len(channels) > 1:
        out["channel"] = channels
    return out
