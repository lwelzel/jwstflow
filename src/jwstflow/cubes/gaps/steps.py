"""The gap step: ``flag_gap_outliers``.

What it tests and why nothing is masked for coverage is described in :mod:`jwstflow.cubes.gaps`;
the figure of its product, ``qa_gap_coverage``, is in :mod:`jwstflow.qa.steps.gaps`.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field, field_validator

from ...steps import RunContext, Step, StepParams
from ...steps.naming import derived_name
from ..detector import dq_value
from .compare import cross_exposure_deviation
from .coverage import coverage_limits, staircase_pixels
from .edges import (
    BLUE_DETECTOR,
    RED_DETECTOR,
    gap_samples,
    gap_window,
    slice_edges,
    slice_edges_from_wcs,
)
from .groups import DEFAULT_GROUP_BY, exposure_of, group_key, group_name
from .product import GAPCOV_SUFFIX, write_gap_product

log = logging.getLogger(__name__)


class FlagGapOutliers(Step):
    """Test the pixels next to the NIRSpec detector gap against the other exposures; flag only those
    that disagree (DQ-edited cal copies) and record the coverage per group.

    One task receives every cal file (``batch = "all"``) and splits them into
    the groups combined into one cube (``group_by``; the default keeps
    observations, gratings and science/background apart). Per group: the gap
    window is read from the files' WAVELENGTH/REGIONS images (bluest NRS1
    slice end to reddest NRS2 slice start); the WCS of every file is
    evaluated over the columns reaching it; on a ``cell``-arcsec sky grid
    each exposure's reach towards the gap is taken from its pixels within
    ``footprint`` arcsec of every cell, and the coverage limit is the reach
    that ``fraction`` of the exposures covering the cell attain. The
    *staircase pixels* beyond that limit -- the ones that build thinly
    sampled voxels, where ``outlier_detection`` compares the fewest
    exposures -- are each tested against the other exposures at their sky
    position, interpolated to their wavelength: a pixel deviating by more
    than ``sigma`` from the median of at least ``min_others`` exposures (its
    error, the reference's and ``rel_tol`` times the reference in quadrature)
    gets ``dq_flags`` (default ``DO_NOT_USE | OUTLIER``). Everything else is
    kept, also the staircase pixels no other exposure can check: thinly
    sampled data are noisier, not wrong, and masking them by coverage removes
    the physical flux of whatever the moving gap has not yet covered.

    Every input is copied into the stage directory (files outside any
    NIRSpec IFU group with both detectors unchanged), keeping the official
    ``_cal`` name; JWFGPNST / JWFGPNTS / JWFGPNPX (staircase pixels, tested,
    flagged) and the parameters in the primary header record the step. Each
    group also gets a ``*_gapcov.fits`` product: the coverage limits and the
    single-exposure reach on the sky grid (TAN WCS), exposure counts and a
    per-file table; ``jwstflow.cubes.gaps.gap_limits_at`` looks a position up and
    ``qa_gap_coverage`` draws it.

    Put the step between stage 2 and the stage-3 cube builds (after other
    cal-level flagging such as ``flag_frames``). For a mosaic, group the
    exposures as the cube does (``group_by``).
    """

    name = "flag_gap_outliers"
    level = 2
    batch = "all"
    inputs = ("*_cal.fits",)
    outputs = (GAPCOV_SUFFIX,)
    writes_official_products = True   # DQ-edited copies keep the official _cal name; JWFGP* records the edit
    version = "1"

    class Params(StepParams):
        sigma: float = Field(5.0, gt=0, description="deviation from the other exposures that flags a pixel")
        rel_tol: float = Field(0.2, ge=0, description="fraction of the reference added to the noise "
                                                     "(spatial structure below the footprint)")
        min_others: int = Field(2, ge=2, description="other exposures needed to test a pixel")
        fraction: float = Field(1.0, gt=0, le=1, description="fraction of the exposures covering a position "
                                                             "that sets its coverage limit")
        footprint: float = Field(0.075, gt=0, description="radius [arcsec] within which an exposure's pixels "
                                                          "cover a sky position (its nearest pixels: half the "
                                                          "diagonal of the 0.1 arcsec pixel pitch)")
        cell: float = Field(0.025, gt=0, description="sky grid cell [arcsec] (finer than the footprint)")
        group_by: list[str] = Field(default_factory=lambda: list(DEFAULT_GROUP_BY),
                                    description="header keywords of the exposures combined into one cube")
        dq_flags: list[str] = Field(default_factory=lambda: ["DO_NOT_USE", "OUTLIER"],
                                    description="DQ flags set on the flagged pixels (DO_NOT_USE always)")

        @field_validator("group_by")
        @classmethod
        def _keys_upper(cls, v: list[str]) -> list[str]:
            if not v:
                raise ValueError("group_by needs at least one keyword")
            return [str(k).strip().upper() for k in v]

    def run(self, inputs: list[Path], ctx: RunContext, *, sigma: float = 5.0, rel_tol: float = 0.2,
            min_others: int = 2, fraction: float = 1.0, footprint: float = 0.075, cell: float = 0.025,
            group_by: list[str] | None = None, dq_flags: list[str] | None = None, **params: Any) -> list[Path]:
        from astropy.io import fits

        if not inputs:
            raise ValueError("flag_gap_outliers received no cal files")
        if cell > footprint:
            raise ValueError(f"cell ({cell} arcsec) must not exceed footprint ({footprint} arcsec)")
        keys = [str(k).upper() for k in (group_by or DEFAULT_GROUP_BY)]
        bits = dq_value(dq_flags if dq_flags is not None else ["DO_NOT_USE", "OUTLIER"])
        test = {"sigma": sigma, "rel_tol": rel_tol, "min_others": min_others, "fraction": fraction,
                "footprint": footprint, "cell": cell}
        groups: dict[tuple[Any, ...], list[tuple[Path, Any]]] = {}
        passthrough: list[Path] = []
        for path in sorted(Path(p) for p in inputs):
            hdr = fits.getheader(path)
            if str(hdr.get("EXP_TYPE", "")).upper() != "NRS_IFU":
                passthrough.append(path)
                continue
            groups.setdefault(group_key(hdr, keys), []).append((path, hdr))
        out: list[Path] = []
        for path in passthrough:
            log.info("%s: not a NIRSpec IFU exposure; copied unchanged", path.name)
            out.append(self._copy(path, ctx, counts=(0, 0, 0), product="none", bits=bits, test=test))
        for members in groups.values():
            out.extend(self._group(members, ctx, keys=keys, bits=bits, test=test))
        return out

    # -- one group of exposures -------------------------------------------------------------
    def _group(self, members: list[tuple[Path, Any]], ctx: RunContext, *, keys: list[str], bits: int,
               test: dict[str, Any]) -> list[Path]:
        from stdatamodels.jwst import datamodels

        first_hdr = members[0][1]
        name = group_name(first_hdr, keys)
        detectors = {str(h.get("DETECTOR", "")).upper() for _, h in members}
        if not {BLUE_DETECTOR, RED_DETECTOR} <= detectors:
            log.info("%s: %d file(s), detectors %s -- no gap between NRS1 and NRS2; copied unchanged",
                     name, len(members), ",".join(sorted(detectors)) or "?")
            return [self._copy(p, ctx, counts=(0, 0, 0), product="none", bits=bits, test=test) for p, _ in members]

        # 1. the gap window from the wavelength images (the WCS where a file has none)
        edges: dict[Path, dict[str, Any]] = {}
        for path, _ in members:
            e = slice_edges(path)
            if e["edges"] is None:
                with datamodels.open(path) as model:
                    e = slice_edges_from_wcs(model)
            edges[path] = e
        blue = [e["edges"] for e in edges.values() if e["detector"] == BLUE_DETECTOR and e["edges"] is not None
                and e["edges"].size]
        red = [e["edges"] for e in edges.values() if e["detector"] == RED_DETECTOR and e["edges"] is not None
               and e["edges"].size]
        if not blue or not red:
            log.warning("%s: no usable pixels on one side of the gap; copied unchanged", name)
            return [self._copy(p, ctx, counts=(0, 0, 0), product="none", bits=bits, test=test) for p, _ in members]
        disp = float(np.nanmedian([e["dispersion"] for e in edges.values()]))
        b, r = np.concatenate(blue), np.concatenate(red)
        window = gap_window(b, r, disp)
        log.info("%s: %d file(s); slice ends NRS1 %.4f-%.4f um, starts NRS2 %.4f-%.4f um; window %.4f-%.4f um",
                 name, len(members), b.min(), b.max(), r.min(), r.max(), *window)

        # 2. the pixels inside the window, on the sky
        samples: list[dict[str, Any]] = []
        for path, hdr in members:
            with datamodels.open(path) as model:
                s = gap_samples(model, window)
            s.update({"path": path, "exposure": exposure_of(path), "detector": str(hdr.get("DETECTOR", "")).upper()})
            samples.append(s)

        # 3. the coverage, the staircase pixels and their test against the other exposures
        limits = coverage_limits(samples, cell=test["cell"], footprint=test["footprint"], fraction=test["fraction"])
        stairs = [staircase_pixels(s, limits) for s in samples]
        results = cross_exposure_deviation(samples, stairs, limits["grid"], footprint=test["footprint"],
                                           dispersion=disp, rel_tol=test["rel_tol"])
        rows: list[dict[str, Any]] = []
        flags: dict[Path, np.ndarray] = {}
        for s, stair, (z, n) in zip(samples, stairs, results, strict=True):
            tested = stair & (n >= test["min_others"])
            flag = tested & (np.abs(z) > test["sigma"])
            flags[s["path"]] = flag
            e = edges[s["path"]]["edges"]
            rows.append({"file": s["path"].name, "exposure": s["exposure"], "detector": s["detector"],
                         "n_examined": int(s["wave"].size), "n_staircase": int(stair.sum()),
                         "n_tested": int(tested.sum()), "n_flagged": int(flag.sum()),
                         "edge_min": float(np.min(e)) if e is not None and e.size else np.nan,
                         "edge_max": float(np.max(e)) if e is not None and e.size else np.nan})
        n_stair = int(sum(r["n_staircase"] for r in rows))
        n_tested = int(sum(r["n_tested"] for r in rows))
        n_flag = int(sum(r["n_flagged"] for r in rows))
        fin = np.isfinite(limits["limlo"]) & np.isfinite(limits["reachlo"])
        fin_hi = np.isfinite(limits["limhi"]) & np.isfinite(limits["reachhi"])
        wid_lo = float(np.median((limits["reachlo"] - limits["limlo"])[fin])) if fin.any() else np.nan
        wid_hi = float(np.median((limits["limhi"] - limits["reachhi"])[fin_hi])) if fin_hi.any() else np.nan
        product = ctx.output_dir / derived_name(f"{name}.fits", GAPCOV_SUFFIX)
        write_gap_product(product, first_hdr, limits, rows, keys={
            "JWFGPGRP": (",".join(keys), "gap: grouping keywords"),
            "JWFGPNEX": (len(limits["exposures"]), "gap: exposures in the group"),
            "JWFGPNST": (n_stair, "gap: staircase pixels, all files"),
            "JWFGPNTS": (n_tested, "gap: staircase pixels tested"),
            "JWFGPNPX": (n_flag, "gap: pixels flagged, all files"),
            "JWFGPWLO": (float(window[0]), "gap: window start [um]"),
            "JWFGPWHI": (float(window[1]), "gap: window end [um]"),
            "JWFGPDLO": (wid_lo, "median partial stretch before the gap [um]"),
            "JWFGPDHI": (wid_hi, "median partial stretch after the gap [um]"),
            **self._param_cards(test, bits),
        })
        log.info("%s: %d exposure(s); partially covered over %.4f um (blue) / %.4f um (red) at the median sky "
                 "cell; %d staircase pixel(s), %d tested against the other exposures, %d flagged -> %s",
                 name, len(limits["exposures"]), wid_lo, wid_hi, n_stair, n_tested, n_flag, product.name)

        # 4. the cal copies
        out: list[Path] = [product]
        for s, row in zip(samples, rows, strict=True):
            flag = flags[s["path"]]
            out.append(self._copy(s["path"], ctx, counts=(row["n_staircase"], row["n_tested"], row["n_flagged"]),
                                  product=product.name, bits=bits, test=test, yx=(s["y"][flag], s["x"][flag])))
        return out

    @staticmethod
    def _param_cards(test: dict[str, Any], bits: int) -> dict[str, Any]:
        return {"JWFGPSIG": (float(test["sigma"]), "gap: flagging threshold [sigma]"),
                "JWFGPTOL": (float(test["rel_tol"]), "gap: relative tolerance"),
                "JWFGPMIN": (int(test["min_others"]), "gap: other exposures needed for a test"),
                "JWFGPFRC": (float(test["fraction"]), "gap: exposure fraction of the limit"),
                "JWFGPFTP": (float(test["footprint"]), "gap: footprint radius [arcsec]"),
                "JWFGPCEL": (float(test["cell"]), "gap: sky grid cell [arcsec]"),
                "JWFGPDQ": (int(bits), "gap: DQ bits set on flagged pixels")}

    @classmethod
    def _copy(cls, src: Path, ctx: RunContext, *, counts: tuple[int, int, int], product: str, bits: int,
              test: dict[str, Any], yx: tuple[np.ndarray, np.ndarray] | None = None) -> Path:
        """Copy a cal file into the stage directory, set ``bits`` on the pixels ``yx`` and record the step."""
        from astropy.io import fits

        dst = ctx.output_dir / src.name
        shutil.copy2(src, dst)
        with fits.open(dst, mode="update") as hdul:
            if yx is not None and len(yx[0]):
                dq = hdul["DQ"].data
                dq[yx] = dq[yx] | np.asarray(bits, dtype=dq.dtype)
            hdr = hdul[0].header
            hdr["JWFGPNST"] = (int(counts[0]), "gap: staircase pixels")
            hdr["JWFGPNTS"] = (int(counts[1]), "gap: staircase pixels tested")
            hdr["JWFGPNPX"] = (int(counts[2]), "gap: pixels flagged")
            for k, v in cls._param_cards(test, bits).items():
                hdr[k] = v
            hdr["JWFGPPRD"] = (product, "gap: coverage product of the group")
        return dst
