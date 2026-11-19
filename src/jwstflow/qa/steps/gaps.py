"""QA of the detector-gap coverage: where the gap edges lie on the sky, and how thinly sampled their stretch is."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ... import qa
from ...cubes.gaps import GAPCOV_SUFFIX, read_gap_product
from ...steps import RunContext, Step

log = logging.getLogger(__name__)


class QaGapCoverage(Step):
    """The coverage next to the NIRSpec detector gap on the sky: where all exposures stop, and how long
    the thinly sampled stretch beyond is.

    One figure per ``*_gapcov.fits`` product, four frames on the product's
    sky grid (pixel and arcsec offsets from the target): on top the coverage
    limits -- the last wavelength before the gap and the first after it that
    all exposures covering a position reach (``LIMLO``/``LIMHI``); below,
    the stretch between that limit and the edge the farthest-reaching
    exposure attains (``REACHLO - LIMLO``, ``LIMHI - REACHHI``, in nm), where
    only some exposures contribute and whose pixels ``flag_gap_outliers``
    tests. Positions no exposure covers are grey. The legend notes the group
    (exposures, staircase pixels tested and flagged, the window) and the
    parameters the product records.
    """

    name = "qa_gap_coverage"
    level = "qa"
    inputs = (f"*_{GAPCOV_SUFFIX}.fits",)
    version = "2"   # 1 -> 2: the product of flag_gap_outliers (pixels tested, not flagged by coverage)

    def run(self, inputs: list[Path], ctx: RunContext, *, panel: float = 4.2, dpi: int = 150,
            **params: Any) -> list[Path]:
        out: list[Path] = []
        for path in inputs:
            prod = read_gap_product(path)
            hdr = prod["header"]
            shape = prod["limlo"].shape
            sky = qa.SkyFrame.from_header(prod["map_header"], shape, primary=hdr)
            ratio = shape[0] / max(shape[1], 1)
            fig, axes = qa.subplots(2, 2, figsize=(2 * panel + 2.0, 2 * panel * ratio + 1.2), squeeze=False)
            lam = qa.label(r"$\lambda$", qa.UM)
            dlam = qa.label(r"$\Delta\lambda$", "nm")
            with np.errstate(invalid="ignore"):
                wid_lo = (prod["reachlo"] - prod["limlo"]) * 1e3
                wid_hi = (prod["limhi"] - prod["reachhi"]) * 1e3
            panels = (
                (axes[0, 0], prod["limlo"], lam, "all exposures reach this before the gap (NRS1)", "lt"),
                (axes[0, 1], prod["limhi"], lam, "all exposures reach this after the gap (NRS2)", "tr"),
                (axes[1, 0], wid_lo, dlam, "partially covered stretch before the gap", "lb"),
                (axes[1, 1], wid_hi, dlam, "partially covered stretch after the gap", "rb"),
            )
            frame = None
            for ax, img, cbar_label, text, sides in panels:
                frame = qa.Frame(ax)
                finite = img[np.isfinite(img)]
                lo, hi = (np.percentile(finite, [1.0, 99.0]) if finite.size else (0.0, 1.0))
                if not hi > lo:
                    hi = lo + (1e-3 if cbar_label == lam else 1.0)
                frame.image(img, sky=sky, stretch="linear", vmin=float(lo), vmax=float(hi), cbar_label=cbar_label,
                            sides=sides)
                qa.tag(ax, text)
            n_exp = int(hdr.get("JWFGPNEX", 0))
            qa.annotate(axes[0, 0], path.name,
                        f"{n_exp} exposure(s), {len(prod['files'])} cal file(s); window "
                        f"{qa.um_range(float(hdr.get('JWFGPWLO', np.nan)), float(hdr.get('JWFGPWHI', np.nan)), '.4f')}; "
                        f"{int(hdr.get('JWFGPNST', 0))} staircase pixels, {int(hdr.get('JWFGPNTS', 0))} tested "
                        f"against the other exposures, {int(hdr.get('JWFGPNPX', 0))} flagged "
                        rf"(> {float(hdr.get('JWFGPSIG', np.nan)):.1f}$\sigma$)",
                        f"median partially covered stretch {float(hdr.get('JWFGPDLO', np.nan)) * 1e3:.1f} nm before, "
                        f"{float(hdr.get('JWFGPDHI', np.nan)) * 1e3:.1f} nm after the gap; footprint "
                        f"{float(hdr.get('JWFGPFTP', np.nan)):.3f} arcsec, cell {float(hdr.get('JWFGPCEL', np.nan)):.3f} arcsec",
                        frame.note() if frame is not None else "")
            qa.figlegend(fig, ncol=1)
            out.append(qa.save(fig, ctx.output_dir / f"{path.stem}.png", dpi=dpi))
        log.info("qa_gap_coverage: %d figure(s)", len(out))
        return out
