"""QA of bad spaxel clusters: every dither at the affected wavelengths, from the products alone."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from .. import qafig
from ..clusters import CLUSTERMASK_SUFFIX, band_key, read_cluster_product
from ..naming import slug
from ..steps.base import RunContext, Step

log = logging.getLogger(__name__)

IMAGE_UNIT = "MJy/sr"


class QaSpaxelClusters(Step):
    """Per region: all dither cubes at the affected wavelengths, the region marked, included
    dithers told from excluded ones.

    One task receives every ``*_clustermask.fits`` product (``batch = "all"``)
    and draws one figure per band and region: **one row per dither cube, one
    column per wavelength plane** stored in the product (the window planes
    plus the ``qa_pad`` context planes on each side; columns are matched
    across dithers by wavelength). Every panel shows the plane with the
    search aperture as a dashed contour; on an *included* dither the flagged
    spaxels are drawn as a solid contour. The legend carries, per dither,
    the decision and its numbers (peak significance, deviant and flagged
    spaxel-planes) and, per region, the position, aperture and window.
    Display limits are shared by all panels of a figure (robust percentiles
    of the *other* dithers' reference images, so the cluster itself never
    sets them). ``panel`` sets the size of one panel in inches.
    """

    name = "qa_spaxel_clusters"
    level = "qa"
    batch = "all"
    inputs = (f"*_{CLUSTERMASK_SUFFIX}.fits",)
    version = "1"

    def run(self, inputs: list[Path], ctx: RunContext, *, panel: float = 2.6, percentiles: tuple[float, float] | list[float] = (1.0, 99.5),
            dpi: int = 150, **params: Any) -> list[Path]:
        products = [read_cluster_product(p) for p in sorted(inputs)]
        by_band: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for p in products:
            by_band.setdefault(band_key(p["header"]), []).append(p)
        out: list[Path] = []
        for members in by_band.values():
            members.sort(key=lambda p: (int(p["header"].get("JWFCLDIT", 0)), p["path"].name))
            region_ids: list[str] = []
            for p in members:
                for rid in p["regions"]["id"] if len(p["regions"]) else []:
                    if str(rid) not in region_ids:
                        region_ids.append(str(rid))
            for rid in region_ids:
                rows = [(p, i) for p in members for i, r in enumerate(p["regions"]) if str(r["id"]) == rid and i < len(p["slices"])]
                if not rows:
                    continue
                out.append(self._figure(rows, rid, ctx, panel=panel, percentiles=tuple(percentiles), dpi=dpi))
        log.info("qa_spaxel_clusters: %d figure(s) from %d product(s)", len(out), len(products))
        return out

    def _figure(self, rows: list[tuple[dict[str, Any], int]], rid: str, ctx: RunContext, *, panel: float,
                percentiles: tuple[float, float], dpi: int) -> Path:
        first, i0 = rows[0]
        col_waves = first["slices"][i0]["wave"]
        nrows, ncols = len(rows), len(col_waves)
        fig, axes = qafig.subplots(nrows, ncols, figsize=(panel * ncols + 1.2, panel * nrows + 0.8), squeeze=False)
        # shared limits from the reference images (the other dithers), in display units
        refs = np.concatenate([qafig.to_mjy(p["slices"][i]["ref"], IMAGE_UNIT)[0].ravel() for p, i in rows])
        refs = refs[np.isfinite(refs)]
        if refs.size == 0:
            refs = np.concatenate([qafig.to_mjy(p["slices"][i]["data"], IMAGE_UNIT)[0].ravel() for p, i in rows])
            refs = refs[np.isfinite(refs)]
        vmin, vmax = (np.percentile(refs, percentiles) if refs.size else (0.0, 1.0))
        if not vmax > vmin:
            vmax = vmin + 1.0
        aper_c, flag_c = qafig.OVERLAY_COLORS[:2]
        region = first["regions"][i0]
        shape = str(region["shape"])
        size = f"r = {float(region['radius']):.3f}\"" if shape == "circle" else shape
        qafig.annotate(axes[0, 0], f"region {rid}: RA {float(region['ra']):.6f} Dec {float(region['dec']):.6f} ({size}), "
                                   f"{float(region['wave_min']):.4f}-{float(region['wave_max']):.4f} um")
        qafig.contour_proxy(axes[0, 0], aper_c, ls="dashed", label="search aperture")
        qafig.contour_proxy(axes[0, 0], flag_c, label="flagged spaxels (included dithers)")
        for r, (p, i) in enumerate(rows):
            row = p["regions"][i]
            sl = p["slices"][i]
            included = bool(row["included"])
            step = float(np.median(np.abs(np.diff(sl["wave"])))) if sl["wave"].size > 1 else np.inf
            for c, wave in enumerate(col_waves):
                ax = axes[r, c]
                k = int(np.argmin(np.abs(sl["wave"] - wave)))
                if abs(sl["wave"][k] - wave) > 0.5 * step:
                    ax.set_axis_off()
                    continue
                qafig.imshow(ax, sl["data"][k], unit=IMAGE_UNIT, vmin=vmin, vmax=vmax, cbar=(c == ncols - 1),
                             cbar_label=qafig.SB_LABEL if c == ncols - 1 else None)
                if sl["aperture"].any():
                    ax.contour(sl["aperture"], levels=[0.5], colors=aper_c, linewidths=0.9, linestyles="dashed")
                in_window = int(row["plane_min"]) <= int(np.argmin(np.abs(p["waves"] - sl["wave"][k]))) <= int(row["plane_max"])
                plane_idx = int(np.argmin(np.abs(p["waves"] - sl["wave"][k])))
                if included and p["mask"][plane_idx].any():
                    ax.contour(p["mask"][plane_idx], levels=[0.5], colors=flag_c, linewidths=1.2)
                ax.text(0.03, 0.97, f"{sl['wave'][k]:.4f} um" + ("" if in_window else " (context)"),
                        transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color="white",
                        bbox={"boxstyle": "round,pad=0.2", "fc": "black", "alpha": 0.5, "lw": 0})
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(f"dither {int(row['dither'])} ({'in' if included else 'out'})\ny [pix]", fontsize=7)
                if r == nrows - 1:
                    ax.set_xlabel("x [pix]", fontsize=7)
            verdict = "included" if included else "excluded"
            qafig.annotate(axes[r, 0], f"dither {int(row['dither'])} [{p['path'].name}]: {verdict} -- "
                                       f"peak {float(row['score']):.1f} sigma, {int(row['n_deviant'])} deviant, "
                                       f"{int(row['n_flagged'])} flagged spaxel-planes ({row['reason']})")
        qafig.figlegend(fig, ncol=1)
        base = re.sub(r"_dither\d+", "", first["path"].stem.replace(f"_{CLUSTERMASK_SUFFIX}", ""))
        return qafig.save(fig, ctx.output_dir / f"{base}_{slug(rid)}_clusters.png", dpi=dpi)
