"""Example user-defined steps.

These double as documentation of the :class:`jwstflow.Step` API:

* ``run(inputs, ctx, **params)`` receives the selected files and a
  :class:`RunContext` (output directory, other stage directories, ...),
* it returns the paths it wrote (jwstflow records them for checkpointing),
* ``batch = "all"`` makes a step receive every input at once.

Reference them in YAML by entry-point name (``plot_spectrum``) or dotted path
(``jwstflow.contrib.qa:PlotSpectrum``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from ..data.discovery import read_metadata
from ..steps.base import RunContext, Step

log = logging.getLogger(__name__)


def _matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("plot steps need `pip install matplotlib`") from exc


class PlotSpectrum(Step):
    """Quick-look plot of every 1-D spectrum (``*_x1d.fits`` / ``*_c1d.fits``)."""

    level = "qa"

    def run(
        self,
        inputs: list[Path],
        ctx: RunContext,
        *,
        ylim: tuple[float, float] | list[float] | None = None,
        column: str = "FLUX",
        dpi: int = 120,
        fmt: str = "png",
        **_: Any,
    ) -> Iterable[Path]:
        from astropy.io import fits

        plt = _matplotlib()
        out: list[Path] = []
        for inp in inputs:
            with fits.open(inp) as hdul:
                tables = [h for h in hdul if h.name == "EXTRACT1D" or (h.name == "COMBINE1D")]
                if not tables:
                    log.warning("%s has no EXTRACT1D/COMBINE1D extension; skipped", inp.name)
                    continue
                fig, ax = plt.subplots(figsize=(9, 4))
                for i, h in enumerate(tables):
                    tab = h.data
                    if "WAVELENGTH" not in tab.names or column not in tab.names:
                        continue
                    label = h.header.get("SLTNAME") or h.header.get("SRCNAME") or f"ext {i + 1}"
                    ax.plot(tab["WAVELENGTH"], tab[column], lw=0.8, label=str(label))
                ax.set_xlabel("wavelength [um]")
                ax.set_ylabel(column)
                ax.set_title(inp.name)
                if ylim:
                    ax.set_ylim(*ylim)
                if len(tables) > 1 and len(tables) <= 12:
                    ax.legend(fontsize=7)
                fig.tight_layout()
                path = ctx.output_dir / f"{inp.stem}.{fmt}"
                fig.savefig(path, dpi=dpi)
                plt.close(fig)
                out.append(path)
        return out


class QuicklookImage(Step):
    """PNG of the SCI extension (2-D images, or the middle slice of 3-D cubes)."""

    level = "qa"

    def run(
        self,
        inputs: list[Path],
        ctx: RunContext,
        *,
        percentiles: tuple[float, float] | list[float] = (1.0, 99.0),
        cmap: str = "viridis",
        dpi: int = 120,
        **_: Any,
    ) -> Iterable[Path]:
        import numpy as np
        from astropy.io import fits

        plt = _matplotlib()
        out: list[Path] = []
        for inp in inputs:
            with fits.open(inp) as hdul:
                if "SCI" not in hdul:
                    log.warning("%s has no SCI extension; skipped", inp.name)
                    continue
                data = np.asarray(hdul["SCI"].data, dtype=float)
            while data.ndim > 2:
                data = data[data.shape[0] // 2]
            finite = data[np.isfinite(data)]
            lo, hi = (np.percentile(finite, percentiles) if finite.size else (0.0, 1.0))
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(data, origin="lower", vmin=lo, vmax=hi, cmap=cmap)
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(inp.name, fontsize=9)
            fig.tight_layout()
            path = ctx.output_dir / f"{inp.stem}.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            out.append(path)
        return out


class HeaderSummary(Step):
    """One JSON/CSV table with the key header values of all inputs (batch step)."""

    level = "qa"

    batch: ClassVar[str] = "all"

    def run(
        self,
        inputs: list[Path],
        ctx: RunContext,
        *,
        name: str = "summary",
        keys: list[str] | None = None,
        **_: Any,
    ) -> Iterable[Path]:
        keys = keys or ["EXP_TYPE", "DETECTOR", "GRATING", "FILTER", "CHANNEL", "BAND", "TARGPROP", "BKGDTARG", "IS_IMPRT", "EFFEXPTM"]
        rows = []
        for inp in inputs:
            meta = read_metadata(inp)
            rows.append({"file": inp.name, **{k: meta.get(k) for k in keys}})
        js = ctx.output_dir / f"{name}.json"
        js.write_text(json.dumps(rows, indent=2, default=str))
        csv = ctx.output_dir / f"{name}.csv"
        with csv.open("w") as fh:
            fh.write(",".join(["file", *keys]) + "\n")
            for r in rows:
                fh.write(",".join(str(r.get(k, "")) for k in ["file", *keys]) + "\n")
        return [js, csv]


def copy_inputs(inputs: list[Path], ctx: RunContext, *, suffix: str = "_copy") -> list[Path]:
    """Trivial function-style step (mostly for tests and as a template)."""
    import shutil

    out = []
    for inp in inputs:
        dst = ctx.output_dir / f"{inp.stem}{suffix}{inp.suffix}"
        shutil.copy2(inp, dst)
        out.append(dst)
    return out
