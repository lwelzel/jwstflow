"""Quicklook of image-like products."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ... import qa
from ...steps.base import RunContext, Step

log = logging.getLogger(__name__)


class QuicklookImage(Step):
    """PNG of the SCI extension (2-D images; 3-D cubes are nan-median collapsed).

    For each input file, the SCI array is rendered as one frame: cubes are
    first collapsed along the spectral axis with a nan-aware statistic
    (``collapse``: median/mean/sum/max; spaxels with less than
    ``min_coverage`` finite planes are blanked -- at the footprint edges a
    nanmedian over a handful of planes is noise that would otherwise stretch
    the display limits), then shown with robust percentile limits
    (``percentiles``) under the ``stretch`` (``asinh`` by default: linear
    around the background, logarithmic above, so a bright source does not
    blank the rest of the field; ``linear``/``sqrt``/``log`` are the
    alternatives), and a height-matched colorbar labelled with the data's
    BUNIT (surface brightness is converted to mJy/arcsec^2). Pixels without
    data are grey. When the SCI header carries a celestial WCS (cubes,
    resampled images) the frame has two sets of axes from one zero at the
    target position: offsets in pixels and offsets on the sky in arcsec;
    detector images keep plain pixel coordinates. Inputs without a SCI
    extension are skipped with a warning. One PNG per input file, named
    after it.
    """

    level = "qa"
    version = "5"   # 4 -> 5: pixel and sky offsets from one zero

    def run(
        self,
        inputs: list[Path],
        ctx: RunContext,
        *,
        percentiles: tuple[float, float] | list[float] = (1.0, 99.0),
        collapse: str = "median",
        min_coverage: float = 0.5,
        stretch: str = "asinh",
        cmap: str | None = None,
        dpi: int = 150,
        **_: Any,
    ) -> Iterable[Path]:
        from astropy.io import fits

        out: list[Path] = []
        for inp in inputs:
            with fits.open(inp) as hdul:
                if "SCI" not in hdul:
                    log.warning("%s has no SCI extension; skipped", inp.name)
                    continue
                data = np.asarray(hdul["SCI"].data, dtype=float)
                unit = hdul["SCI"].header.get("BUNIT")
                sky = qa.SkyFrame.from_header(hdul["SCI"].header, data.shape, primary=hdul[0].header)
            frame = qa.Frame()
            frame.image(qa.collapse(data, collapse, min_coverage=min_coverage), unit=unit, sky=sky, stretch=stretch,
                        percentiles=tuple(percentiles), cmap=cmap)
            out.append(frame.finish(ctx.output_dir / f"{inp.stem}.png", dpi=dpi, notes=[
                inp.name, f"{collapse} over the spectral axis" if data.ndim > 2 else ""]))
        return out
