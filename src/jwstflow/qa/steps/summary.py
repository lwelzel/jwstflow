"""Run overviews that are tables rather than figures."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from ...data.discovery import read_metadata
from ...steps.base import RunContext, Step


class HeaderSummary(Step):
    """One JSON/CSV table with the key header values of all inputs (batch step).

    Runs once over the whole input set (``batch=all``): for every file the
    primary-header keywords in ``keys`` (default: exposure type, detector,
    optical elements, target and exposure time -- EXP_TYPE, DETECTOR,
    GRATING, FILTER, CHANNEL, BAND, TARGPROP, BKGDTARG, IS_IMPRT, EFFEXPTM)
    are collected into one row, and the resulting table is written twice:
    ``<name>.json`` and ``<name>.csv``. Handy as a run overview -- which
    exposure is which grating/band, which are backgrounds or imprints --
    without opening any FITS file.
    """

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
