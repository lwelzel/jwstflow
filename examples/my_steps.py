"""Custom steps for jwstflow.

Three flavours are shown; reference them from YAML by dotted path
(``step: my_steps:Detector1WithSnowballMask``) or install a package that
exposes them via the ``jwstflow.steps`` entry-point group.

1. Subclassing an official pipeline / step (any ``stpipe.Step`` works).
2. A ``jwstflow.Step`` subclass — plain Python, gets a ``RunContext``.
3. A plain function ``f(inputs, ctx, **params) -> list[Path]``.

Run ``python -m pytest`` in this directory for a tiny self-test, or point the
``examples/custom_steps.yaml`` workflow at real data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from jwstflow import RunContext, Step

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1) Subclass an official pipeline: same YAML parameters, extra behaviour.
#    jwstflow detects stpipe classes lazily inside the worker processes, so the
#    jwst import is local: `jwstflow validate --resolve` still works elsewhere.
# ---------------------------------------------------------------------------
try:
    from jwst.pipeline import Detector1Pipeline
    from jwst.step import Extract1dStep
except ImportError:  # keep the module importable without jwst
    Detector1Pipeline = object  # type: ignore[assignment,misc]
    Extract1dStep = object  # type: ignore[assignment,misc]


class Detector1WithSnowballMask(Detector1Pipeline):  # type: ignore[misc,valid-type]
    """Detector1 that forces snowball flagging on and stores a small QA summary.

    It inherits `class_alias = "calwebb_detector1"`, so in a workflow it *is* the
    detector1 stage (same directory, replaces the preset's). Give a subclass its
    own `class_alias` only when it should be a separate stage.
    """

    # Extra parameters are declared with the normal stpipe spec syntax and can be
    # set from YAML under `parameters:` like any other pipeline parameter.
    spec = """
    qa_summary = boolean(default=True)  # write <stem>_det1qa.json next to the rate file
    """

    def process(self, input):
        self.jump.expand_large_events = True
        result = super().process(input)
        if self.qa_summary and self.output_dir:
            stem = Path(self.output_file or getattr(result.meta, "filename", "output")).stem
            summary = {"nints": int(result.meta.exposure.nints), "ngroups": int(result.meta.exposure.ngroups)}
            Path(self.output_dir, f"{stem}_det1qa.json").write_text(json.dumps(summary, indent=2))
        return result


class Extract1dWideAperture(Extract1dStep):  # type: ignore[misc,valid-type]
    """extract_1d with a different default; run it as a separate stage on _s3d cubes."""

    def process(self, input):
        self.ifu_rfcorr = True
        return super().process(input)


# ---------------------------------------------------------------------------
# 2) A jwstflow-native step. `batch = "all"` gives the step every input file at
#    once (one task); the default `"per_file"` gives one task per input file,
#    which then run in parallel across workers.
# ---------------------------------------------------------------------------
class SpectrumReport(Step):
    """Collect all *_x1d.fits products of a stage into one JSON report."""

    batch = "all"

    def run(self, inputs: list[Path], ctx: RunContext, *, min_snr: float = 3.0, **params) -> list[Path]:
        from astropy.io import fits

        rows = []
        for path in inputs:
            with fits.open(path) as hdul:
                hdr = hdul[0].header
                rows.append({"file": path.name, "target": hdr.get("TARGPROP"), "grating": hdr.get("GRATING")})
        out = ctx.output_dir / f"{ctx.run_name}_spectrum_report.json"
        out.write_text(json.dumps({"min_snr": min_snr, "n": len(rows), "rows": rows}, indent=2))
        log.info("wrote report for %d spectra (CRDS context %s)", len(rows), ctx.crds_context)
        return [out]


# ---------------------------------------------------------------------------
# 3) The smallest possible step: a function.
# ---------------------------------------------------------------------------
def tag_header(inputs: list[Path], ctx: RunContext, *, keyword: str = "JWSTFLOW", value: str = "1") -> list[Path]:
    """Copy each input into the stage directory with an extra header keyword."""
    from astropy.io import fits

    outputs = []
    for path in inputs:
        dst = ctx.output_dir / path.name.replace(".fits", "_tagged.fits")
        with fits.open(path) as hdul:
            hdul[0].header[keyword] = value
            hdul.writeto(dst, overwrite=True)
        outputs.append(dst)
    return outputs
