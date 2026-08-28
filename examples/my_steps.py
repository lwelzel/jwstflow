"""Custom steps for jwstflow -- the three flavours, written against the step contract.

Reference them from a workflow by dotted path (``step: my_steps:SpectrumReport``
with ``plugins: [./my_steps.py]``), by file path (``step: ./my_steps.py:tag_header``),
or install a package that registers them in the ``jwstflow.steps`` entry-point group.

1. Subclass an official pipeline/step (any ``stpipe.Step``): same YAML
   parameters as the original plus your own; it keeps the parent's
   ``class_alias`` and therefore its stage name unless you set a new alias.
2. A ``jwstflow.Step`` subclass: declare what it accepts and produces, type the
   parameters, implement ``run``. This is the recommended form.
3. A plain function ``f(inputs, ctx, **params) -> list[Path]`` for one-liners;
   the same declaration is possible through function attributes.

Check and test them without a run:

    jwstflow check-step ./my_steps.py:SpectrumReport
    python -m pytest examples/test_my_steps.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import Field

from jwstflow import RunContext, Step, StepParams

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1) Subclass an official pipeline. The jwst import is local so the module also
#    imports where jwst is absent (`jwstflow validate --resolve` on a laptop).
# ---------------------------------------------------------------------------
try:
    from jwst.pipeline import Detector1Pipeline
    from jwst.step import Extract1dStep
except ImportError:  # keep the module importable without jwst
    Detector1Pipeline = object  # type: ignore[assignment,misc]
    Extract1dStep = object  # type: ignore[assignment,misc]


class Detector1WithSnowballMask(Detector1Pipeline):  # type: ignore[misc,valid-type]
    """Detector1 that forces snowball flagging on and stores a small QA summary.

    It inherits ``class_alias = "calwebb_detector1"``, so in a workflow it *is* the
    detector1 stage (same directory, replaces the preset's). Give a subclass its
    own ``class_alias`` only when it should be a separate stage. Parameters are
    declared the stpipe way (``spec``) and set from YAML like any pipeline option.
    """

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
# 2) A jwstflow Step: the recommended form.
#    - `level`, `batch`, `inputs`, `outputs`, `version` describe the step;
#    - `Params` types and documents the `parameters:` block (validated by
#      `jwstflow validate`, listed by `jwstflow check-step`);
#    - `run` gets the inputs, a RunContext, and the validated parameters, and
#      returns every file it wrote (paths from `ctx.derived_path` follow the
#      naming rules automatically).
# ---------------------------------------------------------------------------
class SpectrumReport(Step):
    """Collect all extracted spectra of a stage into one JSON report."""

    level = "qa"
    batch = "all"                         # one task with every input
    inputs = ("*_x1d.fits", "*_s1d.fits")
    outputs = ("report",)                 # -> <run>_report.json
    version = "1"

    class Params(StepParams):
        min_snr: float = Field(3.0, ge=0, description="S/N below which a spectrum is flagged in the report")

    def run(self, inputs: list[Path], ctx: RunContext, *, min_snr: float = 3.0, **params) -> list[Path]:
        from astropy.io import fits

        rows = []
        for path in inputs:
            with fits.open(path) as hdul:
                hdr = hdul[0].header
                rows.append({"file": path.name, "target": hdr.get("TARGPROP"), "grating": hdr.get("GRATING"),
                             "channel": hdr.get("CHANNEL"), "band": hdr.get("BAND")})
        out = ctx.derived_path(f"{ctx.run_name.replace('/', '_')}.fits", "report", ext=".json")
        out.write_text(json.dumps({"min_snr": min_snr, "n": len(rows), "rows": rows}, indent=2))
        ctx.log.info("wrote report for %d spectra (CRDS context %s)", len(rows), ctx.crds_context)
        return [out]


# ---------------------------------------------------------------------------
# 3) The smallest possible step: a function. Declaration attributes are optional.
# ---------------------------------------------------------------------------
def tag_header(inputs: list[Path], ctx: RunContext, *, keyword: str = "JWSTFLOW", value: str = "1", **params) -> list[Path]:
    """Copy each input with an extra header keyword (a jwstflow product, `_tagged` suffix)."""
    from astropy.io import fits

    outputs = []
    for path in inputs:
        dst = ctx.derived_path(path, "tagged")
        with fits.open(path) as hdul:
            hdul[0].header[keyword] = value
            hdul.writeto(dst, overwrite=True)
        outputs.append(dst)
    return outputs


tag_header.level = 4              # type: ignore[attr-defined]
tag_header.outputs = ("tagged",)  # type: ignore[attr-defined]
