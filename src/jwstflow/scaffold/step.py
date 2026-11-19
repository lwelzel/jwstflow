"""Scaffolding for a single custom step (``jwstflow new-step``): a documented Step and its test."""

from __future__ import annotations

import re
from pathlib import Path

STEP_TEMPLATE = '''"""{doc}"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from jwstflow import RunContext, Step, StepParams


class {cls}(Step):
    """{doc}"""

    level = {level!r}                 # 1/2/3 jwst stages, 4 derived products, "qa" plots
    batch = "per_file"                # "all": one task receives every input
    inputs = ({inputs!r},)            # accepted input files (glob)
    outputs = ("{suffix}",)           # product suffix(es) written; never a jwst one
    version = "1"                     # bump when results change

    class Params(StepParams):
        scale: float = Field(1.0, gt=0, description="multiply the data by this")

    def run(self, inputs: list[Path], ctx: RunContext, *, scale: float = 1.0, **params) -> list[Path]:
        from astropy.io import fits

        (src,) = inputs
        out = ctx.derived_path(src, "{suffix}")        # <input base>_{suffix}.fits in the stage directory
        with fits.open(src) as hdul:
            hdul["SCI"].data = hdul["SCI"].data * scale
            hdul[0].header["JWFSCALE"] = (scale, "applied by {snake}")
            hdul.writeto(out, overwrite=True)
        ctx.log.info("%s -> %s (scale %.3f)", src.name, out.name, scale)
        return [out]
'''

TEST_TEMPLATE = '''from pathlib import Path

from jwstflow.testing import check_step, run_step, synthetic_image

from {module} import {cls}


def test_declaration_is_valid():
    _, problems = check_step({cls})
    assert problems == []


def test_run_on_synthetic_data(tmp_path: Path):
    src = synthetic_image(tmp_path / "jw00001001001_01101_00001_nrs1_cal.fits", value=2.0)
    outputs = run_step({cls}, [src], tmp_path, params={{"scale": 3.0}})
    assert [o.name for o in outputs] == ["jw00001001001_01101_00001_nrs1_{suffix}.fits"]
    from astropy.io import fits

    assert fits.getdata(outputs[0], "SCI")[0, 0] == 6.0
'''


def scaffold_step(class_name: str, directory: Path, *, level: int | str = 4, inputs: str = "*_cal.fits",
             suffix: str | None = None, doc: str | None = None) -> list[Path]:
    """Write ``<snake>.py`` (a documented Step following the contract) and ``test_<snake>.py``."""
    if not re.fullmatch(r"[A-Z][A-Za-z0-9]*", class_name):
        raise ValueError("class name must be CamelCase, e.g. ExtractExtended")
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
    suffix = suffix or re.sub(r"[^a-z0-9]", "", snake)[:8] or "custom"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    module = directory / f"{snake}.py"
    test = directory / f"test_{snake}.py"
    if module.exists() or test.exists():
        raise FileExistsError(f"{module} or {test} already exists")
    module.write_text(STEP_TEMPLATE.format(cls=class_name, doc=doc or f"{class_name}: describe what it does here.",
                                           level=level, inputs=inputs, suffix=suffix, snake=snake))
    test.write_text(TEST_TEMPLATE.format(module=snake, cls=class_name, suffix=suffix))
    return [module, test]
