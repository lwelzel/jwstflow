"""Tools for developing and testing custom steps without real data or a full run.

* :func:`run_step` executes a step exactly as a worker would (RunContext,
  parameter validation, output checks) on files of your choice;
* :func:`synthetic_cube`, :func:`synthetic_x1d`, :func:`synthetic_image` write
  small JWST-like products (IFU cube, extracted spectrum, rate/cal image)
  with the keywords jwstflow steps typically read;
* :func:`check_step` audits a step's declaration and signature without running
  it (what ``jwstflow check-step`` prints);
* :func:`scaffold` writes a new step module plus a test from a template
  (``jwstflow new-step``).
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import numpy as np

from .steps.base import (
    FunctionStep,
    RunContext,
    Step,
    check_declaration,
    check_outputs,
    is_stpipe_step,
    make_step,
    resolve_target,
)

# --------------------------------------------------------------------------- running


def make_context(tmp_path: Path, *, stage: str = "test_step", target: str = "Test Target",
                 target_coords: dict[str, Any] | None = None) -> RunContext:
    """A RunContext rooted at ``tmp_path/<target>/<run>`` with the usual directories created."""
    target_dir = Path(tmp_path) / "test-target"
    run_dir = target_dir / "test_run"
    out = run_dir / "stage4" / stage
    for d in (target_dir / "raw", out, run_dir / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    return RunContext(run_name="test-target/test_run", root=run_dir, stage=stage, output_dir=out, log_dir=run_dir / "logs",
                      raw_dir=target_dir / "raw", stage_dirs={}, crds_context=None, dry_run=False, task_id="test",
                      target=target, target_coords=target_coords, target_dir=target_dir,
                      reference_dir=target_dir / "mast_reference" / "test_run")


def run_step(step: str | type | Step | Any, inputs: list[Path] | Path, tmp_path: Path, *,
             params: dict[str, Any] | None = None, ctx: RunContext | None = None, **ctx_kwargs: Any) -> list[Path]:
    """Run ``step`` (spec string, class, instance or function) on ``inputs`` like a worker would:
    parameters validated against ``Params``, outputs checked (exist, in the output directory,
    non-reserved suffix). Returns the outputs."""
    obj = make_step(step) if isinstance(step, str) else step
    if inspect.isclass(obj):
        obj = obj()
    elif not isinstance(obj, Step) and callable(obj):
        obj = FunctionStep(obj)
    ctx = ctx or make_context(tmp_path, **ctx_kwargs)
    files = [Path(p) for p in ([inputs] if isinstance(inputs, (str, Path)) else inputs)]
    validated = obj.validate_params(dict(params or {}))
    outputs = obj.run(files, ctx, **validated)
    return check_outputs(outputs, ctx, obj)


# --------------------------------------------------------------------------- synthetic data


def _primary(instrument: str, **keys: Any) -> Any:
    from astropy.io import fits

    hdr = fits.Header()
    base = {"TELESCOP": "JWST", "INSTRUME": instrument, "PROGRAM": "01234", "OBSERVTN": "001", "VISIT": "001",
            "TARGPROP": "TEST-TARGET", "TARG_RA": 10.0, "TARG_DEC": -20.0, "BKGDTARG": False, "IS_IMPRT": False,
            "PATT_NUM": 1, "DATE-OBS": "2024-01-01", "CAL_VER": "3.0.0", "CRDS_CTX": "jwst_1584.pmap"}
    base.update(keys)
    for k, v in base.items():
        hdr[k] = v
    return fits.PrimaryHDU(header=hdr)


def synthetic_image(path: Path, *, instrument: str = "NIRSPEC", shape: tuple[int, int] = (64, 64),
                    value: float = 1.0, **keys: Any) -> Path:
    """A rate/cal-like file with SCI, ERR and DQ extensions."""
    from astropy.io import fits

    keys.setdefault("EXP_TYPE", "NRS_IFU" if instrument == "NIRSPEC" else "MIR_MRS")
    keys.setdefault("DETECTOR", "NRS1" if instrument == "NIRSPEC" else "MIRIFUSHORT")
    sci = np.full(shape, value, dtype="f4")
    fits.HDUList([_primary(instrument, **keys), fits.ImageHDU(sci, name="SCI"), fits.ImageHDU(np.full(shape, 0.1, "f4"), name="ERR"),
                  fits.ImageHDU(np.zeros(shape, "u4"), name="DQ")]).writeto(path, overwrite=True)
    return Path(path)


def synthetic_cube(path: Path, *, instrument: str = "MIRI", nwave: int = 50, size: int = 40, wave_min: float = 4.9,
                   wave_step: float = 0.01, source_flux_jy: float = 1e-3, fwhm_pix: float = 2.5, background: float = 0.0,
                   pix_arcsec: float = 0.13, **keys: Any) -> Path:
    """An IFU cube (``IFUCubeModel`` when stdatamodels is available, plain FITS otherwise) with a
    Gaussian point source of ``source_flux_jy`` in every plane plus a flat background (MJy/sr)."""
    yy, xx = np.mgrid[:size, :size]
    sig = fwhm_pix / 2.3548
    area_sr = (pix_arcsec / 206265.0) ** 2
    peak = source_flux_jy / (2 * np.pi * sig**2) / area_sr / 1e6
    psf = peak * np.exp(-((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (2 * sig**2)) + background
    data = np.repeat(psf[None], nwave, axis=0).astype("f4")
    try:
        from stdatamodels.jwst import datamodels
    except ImportError:
        datamodels = None
    if datamodels is not None:
        cube = datamodels.IFUCubeModel(data=data, err=np.full_like(data, 1e-3), dq=np.zeros(data.shape, "u4"))
        cube.meta.instrument.name = instrument
        if instrument == "MIRI":
            cube.meta.instrument.channel, cube.meta.instrument.band, cube.meta.exposure.type = "1", "SHORT", "MIR_MRS"
        else:
            cube.meta.instrument.grating, cube.meta.instrument.filter, cube.meta.exposure.type = "G235H", "F170LP", "NRS_IFU"
        cube.meta.target.proposer_name, cube.meta.target.ra, cube.meta.target.dec = "TEST-TARGET", 10.0, -20.0
        w = cube.meta.wcsinfo
        w.ctype1, w.ctype2, w.ctype3 = "RA---TAN", "DEC--TAN", "WAVE"
        w.crval1, w.crval2, w.crval3 = 10.0, -20.0, wave_min
        w.crpix1, w.crpix2, w.crpix3 = size / 2 + 1, size / 2 + 1, 1.0
        w.cdelt1, w.cdelt2, w.cdelt3 = -pix_arcsec / 3600, pix_arcsec / 3600, wave_step
        w.cunit1 = w.cunit2 = "deg"
        cube.meta.photometry.pixelarea_steradians, cube.meta.photometry.pixelarea_arcsecsq = area_sr, pix_arcsec**2
        for k, v in keys.items():
            cube.extra_fits.PRIMARY.header.append((k, v))  # type: ignore[attr-defined]
        cube.save(str(path))
        return Path(path)
    from astropy.io import fits

    sci = fits.ImageHDU(data, name="SCI")
    for k, v in {"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CTYPE3": "WAVE", "CRVAL1": 10.0, "CRVAL2": -20.0, "CRVAL3": wave_min,
                 "CRPIX1": size / 2 + 1, "CRPIX2": size / 2 + 1, "CRPIX3": 1.0, "CDELT1": -pix_arcsec / 3600,
                 "CDELT2": pix_arcsec / 3600, "CDELT3": wave_step, "PIXAR_SR": area_sr, "PIXAR_A2": pix_arcsec**2}.items():
        sci.header[k] = v
    fits.HDUList([_primary(instrument, **keys), sci, fits.ImageHDU(np.full_like(data, 1e-3), name="ERR")]).writeto(path, overwrite=True)
    return Path(path)


def synthetic_x1d(path: Path, *, wave: np.ndarray | None = None, flux: np.ndarray | None = None,
                  instrument: str = "MIRI", **keys: Any) -> Path:
    """An extracted spectrum in the jwst x1d table layout."""
    from .spectra import Spectrum1D, write_x1d

    w = np.linspace(4.9, 5.7, 200) if wave is None else np.asarray(wave, float)
    f = np.ones_like(w) if flux is None else np.asarray(flux, float)
    header = {"INSTRUME": instrument, "TARGPROP": "TEST-TARGET", "TARG_RA": 10.0, "TARG_DEC": -20.0, "SRCNAME": "test"}
    header.update(keys)
    return write_x1d(Path(path), Spectrum1D(w, f, np.full_like(f, 0.01)), header=header)


# --------------------------------------------------------------------------- auditing


def check_step(spec: str | type | Any) -> tuple[dict[str, Any], list[str]]:
    """Audit a step without running it: ``(description, problems)``.

    Checks: the spec imports; it is a jwstflow Step, an stpipe Step, or a
    callable; a Step's declaration (name, level, batch, inputs, outputs, Params)
    is valid; ``run`` has the ``(inputs, ctx, **params)`` shape; keyword
    parameters of ``run`` are consistent with ``Params`` when both exist.
    """
    problems: list[str] = []
    try:
        target = resolve_target(spec) if isinstance(spec, str) else spec
    except Exception as exc:
        return {"spec": str(spec)}, [f"cannot import {spec!r}: {exc}"]
    if is_stpipe_step(target):
        return {"spec": str(spec), "kind": "stpipe", "object": f"{target.__module__}.{target.__name__}",
                "class_alias": getattr(target, "class_alias", None)}, []
    if isinstance(target, Step):
        cls: Any = type(target)
    elif inspect.isclass(target) and issubclass(target, Step):
        cls = target
    elif callable(target):
        return _check_function(target, spec)
    else:
        return {"spec": str(spec)}, [f"{spec!r} is neither a jwstflow Step, an stpipe Step nor a function"]
    problems += check_declaration(cls)
    sig = inspect.signature(cls.run)
    names = [p.name for p in sig.parameters.values()]
    if names[:3] != ["self", "inputs", "ctx"]:
        problems.append(f"run() must start with (self, inputs, ctx), got ({', '.join(names)})")
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        problems.append("run() should accept **params so that extra parameters never break it")
    keyword_only = [p.name for p in sig.parameters.values() if p.kind is inspect.Parameter.KEYWORD_ONLY]
    if cls.Params is not None:
        declared = set(cls.Params.model_fields)
        missing = [k for k in keyword_only if k not in declared]
        if missing:
            problems.append(f"run() keyword parameters {missing} are not declared in Params")
    if not (cls.__doc__ or "").strip():
        problems.append("add a docstring: its first line is the step's description")
    desc = cls.describe()
    desc["spec"] = str(spec)
    desc["kind"] = "step"
    return desc, problems


def _check_function(func: Any, spec: Any) -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []
    names = [p.name for p in inspect.signature(func).parameters.values()]
    if names[:2] != ["inputs", "ctx"]:
        problems.append(f"a step function must be f(inputs, ctx, **params), got ({', '.join(names)})")
    wrapped = FunctionStep(func)
    problems += check_declaration(type("_Decl", (Step,), {"run": lambda self, i, c, **p: None,
                                                                 **{a: getattr(wrapped, a) for a in FunctionStep.DECLARATION if hasattr(wrapped, a)}}))
    return {"spec": str(spec), "kind": "function", "object": f"{func.__module__}.{func.__qualname__}",
            "level": wrapped.level, "batch": wrapped.batch, "outputs": list(wrapped.outputs)}, problems


# --------------------------------------------------------------------------- scaffolding

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


def scaffold(class_name: str, directory: Path, *, level: int | str = 4, inputs: str = "*_cal.fits",
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
