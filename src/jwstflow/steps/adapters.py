"""Official pipeline steps behind the jwstflow step interface.

Any subclass of ``stpipe.Step`` -- the official pipelines, single steps, or your own
subclass such as ``class MyDet1(Detector1Pipeline)`` -- is run through
:class:`JwstStepAdapter`, which calls ``Step.call(...)`` (the preferred API: it pulls
the CRDS parameter-reference files) and reports the files it produced.
:func:`identity_of` gives both kinds of step one identity: the canonical stage name
and the calibration level that selects the stage directory.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import shutil
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .interface import RunContext, Step

log = logging.getLogger(__name__)

# Calibration level of the official pipelines and steps, keyed by stpipe class_alias.
# 1: ramps -> rate, 2: exposure calibration, 3: combined products; unknown single steps -> 3.
STPIPE_LEVELS: dict[str, int] = {
    "calwebb_detector1": 1, "calwebb_dark": 1, "calwebb_guider": 1,
    **{a: 1 for a in ("group_scale", "dq_init", "emicorr", "saturation", "ipc", "superbias", "refpix", "rscd",
                      "firstframe", "lastframe", "linearity", "dark_current", "reset", "persistence",
                      "charge_migration", "jump", "clean_flicker_noise", "ramp_fit", "gain_scale", "undersampling_correction")},
    "calwebb_spec2": 2, "calwebb_image2": 2, "calwebb_wfs-image2": 2,
    **{a: 2 for a in ("assign_wcs", "badpix_selfcal", "msa_flagging", "bkg_subtract", "imprint_subtract", "extract_2d",
                      "master_background_mos", "targ_centroid", "wavecorr", "flat_field", "srctype", "straylight", "fringe",
                      "residual_fringe", "pathloss", "barshadow", "wfss_contam", "photom", "picture_frame", "resample_spec",
                      "nsclean")},
    "calwebb_spec3": 3, "calwebb_image3": 3, "calwebb_tso3": 3, "calwebb_coron3": 3, "calwebb_ami3": 3, "calwebb_wfs-image3": 3,
    **{a: 3 for a in ("assign_mtwcs", "master_background", "outlier_detection", "cube_build", "pixel_replace", "extract_1d",
                      "combine_1d", "spectral_leak", "adaptive_trace_model", "resample", "tweakreg", "skymatch",
                      "source_catalog", "klip", "stack_refs", "align_refs", "white_light", "tso_photometry")},
}


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def identity_of(target: Any, spec: str = "") -> tuple[str, int | str]:
    """Canonical ``(name, level)`` of a resolved step object.

    stpipe classes: their ``class_alias`` (``calwebb_spec3``, ``extract_1d``; a
    subclass inherits its parent's alias unless it sets its own) and the level
    of that alias. jwstflow steps: the class/function ``name`` attribute or the
    snake-cased class name, and their ``level`` attribute (default 4, "qa" for
    plot-type steps).
    """
    if is_stpipe_step(target):
        alias = getattr(target, "class_alias", None) or _snake(target.__name__)
        return str(alias), STPIPE_LEVELS.get(str(alias), 3)
    obj = target if isinstance(target, type) or callable(target) else type(target)
    name = getattr(obj, "name", None) or getattr(obj, "__name__", None) or spec.rpartition(":")[2] or spec
    level = getattr(obj, "level", 4)
    if isinstance(name, str) and name and name[0].isupper():
        name = _snake(name)
    return str(name), level


def is_stpipe_step(obj: Any) -> bool:
    """True if ``obj`` is a class deriving from stpipe.Step (checked by name so
    we never have to import stpipe in the parent process)."""
    if not inspect.isclass(obj):
        return False
    return any(
        base.__name__ == "Step" and base.__module__.split(".")[0] == "stpipe"
        for base in obj.__mro__
    )


class JwstStepAdapter(Step):
    """Runs any ``stpipe.Step`` subclass through ``Step.call``.

    Notes on the call:

    * ``call`` (not ``run``) is used so CRDS parameter-reference files apply.
    * ``configure_log=False`` is passed when supported (jwst >= 1.20), because
      jwstflow configures logging itself (one file per task).
    * The step writes into a private ``<stage>/.tmp-<task>/`` directory which
      is moved into the stage directory on success. That makes output detection
      exact even when many tasks write to the same stage directory, and a
      crashed task leaves no half-written products behind.
    * The returned datamodel is closed immediately to keep memory flat.
    """

    def __init__(self, step_cls: type):
        self.step_cls = step_cls

    def run(self, inputs: list[Path], ctx: RunContext, **params: Any) -> Iterable[Path]:
        if len(inputs) != 1:
            raise ValueError(
                f"{self.step_cls.__name__} expects exactly one input (a file or an association); "
                f"got {len(inputs)}. Use `association:` or `batch: per_file`."
            )
        (inp,) = inputs
        out_dir = ctx.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = out_dir / f".tmp-{ctx.task_id or uuid.uuid4().hex[:8]}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()
        kwargs: dict[str, Any] = dict(params)
        kwargs["output_dir"] = str(tmp_dir)
        kwargs.setdefault("save_results", ctx.extra.get("save_results", True))

        t0 = time.time()
        try:
            result = self._call(str(inp), kwargs)
            _close(result)
            outputs: list[Path] = []
            for p in sorted(tmp_dir.iterdir()):
                dst = out_dir / p.name
                os.replace(p, dst)
                outputs.append(dst)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        tmp_dir.rmdir()
        log.info(
            "%s finished in %.1fs, %d file(s)", self.step_cls.__name__, time.time() - t0, len(outputs)
        )
        return outputs

    def _call(self, inp: str, kwargs: dict[str, Any]) -> Any:
        try:
            return self.step_cls.call(inp, configure_log=False, **kwargs)
        except Exception as exc:
            # Older stpipe (< jwst 1.20) rejects the unknown `configure_log` parameter
            # during config validation, i.e. before any processing happened.
            if "configure_log" in str(exc):
                log.debug("stpipe does not accept configure_log; retrying without it")
                return self.step_cls.call(inp, **kwargs)
            raise


def _close(result: Any) -> None:
    """Close whatever a pipeline returned (DataModel, ModelContainer, list, ...)."""
    if result is None:
        return
    items = result if isinstance(result, (list, tuple)) else [result]
    for item in items:
        close = getattr(item, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover
                pass
