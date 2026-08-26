"""NIRSpec-specific helper steps."""

from __future__ import annotations

import logging
from pathlib import Path

from ..steps.base import RunContext

log = logging.getLogger(__name__)


def fix_msa_metafile(inputs: list[Path], ctx: RunContext, *, search: str = "raw") -> list[Path]:
    """Point ``MSAMETFL`` at the absolute path of the MSA metadata file.

    ``calwebb_spec2`` opens the file named in the ``MSAMETFL`` keyword; a bare
    file name is only found if it sits in the working directory. Because
    jwstflow keeps downloads in ``<root>/raw`` and rate files in
    ``<root>/detector1``, this step rewrites the keyword in place to the
    absolute path of the ``*_msa.fits`` file found in the ``search`` stage
    directory. Idempotent: files already pointing at an existing absolute path
    are left alone.
    """
    from astropy.io import fits

    out: list[Path] = []
    search_dir = ctx.dir_of(search) if search in ctx.stage_dirs else ctx.raw_dir
    for inp in inputs:
        with fits.open(inp, mode="update") as hdul:
            hdr = hdul[0].header
            current = hdr.get("MSAMETFL")
            if not current:
                log.warning("%s has no MSAMETFL keyword; skipped", inp.name)
                continue
            cur_path = Path(current)
            if cur_path.is_absolute() and cur_path.exists():
                out.append(inp)
                continue
            candidates = list(search_dir.rglob(cur_path.name))
            if not candidates:
                raise FileNotFoundError(f"{cur_path.name} (from {inp.name}) not found under {search_dir}")
            hdr["MSAMETFL"] = str(candidates[0].resolve())
            hdul.flush()
            log.info("%s: MSAMETFL -> %s", inp.name, candidates[0].name)
        out.append(inp)
    return out
