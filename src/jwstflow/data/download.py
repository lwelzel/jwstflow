"""Download JWST data from MAST.

Two backends behind one function, :func:`download`:

* ``astroquery`` (default): ``astroquery.mast.Observations``. Query by
  program / observation numbers / instrument mode, filter products, download
  into a flat directory, skipping files already present.
* ``jwst_mast_query``: thin wrapper around STScI's ``jwst_download.py`` for
  people who already use it (arguments are passed through).

Both return the list of local files, which the ``raw`` pseudo-stage exposes to
the workflow.
"""

from __future__ import annotations

import fnmatch
import re
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..config.schema import DownloadConfig

log = logging.getLogger(__name__)


def download(cfg: DownloadConfig, dest: Path, *, dry_run: bool = False) -> list[Path]:
    """Download the requested products; if MAST is unreachable but the raw directory already
    holds files, warn and continue with those (an archive outage must not stop a rerun)."""
    dest = Path(dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        if cfg.backend == "astroquery":
            return _download_astroquery(cfg, dest, dry_run=dry_run)
        return _download_jwst_mast_query(cfg, dest, dry_run=dry_run)
    except Exception as exc:  # RemoteServiceError, connection errors, HTTP errors, ...
        existing = sorted(dest.glob("*.fits"))
        if not existing:
            raise RuntimeError(f"MAST query/download failed and {dest} holds no data yet: {exc}") from exc
        log.warning("MAST query/download failed (%s: %s); continuing with the %d file(s) already in %s "
                    "(use --skip-download to avoid contacting MAST at all)",
                    type(exc).__name__, str(exc).splitlines()[0][:160], len(existing), dest)
        return existing


# ---------------------------------------------------------------------------
# astroquery backend
# ---------------------------------------------------------------------------


# MAST's `instrument_name` values are "<INSTRUMENT>/<MODE>" with MAST's own mode names;
# accept the common instrument-side names as well.
MAST_MODE_ALIASES: dict[str, str] = {
    "MRS": "IFU",          # MIRI MRS observations are listed as MIRI/IFU
    "IMAGING": "IMAGE", "IMAGER": "IMAGE",
    "LRS": "SLIT",         # MIRI LRS fixed slit; slitless is SLITLESS
    "LRS-SLITLESS": "SLITLESS",
    "FS": "SLIT", "FIXEDSLIT": "SLIT",
    "MOS": "MSA",
    "BOTS": "SLIT",
    "CORONAGRAPHY": "CORON",
}


def mast_instrument_name(instrument: str, mode: str) -> str:
    return f"{instrument.upper()}/{MAST_MODE_ALIASES.get(mode.upper(), mode.upper())}"


def _instrument_names(cfg: DownloadConfig) -> list[str] | str:
    if not cfg.modes:
        return f"{cfg.instrument}*"
    return sorted({mast_instrument_name(cfg.instrument, m) for m in cfg.modes})


def fits_is_complete(path: Path) -> bool:
    """True when the file holds every byte its FITS structure declares.

    Interrupted transfers leave truncated files that only fail much later, deep
    inside detector1 ("cannot reshape array of size ..."); this catches them at
    download time. Lazy loading makes the check cheap (headers only).
    """
    try:
        from astropy.io import fits

        with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
            info = hdul[-1].fileinfo()  # indexing [-1] walks every header without reading data
            expected = ((info["datLoc"] + max(info["datSpan"], 0) + 2879) // 2880) * 2880
        return path.stat().st_size >= expected
    except Exception as exc:
        log.debug("%s failed the FITS completeness check: %s", path.name, exc)
        return False


def ensure_complete_fits(paths: list[Path], *, delete: bool = True) -> list[Path]:
    """Return the truncated/corrupt files among ``paths`` (and delete them so a retry re-downloads)."""
    bad = [p for p in paths if p.suffix == ".fits" and p.exists() and not fits_is_complete(p)]
    for p in bad:
        log.error("%s is truncated or corrupt (%d bytes)%s", p.name, p.stat().st_size, "; deleting for re-download" if delete else "")
        if delete:
            p.unlink()
    return bad


MAST_LOG = "mast_observations.json"


def record_observations(dest: Path, obs: Any) -> None:
    """Keep the MAST observation rows (obs_id, target, instrument, ...) next to the raw files.

    The obs_id carries the DMS target id (``jw01751-o006_t010_nirspec_...``) which
    is not in the FITS headers but is needed for DMS-compliant level-3 product names.
    """
    keep = ("obs_id", "target_name", "instrument_name", "filters", "proposal_id", "obs_collection", "s_ra", "s_dec", "t_exptime")
    rows = []
    for r in obs:
        rows.append({k: (None if r[k] is None else (float(r[k]) if hasattr(r[k], "dtype") and r[k].dtype.kind == "f" else str(r[k])))
                     for k in keep if k in obs.colnames})
    path = dest / MAST_LOG
    existing: list[dict[str, Any]] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = []
    ids = {e.get("obs_id") for e in existing}
    existing += [r for r in rows if r.get("obs_id") not in ids]
    path.write_text(json.dumps(existing, indent=2))


_OBSID = re.compile(r"^jw(\d{5})-o(\d{3})_((?:t\d{3})|(?:s\d{5}))_")


def target_ids_from_log(raw_dir: Path) -> dict[tuple[str, str], str]:
    """(program, observation) -> DMS target/source id parsed from recorded MAST obs_ids."""
    path = Path(raw_dir) / MAST_LOG
    if not path.exists():
        return {}
    out: dict[tuple[str, str], str] = {}
    try:
        rows = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    for r in rows:
        m = _OBSID.match(str(r.get("obs_id", "")))
        if m:
            out[(m.group(1), m.group(2))] = m.group(3)
    return out


def build_query(cfg: DownloadConfig) -> dict[str, object]:
    """The ``Observations.query_criteria`` keyword arguments (exposed for testing)."""
    crit: dict[str, object] = {
        "obs_collection": "JWST",
        # MAST stores the proposal id without leading zeros, but be lenient.
        "proposal_id": sorted({str(cfg.program), f"{cfg.program:05d}"}),
        "instrument_name": _instrument_names(cfg),
    }
    if cfg.observations:
        crit["obs_id"] = [f"jw{cfg.program:05d}-o{o:03d}*" for o in cfg.observations]
    if cfg.exclusive_only:
        crit["dataRights"] = "EXCLUSIVE_ACCESS"
    crit.update(cfg.query_extra)
    return crit


def _download_astroquery(cfg: DownloadConfig, dest: Path, *, dry_run: bool) -> list[Path]:
    try:
        from astroquery.mast import Observations
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("the astroquery backend needs `pip install astroquery`") from exc

    token = (os.environ.get(cfg.token_env) or "").strip()
    if token:
        try:
            Observations.login(token=token)
            log.info("logged in to MAST using $%s", cfg.token_env)
        except Exception as exc:  # astroquery.exceptions.LoginError, network errors, ...
            if cfg.exclusive_only:
                raise RuntimeError(f"MAST login with ${cfg.token_env} failed: {exc}") from exc
            log.warning(
                "MAST login with $%s failed (%s); continuing anonymously, which is fine for "
                "public data. Check that the variable holds the token you expect "
                "(`jwstflow validate` shows which env files were loaded and what shadows them).",
                cfg.token_env, str(exc).splitlines()[0],
            )
    elif cfg.exclusive_only:
        raise RuntimeError(f"exclusive_only needs a MAST token in ${cfg.token_env}")
    else:
        log.info("no $%s set; downloading anonymously (public data only)", cfg.token_env)

    crit = build_query(cfg)
    log.info("querying MAST: %s", crit)
    obs = Observations.query_criteria(**crit)
    if len(obs) == 0:
        log.warning("MAST query returned no observations")
        return sorted(dest.glob("*.fits"))
    log.info("%d observation(s) found: %s", len(obs), ", ".join(str(o) for o in obs["obs_id"][:12]))
    record_observations(dest, obs)
    if cfg.observations:
        found = {m.group(2) for o in obs["obs_id"] if (m := _OBSID.match(str(o)))}
        missing = [o for o in cfg.observations if f"{o:03d}" not in found]
        if missing:
            log.warning("requested observation(s) %s returned no %s products: check the observation numbers "
                        "in the proposal (backgrounds are separate observations) or drop the `modes` filter",
                        missing, cfg.instrument)

    products = Observations.get_product_list(obs)
    log.info("%d product file(s) listed before filtering", len(products))
    filt: dict[str, object] = {"productSubGroupDescription": [p.upper() for p in cfg.products]}
    if cfg.calib_level:
        filt["calib_level"] = list(cfg.calib_level)
    filt.update(cfg.product_filters_extra)
    products = Observations.filter_products(products, **filt)
    log.info("%d after filter_products(%s)", len(products), filt)
    if cfg.filename_patterns:
        keep = [
            any(fnmatch.fnmatch(str(f), pat) for pat in cfg.filename_patterns)
            for f in products["productFilename"]
        ]
        products = products[keep]
        log.info("%d after filename_patterns %s", len(products), cfg.filename_patterns)
    # unique filenames only (MAST lists the same file under several obs)
    _, idx = _unique(list(products["productFilename"]))
    products = products[idx]
    log.info("%d product file(s) selected", len(products))

    wanted = [dest / str(f) for f in products["productFilename"]]
    ensure_complete_fits([p for p in wanted if p.exists()])  # a truncated leftover must be re-fetched, not trusted
    todo = [i for i, p in enumerate(wanted) if not p.exists()]
    log.info("%d already present, %d to download", len(wanted) - len(todo), len(todo))
    if dry_run or not todo:
        return sorted(p for p in wanted if p.exists())

    for attempt in (1, 2):
        manifest = Observations.download_products(
            products[todo], download_dir=str(dest), flat=True, cache=True, mrp_only=False
        )
        for r in manifest:
            if str(r["Status"]).upper() != "COMPLETE":
                log.error("download failed: %s", r["Local Path"])
        bad = ensure_complete_fits([wanted[i] for i in todo])
        todo = [i for i in todo if not wanted[i].exists()]
        if not todo:
            break
        if attempt == 1:
            log.warning("%d file(s) arrived truncated; retrying their download once", len(bad) or len(todo))
    if todo:
        raise RuntimeError(f"{len(todo)} file(s) could not be downloaded intact, e.g. {wanted[todo[0]].name}; "
                           "rerun later or fetch them manually into the raw directory")
    return sorted(p for p in wanted if p.exists())


def _unique(values: list[str]) -> tuple[list[str], list[int]]:
    seen: set[str] = set()
    vals: list[str] = []
    idx: list[int] = []
    for i, v in enumerate(values):
        if v not in seen:
            seen.add(v)
            vals.append(v)
            idx.append(i)
    return vals, idx


# ---------------------------------------------------------------------------
# jwst_mast_query backend
# ---------------------------------------------------------------------------


def _download_jwst_mast_query(cfg: DownloadConfig, dest: Path, *, dry_run: bool) -> list[Path]:
    exe = shutil.which("jwst_download.py") or shutil.which("jwst_download")
    if exe is None:  # pragma: no cover
        raise RuntimeError(
            "jwst_mast_query backend: `jwst_download.py` not found on PATH "
            "(pip install jwst_mast_query)"
        )
    cmd = [
        exe,
        "--instrument",
        cfg.instrument.lower(),
        "--propID",
        str(cfg.program),
        "--filetypes",
        *[p.lower() for p in cfg.products],
        "--outrootdir",
        str(dest),
        "--skip_propID2outsubdir",
        *cfg.extra_args,
    ]
    if cfg.observations:
        cmd += ["--obsnums", *[str(o) for o in cfg.observations]]
    log.info("running: %s", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)
    return sorted(p for p in dest.rglob("*.fits"))


REFERENCE_LOG = "provenance.json"
_IMAGING_PRODUCTS = {"I2D", "CAT", "SEGM", "PHOT", "WHTLT"}


def reference_subdir(product_type: str, calib_level: int, instrument_mode: str = "") -> str:
    """Where a MAST product goes inside ``mast_reference/<run>/``, mirroring jwstflow's layout:
    ``stage1/calwebb_detector1``, ``stage2/calwebb_spec2|image2``, ``stage3/calwebb_spec3|image3``."""
    ptype = product_type.upper()
    imaging = ptype in _IMAGING_PRODUCTS or instrument_mode.upper().endswith("/IMAGE")
    if ptype == "UNCAL" or calib_level <= 1 and ptype not in ("RATE", "RATEINTS"):
        return "raw"
    if ptype in ("RATE", "RATEINTS"):
        return "stage1/calwebb_detector1"
    if calib_level >= 3:
        return "stage3/calwebb_image3" if imaging else "stage3/calwebb_spec3"
    return "stage2/calwebb_image2" if imaging else "stage2/calwebb_spec2"


def download_reference_products(cfg: DownloadConfig, dest: Path, *, dry_run: bool = False) -> list[Path]:
    """Fetch MAST's own calibrated products (opt-in) into ``dest`` (= ``<target>/mast_reference/<run>``).

    Files keep their archive names and are placed in the stage/step directory
    jwstflow would use for the same product type, so a jwstflow product and its
    MAST reference sit at the same relative path under the run and the
    reference tree. ``provenance.json`` lists, per file, the calibration
    software version and CRDS context from the header plus the download time.
    """
    from astropy.io import fits
    from astroquery.mast import Observations

    dest = Path(dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    obs = Observations.query_criteria(**build_query(cfg))
    if len(obs) == 0:
        log.warning("MAST reference products: query returned no observations")
        return []
    mode_by_obsid = {str(r["obs_id"]): str(r["instrument_name"]) for r in obs}
    products = Observations.get_product_list(obs)
    filt = {"productSubGroupDescription": [p.upper() for p in cfg.reference_product_types]}
    products = Observations.filter_products(products, **filt)
    # `filename_patterns` selects raw exposures (e.g. "*mirifu*"); level-3 products are named
    # after the observation (jw01751-o010_t005_miri_ch1-short_s3d.fits) and must not be filtered
    # by it. The instrument-mode restriction of the query already excludes other detectors.
    _, idx = _unique(list(products["productFilename"]))
    products = products[idx]
    levels = sorted({int(r["calib_level"]) for r in products})
    log.info("MAST reference products: %d file(s) at calibration level(s) %s", len(products), levels)
    targets: dict[str, Path] = {}
    for row in products:
        name = str(row["productFilename"])
        sub = reference_subdir(str(row["productSubGroupDescription"]), int(row["calib_level"]),
                               mode_by_obsid.get(str(row["obs_id"]), ""))
        targets[name] = dest / sub / name
    todo = [i for i, f in enumerate(products["productFilename"]) if not targets[str(f)].exists()]
    log.info("MAST reference products: %d file(s) selected, %d to download into %s", len(products), len(todo), dest)
    if todo and not dry_run:
        staging = dest / ".incoming"
        staging.mkdir(exist_ok=True)
        manifest = Observations.download_products(products[todo], download_dir=str(staging), flat=True, cache=True, mrp_only=False)
        for r in manifest:
            local = Path(str(r["Local Path"]))
            if str(r["Status"]).upper() != "COMPLETE" or not local.exists():
                log.error("MAST reference download failed: %s", local.name)
                continue
            final = targets[local.name]
            final.parent.mkdir(parents=True, exist_ok=True)
            local.replace(final)
        shutil.rmtree(staging, ignore_errors=True)
        ensure_complete_fits([t for t in targets.values() if t.exists()])  # truncated ones vanish; the next run refetches
    files = sorted(p for p in targets.values() if p.exists())
    prov_path = dest / REFERENCE_LOG
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    from datetime import datetime, timezone

    for f in files:
        rel = str(f.relative_to(dest))
        if rel in prov:
            continue
        try:
            hdr = fits.getheader(f)
            prov[rel] = {"origin": "MAST archive product (reference, not a jwstflow output)", "cal_ver": hdr.get("CAL_VER"),
                         "crds_ctx": hdr.get("CRDS_CTX"), "date_obs": hdr.get("DATE-OBS"),
                         "downloaded": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        except OSError as exc:
            log.warning("cannot read %s: %s", f.name, exc)
    prov_path.write_text(json.dumps(prov, indent=2, sort_keys=True))
    return files
