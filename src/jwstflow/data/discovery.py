"""Find input files and read the handful of header keywords the orchestrator needs.

Header reads are cached in ``<root>/.jwstflow/headers.json`` keyed by
path+size+mtime, so re-planning a run with thousands of files is instant.
Association JSON files are also understood (they are inputs to *3 pipelines).
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Keywords cached from the primary header. Add more here if a filter needs them.
HEADER_KEYWORDS: tuple[str, ...] = (
    "PROGRAM",
    "OBSERVTN",
    "VISIT",
    "VISITGRP",
    "SEQ_ID",
    "ACT_ID",
    "EXPOSURE",
    "OBS_ID",
    "VISIT_ID",
    "INSTRUME",
    "DETECTOR",
    "EXP_TYPE",
    "FILTER",
    "GRATING",
    "CHANNEL",
    "BAND",
    "SUBARRAY",
    "TARGPROP",
    "TARGNAME",
    "TARG_RA",
    "TARG_DEC",
    "BKGDTARG",
    "IS_IMPRT",
    "TSOVISIT",
    "PATT_NUM",
    "NUMDTHPT",
    "PATTTYPE",
    "NINTS",
    "NGROUPS",
    "READPATT",
    "DATE-OBS",
    "TIME-OBS",
    "EXPSTART",
    "EXPMID",
    "EFFEXPTM",
    "MSAMETFL",
    "CAL_VER",
    "CRDS_CTX",
    "DATAMODL",
)


@dataclass
class FileRecord:
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        """File name without extension and without the product suffix (``_uncal`` etc.)."""
        stem = self.path.name
        for ext in (".fits", ".json", ".asdf"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
        return re.sub(r"_[a-z0-9]+$", "", stem) if "_" in stem else stem

    def get(self, key: str, default: Any = None) -> Any:
        return self.meta.get(key.upper(), default)


class HeaderCache:
    """Tiny JSON cache of primary-header keywords."""

    def __init__(self, path: Path | None):
        self.path = path
        self._data: dict[str, dict[str, Any]] = {}
        self._dirty = False
        if path is not None and path.exists():
            try:
                self._data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                self._data = {}

    def _key(self, p: Path) -> str:
        st = p.stat()
        return f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}"

    def get(self, p: Path) -> dict[str, Any]:
        key = self._key(p)
        if key not in self._data:
            self._data[key] = read_metadata(p)
            self._dirty = True
        return self._data[key]

    def save(self) -> None:
        if self.path is None or not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data))
        tmp.replace(self.path)
        self._dirty = False


def read_metadata(p: Path) -> dict[str, Any]:
    """Primary-header keywords of a FITS file, or a summary of an association file."""
    if p.suffix.lower() == ".json":
        try:
            asn = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return {"KIND": "asn", "ASN_TYPE": None}
        products = asn.get("products", [])
        members = [m.get("expname") for pr in products for m in pr.get("members", [])]
        return {
            "KIND": "asn",
            "ASN_TYPE": asn.get("asn_type"),
            "ASN_RULE": asn.get("asn_rule"),
            "PROGRAM": asn.get("program"),
            "PRODUCTS": [pr.get("name") for pr in products],
            "MEMBERS": members,
        }
    if p.suffix.lower() not in (".fits", ".fit", ".fts"):
        return {"KIND": "file"}
    try:
        from astropy.io import fits

        hdr = fits.getheader(p, 0)
    except Exception as exc:  # unreadable file: keep going, filters will exclude it
        log.warning("cannot read header of %s: %s", p, exc)
        return {"KIND": "fits", "ERROR": str(exc)}
    meta: dict[str, Any] = {"KIND": "fits"}
    for k in HEADER_KEYWORDS:
        if k in hdr:
            v = hdr[k]
            meta[k] = v if isinstance(v, (str, int, float, bool)) or v is None else str(v)
    return meta


# ---------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------


def match_filters(meta: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    """Evaluate header filters. Missing keywords never match (except ``null`` filters)."""
    for key, want in filters.items():
        have = meta.get(key.upper())
        if isinstance(want, list):
            if not any(_eq(have, w) for w in want):
                return False
        elif not _eq(have, want):
            return False
    return True


def _eq(have: Any, want: Any) -> bool:
    if want is None:
        return have is None
    if isinstance(want, str) and want.startswith("regex:"):
        return have is not None and re.search(want[6:], str(have)) is not None
    if isinstance(want, bool):
        # A missing boolean keyword (e.g. no BKGDTARG card) counts as False.
        return (_as_bool(have) or False) == want
    if isinstance(have, str) and isinstance(want, str):
        return have.strip().upper() == want.strip().upper()
    return have == want


def _as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip().upper() in ("T", "TRUE", "1", "YES")
    return bool(v)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def discover(
    directory: Path,
    pattern: str = "*.fits",
    *,
    recursive: bool = False,
    exclude: Iterable[str] = (),
    filters: Mapping[str, Any] | None = None,
    cache: HeaderCache | None = None,
) -> list[FileRecord]:
    """Files in ``directory`` matching ``pattern`` and ``filters``, sorted by name."""
    directory = Path(directory).expanduser()
    if not directory.exists():
        return []
    paths = sorted(directory.rglob(pattern) if recursive else directory.glob(pattern))
    exclude = list(exclude)
    cache = cache or HeaderCache(None)
    out: list[FileRecord] = []
    for p in paths:
        if not p.is_file() or any(part.startswith(".tmp-") for part in p.parts):
            continue
        if any(fnmatch.fnmatch(p.name, ex) for ex in exclude):
            continue
        meta = cache.get(p) if (filters or p.suffix.lower() in (".fits", ".json")) else {}
        if filters and not match_filters(meta, filters):
            continue
        out.append(FileRecord(p, meta))
    return out


def fingerprint(paths: Iterable[Path], mode: str = "fast") -> dict[str, str]:
    """Input fingerprints for checkpointing."""
    import hashlib

    out: dict[str, str] = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            out[str(p)] = "missing"
            continue
        st = p.stat()
        if mode == "content":
            h = hashlib.sha1()
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[str(p)] = "sha1:" + h.hexdigest()
        else:
            out[str(p)] = f"{st.st_size}:{st.st_mtime_ns}"
    return out
