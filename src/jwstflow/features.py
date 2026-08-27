"""Spectral feature datasets (gas lines, PAH bands, ice bands) shipped with jwstflow.

The datasets are ECSV tables in ``<data dir>/spectral_features/`` (data lives
outside the code: ``$JWSTFLOW_DATA_DIR``, else ``<project root>/data``, else the
``data/`` directory of the jwstflow checkout) with one
row per feature (``id``, ``label``, ``species``, ``kind``, ``wavelength_um``,
``wave_min_um``, ``wave_max_um``, ``transition``, ``source``, ``notes``) and
file-level metadata (description, references, version). Steps select
features through a small config grammar::

    features: all                                  # every feature of every dataset
    features: [h2_0-0_s1, neii_12.814, pah_7.7]    # ids ...
    features: [H2, PAH]                            # ... or species
    features: {gas_lines: [H2, NeII], pah_bands: all, ice_bands: [co2_4.27]}
    features: {ice_bands: all, wave_min: 2.8, wave_max: 5.3}  # restrict to a wavelength range

``select_features`` turns any of these into ``[(id, label, wmin, wmax), ...]``.
Lines (``kind == 'line'``) get their mask width from ``line_width_um`` or, when
``resolving_power`` is given, from ``n_resolution_elements`` resolution elements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

DATASETS = ("gas_lines", "pah_bands", "ice_bands")


def data_dir() -> Path:
    """Directory holding jwstflow's data products (spectral features, ...).

    Order: ``$JWSTFLOW_DATA_DIR``; ``<project root>/data`` of the working
    directory; the ``data/`` directory next to the jwstflow source tree (an
    editable/checkout install). Data is kept out of the package on purpose.
    """
    from .project import find_project_root

    candidates = []
    if os.environ.get("JWSTFLOW_DATA_DIR"):
        candidates.append(Path(os.environ["JWSTFLOW_DATA_DIR"]).expanduser())
    candidates.append(find_project_root(Path.cwd()) / "data")
    candidates.append(Path(__file__).resolve().parents[2] / "data")
    for c in candidates:
        if (c / "spectral_features").is_dir():
            return c
    raise FileNotFoundError(
        "jwstflow data directory not found; set JWSTFLOW_DATA_DIR or keep a `data/` directory "
        f"in the project root (looked at {[str(c) for c in candidates]})"
    )


@dataclass(frozen=True)
class Feature:
    id: str
    label: str
    species: str
    kind: str
    wavelength: float | None
    wave_min: float | None
    wave_max: float | None
    dataset: str
    source: str = ""
    notes: str = ""

    def window(self, line_width_um: float = 0.02, resolving_power: float | None = None,
               n_resolution_elements: float = 3.0) -> tuple[float, float]:
        """Wavelength range to mask/select for this feature."""
        if self.kind == "band" and self.wave_min is not None and self.wave_max is not None:
            return self.wave_min, self.wave_max
        if self.wavelength is None:
            raise ValueError(f"feature {self.id} has neither edges nor a central wavelength")
        if self.wave_min is not None and self.wave_max is not None:
            return self.wave_min, self.wave_max
        half = line_width_um
        if resolving_power:
            half = 0.5 * n_resolution_elements * self.wavelength / resolving_power
        return self.wavelength - half, self.wavelength + half


@lru_cache(maxsize=None)
def load_dataset(name: str) -> tuple[Feature, ...]:
    """Features of one bundled dataset (or of an external ECSV path)."""
    from astropy.table import Table

    if name in DATASETS:
        table = Table.read(data_dir() / "spectral_features" / f"{name}.ecsv", format="ascii.ecsv")
    else:
        path = Path(name).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"unknown feature dataset {name!r}; bundled: {DATASETS}")
        table = Table.read(path, format="ascii.ecsv")
        name = path.stem
    out = []
    for row in table:
        out.append(Feature(
            id=str(row["id"]), label=str(row["label"]), species=str(row["species"]), kind=str(row["kind"]),
            wavelength=_float(row["wavelength_um"]), wave_min=_float(row["wave_min_um"]), wave_max=_float(row["wave_max_um"]),
            dataset=name, source=str(row["source"]), notes=str(row["notes"]),
        ))
    return tuple(out)


def dataset_metadata(name: str) -> dict[str, Any]:
    from astropy.table import Table

    return dict(Table.read(data_dir() / "spectral_features" / f"{name}.ecsv", format="ascii.ecsv").meta)


def all_features(datasets: tuple[str, ...] = DATASETS) -> list[Feature]:
    return [f for d in datasets for f in load_dataset(d)]


def select_features(spec: Any, *, wave_min: float | None = None, wave_max: float | None = None) -> list[Feature]:
    """Resolve the config grammar described in the module docstring."""
    if spec is None or spec is False:
        return []
    if isinstance(spec, dict):
        spec = dict(spec)
        wave_min = spec.pop("wave_min", wave_min)
        wave_max = spec.pop("wave_max", wave_max)
        chosen: list[Feature] = []
        for dataset, selector in spec.items():
            chosen += _select_from(load_dataset(str(dataset)), selector, dataset)
    else:
        chosen = _select_from(all_features(), spec, "any dataset")
    return [f for f in chosen if _in_range(f, wave_min, wave_max)]


def feature_windows(spec: Any, **window_kwargs: Any) -> list[tuple[float, float]]:
    """Convenience: the (min, max) windows of the selected features."""
    return [f.window(**window_kwargs) for f in select_features(spec)]


def _select_from(features: tuple[Feature, ...] | list[Feature], selector: Any, where: str) -> list[Feature]:
    if selector in ("all", True, "*"):
        return list(features)
    if isinstance(selector, str):
        selector = [selector]
    wanted = [str(x) for x in selector]
    out: list[Feature] = []
    for w in wanted:
        hits = [f for f in features if f.id == w or f.species.lower() == w.lower() or f.label.lower() == w.lower()]
        if not hits:
            ids = sorted({f.species for f in features})
            raise ValueError(f"feature {w!r} not found in {where}; ids are like {features[0].id!r}, species: {ids}")
        out += [h for h in hits if h not in out]
    return out


def _in_range(f: Feature, lo: float | None, hi: float | None) -> bool:
    wmin, wmax = f.window()
    if lo is not None and wmax < lo:
        return False
    if hi is not None and wmin > hi:
        return False
    return True


def _float(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v
