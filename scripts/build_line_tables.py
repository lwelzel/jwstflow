"""Rebuild the line entries of the bundled feature tables from primary line lists.

Run it (with network access) from the repository root::

    python scripts/build_line_tables.py            # report only
    python scripts/build_line_tables.py --write    # update the ECSV tables in place

What it does, per table of ``src/jwstflow/refdata/spectral_features/``:

* ``gas_lines.ecsv`` and ``gas_lines_extended.ecsv`` -- the *lines* are
  regenerated, the bands (Q-branches, CO bands, water and OH forests) are kept
  as they are. ``gas_lines`` is the set data steps mask with ``features: all``:
  its lines keep their ids and only have their wavelengths verified. Every
  other regenerated line goes to ``gas_lines_extended``, an opt-in table that
  QA figures annotate and no data step uses unless it is named:

  - H I recombination lines from the Rydberg formula (vacuum, reduced-mass
    Rydberg constant of hydrogen), series n = 3 ... 8;
  - H2 quadrupole lines (all branches of v = 0-0, 1-0 and 2-1) from HITRAN;
  - He I and the forbidden lines of the abundant atoms and ions from the NIST
    Atomic Spectra Database (Ritz vacuum wavelengths); a curated line is kept
    only when NIST has a line of the right kind next to it.

* ``photospheric_lines.ecsv`` -- every atomic line is replaced by the NIST Ritz
  vacuum wavelength of the strongest transition next to it, the CO bandheads by
  the heads of the R branches in HITRAN.

Existing ids are kept wherever a regenerated line coincides with an old one, so
workflows that select features by id keep working. The script prints every
change it makes; nothing is written without ``--write``.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np

TABLES = Path(__file__).resolve().parents[1] / "src" / "jwstflow" / "refdata" / "spectral_features"
RYDBERG_H = 109677.58340280356          # cm^-1, R_inf * m_p / (m_p + m_e)
GREEK = {1: "alpha", 2: "beta", 3: "gamma", 4: "delta"}
SERIES = {3: ("Pa", range(4, 13)), 4: ("Br", range(5, 26)), 5: ("Pf", range(6, 26)), 6: ("Hu", range(7, 26)),
          7: ("", range(8, 15)), 8: ("", range(9, 15)), 10: ("", (16,))}
#: H2: the quadrupole lines of HITRAN of these (upper, lower) vibrational levels, per branch up
#: to the lower J that is still seen in shocked or fluorescent gas (higher members only clutter).
H2_BANDS = {(0, 0): {"S": 17}, (1, 0): {"S": 11, "Q": 9, "O": 9}, (2, 1): {"S": 9, "Q": 7, "O": 7}}
WAVE_RANGE = (0.95, 28.6)
#: (NIST spectrum, guess [um], forbidden?) -- the curated inventory; NIST decides the wavelength.
ATOMIC = [
    ("He I", 1.0833, False), ("He I", 1.7007, False), ("He I", 1.8690, False), ("He I", 2.0587, False),
    ("He I", 2.1126, False), ("He I", 2.1138, False), ("He I", 4.2960, False),
    ("C I", 0.9853, True), ("S II", 1.0323, True), ("S II", 1.0339, True), ("N I", 1.0401, True), ("P II", 1.1886, True),
    ("Fe II", 1.2570, True), ("Fe II", 1.3209, True), ("Fe II", 1.5339, True), ("Fe II", 1.5999, True),
    ("Fe II", 1.6440, True), ("Fe II", 1.6642, True), ("Fe II", 1.6773, True), ("Fe II", 1.7116, True),
    ("Fe II", 1.7454, True), ("Fe II", 1.7976, True), ("Fe II", 1.8005, True), ("Fe II", 1.8099, True),
    ("Fe II", 4.0763, True), ("Fe II", 4.1150, True), ("Fe II", 4.8891, True), ("Fe II", 5.3402, True),
    ("Fe II", 6.7213, True), ("Fe II", 17.936, True), ("Fe II", 24.519, True), ("Fe II", 25.988, True),
    ("Fe I", 24.042, True), ("Fe III", 22.925, True), ("Ni II", 6.6360, True), ("Ni II", 10.682, True),
    ("Ni II", 12.729, True), ("Ni III", 7.3492, True), ("Co II", 10.523, True), ("Ca IV", 3.2067, True),
    ("Mg IV", 4.4867, True), ("Ar VI", 4.5295, True), ("K III", 4.6180, True), ("Mg V", 5.6099, True),
    ("Ar II", 6.9853, True), ("Na III", 7.3177, True), ("Ne VI", 7.6524, True), ("Ar V", 7.9016, True),
    ("Ar III", 8.9914, True), ("S IV", 10.5105, True), ("Cl I", 11.3333, True), ("Ne II", 12.8136, True),
    ("Ar V", 13.102, True), ("Ne V", 14.3217, True), ("Cl II", 14.3678, True), ("Ne III", 15.5551, True),
    ("P III", 17.885, True), ("S III", 18.713, True), ("Ar III", 21.830, True), ("Ne V", 24.3175, True),
    ("S I", 25.249, True), ("O IV", 25.890, True),
]
CO_BANDHEADS = {"co_3-0": (5, 1, 3, 0), "co_4-1": (5, 1, 4, 1), "co_5-2": (5, 1, 5, 2), "co_6-3": (5, 1, 6, 3),
                "co_7-4": (5, 1, 7, 4), "co_8-5": (5, 1, 8, 5), "co_2-0": (5, 1, 2, 0), "co_3-1": (5, 1, 3, 1),
                "co_4-2": (5, 1, 4, 2), "co_5-3": (5, 1, 5, 3), "co_6-4": (5, 1, 6, 4), "13co_2-0": (5, 2, 2, 0)}


def text(value: object) -> str:
    """A table cell as text; empty for a masked one (which astropy prints as ``--``)."""
    return "" if np.ma.is_masked(value) or str(value).strip() == "--" else str(value)


def number(value: object) -> float:
    """The leading number of a NIST cell (which may carry flags or be masked); NaN without one."""
    match = re.search(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", str(value))
    return float(match.group()) if match and not np.ma.is_masked(value) else float("nan")


def hydrogen() -> list[dict]:
    rows = []
    for lower, (name, uppers) in SERIES.items():
        for upper in uppers:
            wave = 1e4 / (RYDBERG_H * (1.0 / lower**2 - 1.0 / upper**2))
            if not WAVE_RANGE[0] <= wave <= WAVE_RANGE[1]:
                continue
            if name and upper - lower in GREEK:
                label, key = f"H I {name}-{GREEK[upper - lower]}", f"hi_{name.lower()}{GREEK[upper - lower][0]}"
            elif name:
                label, key = f"H I {name} {upper}-{lower}", f"hi_{name.lower()}{upper}"
            else:
                label, key = f"H I {upper}-{lower}", f"hi_{upper}-{lower}"
            rows.append({"id": key, "label": label, "species": "HI", "wavelength_um": round(wave, 5),
                         "transition": f"n={upper}-{lower}", "source": "Rydberg formula (vacuum, R_H)"})
    return rows


def molecular_hydrogen() -> list[dict]:
    import astropy.units as u
    from astroquery.hitran import Hitran

    table = Hitran.query_lines(molecule_number=45, isotopologue_number=1, min_frequency=300 / u.cm,
                               max_frequency=10600 / u.cm)
    best: dict[tuple, tuple[float, float]] = {}
    for row in table:
        match = re.match(r"\s*([OQS])\s*(\d+)q", str(row["local_lower_quanta"]))
        if not match:
            continue
        key = (int(row["global_upper_quanta"]), int(row["global_lower_quanta"]), match.group(1), int(match.group(2)))
        if key not in best or float(row["a"]) > best[key][1]:
            best[key] = (float(row["nu"]), float(row["a"]))
    rows = []
    for (v_up, v_lo, branch, j), (nu, _a) in sorted(best.items()):
        if j <= H2_BANDS.get((v_up, v_lo), {}).get(branch, -1):
            wave = 1e4 / nu
            if WAVE_RANGE[0] <= wave <= WAVE_RANGE[1]:
                rows.append({"id": f"h2_{v_up}-{v_lo}_{branch.lower()}{j}", "label": f"H2 {v_up}-{v_lo} {branch}({j})",
                             "species": "H2", "wavelength_um": round(wave, 5), "transition": f"v={v_up}-{v_lo} {branch}({j})",
                             "source": "HITRAN2020 (Gordon et al. 2022), H2 quadrupole lines"})
    return rows


def nist_line(spectrum: str, guess: float, forbidden: bool) -> tuple[float, str] | None:
    """The NIST line next to ``guess`` [um]: for a forbidden line the nearest M1/E2 transition, for
    an allowed one the strongest transition within the blend. ``(Ritz vacuum wavelength, type)``."""
    import astropy.units as u
    from astroquery.nist import Nist

    # a forbidden line stands alone; an allowed one is the strongest component of a narrow blend
    half = max(0.0015, 6e-4 * guess) if forbidden else max(0.0006, 2.5e-4 * guess)
    try:
        table = Nist.query((guess - half) * u.um, (guess + half) * u.um, linename=spectrum, wavelength_type="vacuum")
    except Exception as exc:  # noqa: BLE001 -- "no lines" comes back as an exception
        print(f"  {spectrum} {guess}: NIST query failed ({str(exc)[:60]})")
        return None
    candidates = []
    for row in table:
        wave = number(row["Ritz"])
        if not np.isfinite(wave):
            wave = number(row["Observed"])
        kind = "" if np.ma.is_masked(row["Type"]) else str(row["Type"]).strip()
        if not np.isfinite(wave) or forbidden != bool(kind):   # forbidden lines carry a type (M1, E2), allowed ones none
            continue
        strength = number(row["Aki"])
        candidates.append((wave, kind or "E1", strength if np.isfinite(strength) else 0.0))
    if not candidates:
        return None
    if forbidden:
        wave, kind, _ = min(candidates, key=lambda c: abs(c[0] - guess))
    else:
        wave, kind, _ = max(candidates, key=lambda c: c[2])
    return wave, kind


def atoms() -> list[dict]:
    rows = []
    for spectrum, guess, forbidden in ATOMIC:
        found = nist_line(spectrum, guess, forbidden)
        if found is None:
            print(f"  {spectrum} {guess}: no {'forbidden ' if forbidden else ''}NIST line next to it, left out")
            continue
        wave, kind = found
        element, stage = spectrum.split()
        digits = 3 if wave >= 1 else 4
        label = f"[{spectrum}] {wave:.{digits}f}" if forbidden else f"{spectrum} {wave:.{digits}f}"
        rows.append({"id": f"{element.lower()}{stage.lower()}_{wave:.{digits}f}", "label": label, "species": element + stage,
                     "wavelength_um": round(wave, 5), "transition": kind,
                     "source": "NIST ASD (Kramida et al.), Ritz vacuum wavelength"})
    return rows


def co_bandheads() -> dict[str, float]:
    import astropy.units as u
    from astroquery.hitran import Hitran

    heads = {}
    guesses = {"co_3-0": 1.5582, "co_4-1": 1.5780, "co_5-2": 1.5982, "co_6-3": 1.6189, "co_7-4": 1.6401, "co_8-5": 1.6618,
               "co_2-0": 2.2935, "co_3-1": 2.3227, "co_4-2": 2.3525, "co_5-3": 2.3829, "co_6-4": 2.4142, "13co_2-0": 2.3448}
    for key, (molecule, isotopologue, v_up, v_lo) in CO_BANDHEADS.items():
        centre = 1e4 / guesses[key]
        try:
            table = Hitran.query_lines(molecule_number=molecule, isotopologue_number=isotopologue,
                                       min_frequency=(centre - 12) / u.cm, max_frequency=(centre + 12) / u.cm)
        except Exception as exc:  # noqa: BLE001
            print(f"  {key}: HITRAN query failed ({str(exc)[:60]})")
            continue
        lines = sorted((int(str(r["local_lower_quanta"]).split()[1]), float(r["nu"])) for r in table
                       if int(r["global_upper_quanta"]) == v_up and int(r["global_lower_quanta"]) == v_lo
                       and str(r["local_lower_quanta"]).strip().startswith("R"))
        # a head is the turning point of the R branch: only trust it when the list goes past it
        if lines and max(lines, key=lambda line: line[1])[0] < lines[-1][0]:
            heads[key] = round(1e4 / max(nu for _j, nu in lines), 5)
        else:
            print(f"  {key}: the R branch in HITRAN stops before its head; quoted value kept")
    return heads


def rebuild_gas(write: bool) -> None:
    from astropy.table import Table

    path = TABLES / "gas_lines.ecsv"
    table = Table.read(path, format="ascii.ecsv")
    old_lines = [dict(zip(table.colnames, row)) for row in table if row["kind"] == "line"]
    kept = table[table["kind"] != "line"]
    new = hydrogen() + molecular_hydrogen() + atoms()
    seen = set()
    for row in new:
        match = [o for o in old_lines if o["species"] == row["species"] and abs(float(o["wavelength_um"]) - row["wavelength_um"])
                 < max(0.004, 4e-4 * row["wavelength_um"])]
        if match:
            old = match[0]
            if abs(float(old["wavelength_um"]) - row["wavelength_um"]) > 2e-4:
                print(f"  corrected {old['id']}: {float(old['wavelength_um']):.4f} -> {row['wavelength_um']:.5f}")
            row["id"], row["notes"] = str(old["id"]), text(old["notes"])
            seen.add(str(old["id"]))
        else:
            print(f"  added {row['id']}: {row['label']} at {row['wavelength_um']:.5f}")
    for old in old_lines:
        if str(old["id"]) not in seen:
            print(f"  kept as quoted {old['id']} ({old['label']}, {float(old['wavelength_um']):.4f}): not in the line lists queried")
            new.append({**{k: old[k] for k in ("id", "label", "species", "wavelength_um", "transition", "source")},
                        "notes": (text(old["notes"]).split("; wavelength as quoted")[0]
                                  + "; wavelength as quoted in the literature (not in the NIST line table)").strip("; ")})
    defaults = {"kind": "line", "wave_min_um": float("nan"), "wave_max_um": float("nan"), "notes": ""}
    core_ids = {str(o["id"]) for o in old_lines}
    ordered = sorted(new, key=lambda r: float(r["wavelength_um"]))

    def records_of(rows: list[dict]) -> list[tuple]:
        return [tuple(row.get(c, defaults.get(c, "")) for c in table.colnames) for row in rows]

    extended = Table(rows=records_of([r for r in ordered if r["id"] not in core_ids]), names=table.colnames)
    out = Table(rows=records_of([r for r in ordered if r["id"] in core_ids])
                + [tuple(text(row[c]) if c == "notes" else row[c] for c in table.colnames) for row in kept],
                names=table.colnames)
    for name in table.colnames:
        extended[name].unit, extended[name].description = table[name].unit, table[name].description
    extended.meta = type(table.meta)(table.meta)
    extended.meta.update(dataset="gas_lines_extended", version="2026.11.0",
                         description="Further atomic and molecular gas-phase lines in the 1-28 um range, beyond the set of "
                                     "gas_lines: the higher H I series members, the full H2 1-0 and 2-1 branches, He I, and "
                                     "the forbidden lines of less abundant or more highly ionised species",
                         provenance="Generated by scripts/build_line_tables.py from primary lists: H I from the Rydberg "
                                    "formula, H2 from HITRAN2020, He I and forbidden lines from NIST ASD (Ritz); all vacuum.",
                         notes="An opt-in dataset: it is not part of a bare `features: all` (so it never changes what data "
                               "steps mask); QA figures annotate it together with gas_lines.",
                         usage="Select in a workflow with `features: {gas_lines_extended: all | [ids or species]}`.")
    for name in table.colnames:
        out[name].unit, out[name].description = table[name].unit, table[name].description
    out.meta = table.meta
    out.meta["version"] = "2026.11.0"
    out.meta["provenance"] = ("Wavelengths of the lines verified by scripts/build_line_tables.py against primary lists: H I "
                              "from the Rydberg formula, H2 from HITRAN2020, He I and forbidden lines from NIST ASD (Ritz); "
                              "all vacuum. Bands and Q-branches as compiled from the references below. Further lines: "
                              "gas_lines_extended.")
    print(f"gas_lines: {len(old_lines)} lines (wavelengths verified), {len(kept)} bands kept; "
          f"gas_lines_extended: {len(extended)} lines")
    if write:
        out.write(path, format="ascii.ecsv", overwrite=True)
        extended.write(TABLES / "gas_lines_extended.ecsv", format="ascii.ecsv", overwrite=True)


def rebuild_photosphere(write: bool) -> None:
    from astropy.table import Table

    path = TABLES / "photospheric_lines.ecsv"
    table = Table.read(path, format="ascii.ecsv")
    heads = co_bandheads()
    for row in table:
        key, old = str(row["id"]), float(row["wavelength_um"])
        head = next((v for k, v in heads.items() if key.startswith(k + "_")), None)
        match = re.match(r"([A-Z][a-z]?)(I+)$", str(row["species"]))
        if head is not None:
            new, source = head, "HITRAN2020 (turning point of the R branch)"
        elif match:
            found = nist_line(f"{match.group(1)} {match.group(2)}", old, False)
            if found is None:
                print(f"  {key}: no NIST line next to {old}; left as it was")
                continue
            new, source = round(found[0], 5), "NIST ASD (Kramida et al.), Ritz vacuum wavelength of the strongest component"
        else:
            continue
        if abs(new - old) > 1.5e-4:
            print(f"  {key}: {old:.4f} -> {new:.5f}")
        row["wavelength_um"], row["source"] = new, source
    table.meta["version"] = "2026.11.1"
    table.meta["provenance"] = ("Atomic lines: NIST ASD Ritz vacuum wavelengths of the strongest transition of each blend; CO "
                                "bandheads: heads of the R branches in HITRAN2020; SiO bandheads as commonly quoted "
                                "(approximate). Regenerated by scripts/build_line_tables.py.")
    if write:
        table.write(path, format="ascii.ecsv", overwrite=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true", help="update the ECSV tables in place")
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    rebuild_gas(args.write)
    rebuild_photosphere(args.write)
    sys.exit(0)
