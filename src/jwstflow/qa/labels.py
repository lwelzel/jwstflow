"""The text and color of feature labels: species colors, typeset formulae, the
series a line belongs to.

The search for lines in a spectrum (:func:`detect_lines` and its constants) lives in
:mod:`jwstflow.spectra.lines`; its names are importable from here as well.
"""

from __future__ import annotations

from typing import Any

from ..spectra.lines import (
    DETECT_SIGMA,
    EDGE_SAMPLES,
    GAP_STEP,
    IDENTIFY_VELOCITY,
    UNIDENTIFIED_IN_BAND,
    UNIDENTIFIED_SIGMA,
    detect_lines,
)
from .style import UM

#: One color per group of gas species, sampled from the standard colormap (cmasher *torch*,
#: with *ember* and *flamingo* for the molecules), the same wherever the species appears.
SPECIES_COLORS: dict[str, str] = {
    "H2": "#052675", "noble gases": "#472d97", "metals": "#75418f", "He": "#a45087", "H": "#d35c73",
    "CO": "#fa7651", "H2O": "#ec7801", "organics": "#de4891", "other": "0.35",
}
#: The species of each group: molecular hydrogen, the noble gases (Ne, Ar), the other atoms
#: and ions (forbidden lines of Fe, Ni, S, ...), helium, hydrogen, the carbon oxides, water
#: and OH, and the hydrocarbons / nitriles of inner-disk chemistry.
SPECIES_GROUPS: dict[str, tuple[str, ...]] = {
    "H": ("HI",), "He": ("HeI", "HeII"), "H2": ("H2", "HD"), "CO": ("CO", "13CO", "C18O", "CO2", "13CO2"),
    "H2O": ("H2O", "OH"), "noble gases": ("NeII", "NeIII", "NeV", "NeVI", "ArII", "ArIII", "ArV", "ArVI"),
    "organics": ("C2H2", "13CCH2", "HCN", "CH4", "C2H4", "C2H6", "C6H6", "HC3N", "C3H4", "C4H2", "CH3", "NH3"),
}
#: The photospheric lines are one family of greens (cmasher *emerald*, *jungle*, *swamp*,
#: *toxic*; dashed stems, labels under the spectrum), a shade per element, so that the group
#: reads as one and never as a gas line.
PHOTOSPHERE_COLORS: dict[str, str] = {
    "Na": "#129247", "Ca": "#29b884", "Mg": "#689a30", "Al": "#127566", "K": "#36a063", "Fe": "#31692b",
    "Ti": "#117a1a", "Si": "#348f43", "CO": "#099063", "13CO": "#099063", "SiO": "#1c6769", "other": "#129247",
}
#: The band rows below the plot: one color per group of solid-state features (cmasher
#: *freeze*, *amethyst*, *lavender*, *sapphire*, *sunburst*; silicates grey).
BAND_COLORS: dict[str, str] = {
    "H2O ice": "#2295db", "CO2 ice": "#773fe1", "CO ice": "#219f8b", "other ices": "#4b579c", "silicates": "0.55",
    "PAH": "#da5d16",
}


def species_color(species: str) -> str:
    """The color of a gas species (:data:`SPECIES_COLORS`): its group's; an atom or ion outside
    the named groups is a metal, any other molecule grey."""
    import re

    for group, members in SPECIES_GROUPS.items():
        if species in members:
            return SPECIES_COLORS[group]
    return SPECIES_COLORS["metals"] if re.fullmatch(r"[A-Z][a-z]?[IVX]+", species) else SPECIES_COLORS["other"]


def _photosphere_color(species: str) -> str:
    import re

    element = re.match(r"\d*[A-Z][a-z]?", species)
    key = species if species in PHOTOSPHERE_COLORS else (element.group() if element else "other")
    return PHOTOSPHERE_COLORS.get(key, PHOTOSPHERE_COLORS["other"])


def _band_group(entry: dict[str, Any]) -> tuple[str, str]:
    """``(legend name, color)`` of a broad feature."""
    species = entry["species"]
    if entry["cls"] == "ice":
        if species in ("H2O", "CO2", "13CO2", "CO", "13CO"):
            name = species.replace("13", "") + " ice"
        else:
            name = "silicates" if species.lower().startswith("silicate") else "other ices"
        return name, BAND_COLORS[name]
    if species == "PAH":
        return "PAH bands", BAND_COLORS["PAH"]
    return "gas bands (outline; color: species)", species_color(species)


#: The wavelength of an unidentified peak is printed with this many decimals [um].
UNIDENTIFIED_DECIMALS = 3


def _series(label: str) -> tuple[str, str] | None:
    """``(series, tick text)`` of a line that belongs to a series: the H I series by their
    lower level (ticks: the Greek letter or the upper level), the branches of the H2 bands
    (ticks: J). None for every other line."""
    import re

    match = re.fullmatch(r"H I (Pa|Br|Pf|Hu)[- ](?:(alpha|beta|gamma|delta)|(\d+)-\d+)", label)
    if match:
        return f"H I {match.group(1)}", rf"$\{match.group(2)}$" if match.group(2) else match.group(3)
    match = re.fullmatch(r"H I (\d+)-(\d+)", label)
    if match:
        return f"H I n={match.group(2)}", match.group(1)
    match = re.fullmatch(r"H2 (\d+-\d+) ([SQO])\((\d+)\)", label)
    if match:
        return f"H2 {match.group(1)} {match.group(2)}", match.group(3)
    return None


def tex_label(text: str) -> str:
    r"""A feature label typeset: chemical formulae with sub- and superscripts (``13CO2`` ->
    ``$^{13}$CO$_2$``, ``NH4+`` -> ``NH$_4^+$``), Greek letters of the hydrogen series
    (``Br-alpha`` -> ``Br$\alpha$``), ``nu2`` and ``um`` as symbols."""
    import re

    def word(token: str) -> str:
        match = re.fullmatch(r"(\d{1,2})?((?:[A-Z][a-z]?\d*)+)([+-])?", token)
        if not match or re.fullmatch(r"[IVX]+", token):
            return token
        isotope, body, charge = match.groups()
        if not (isotope or charge or re.search(r"[A-Za-z]\d", body)):
            return token
        body = re.sub(r"(\d+)", r"$_{\1}$", body)
        return (f"$^{{{isotope}}}$" if isotope else "") + body + (f"$^{{{charge}}}$" if charge else "")

    text = re.sub(r"-(alpha|beta|gamma|delta|epsilon)\b", lambda m: rf"$\{m.group(1)}$", text)
    text = re.sub(r"\bnu(\d)\b", r"$\\nu_{\1}$", text)
    text = re.sub(r"(?<=\d) um\b", lambda _m: f" {UM}", text)
    return " ".join(word(token) for token in text.split(" "))


def _band_text(entry: dict[str, Any]) -> str:
    """The label of a band: the formula of an ice (no "ice": the row says that), the name of
    the others without what their mark already says."""
    species, label = entry["species"], entry["label"]
    if entry["cls"] == "ice":
        if species.lower().startswith("silicate"):
            return "silicates"
        return tex_label(label.replace(" ice", "")) if species == "mixed" else tex_label(species)
    return tex_label(label.replace(" (gas)", "").replace(" gas ", " "))


def _photosphere_label(label: str) -> str:
    """A photospheric line's label: the species (and band), without the wavelength its
    position already shows (``Na I 2.2062`` -> ``Na I``, ``CO 2-0 bandhead`` -> ``CO 2-0``)."""
    words = [w for w in label.split() if w != "bandhead"]
    try:
        float(words[-1])
        words = words[:-1] or words
    except (ValueError, IndexError):
        pass
    return " ".join(words) or label


def _unidentified_label(wavelength: float) -> str:
    """The label of a peak no catalogue line explains: ``?`` and its wavelength [um]
    (``? [1.234]``)."""
    return f"? [{wavelength:.{UNIDENTIFIED_DECIMALS}f}]"


def _member_span(ticks: list[str]) -> str:
    """The members of a series that share one label: a range where they follow one another
    (``17–25``), else the runs of consecutive members (``17–18, 21, 24–25``) -- the label
    names exactly the lines its stems point to."""
    if not all(t.isdigit() for t in ticks):
        return f"{ticks[0]}\N{EN DASH}{ticks[-1]}"
    numbers = sorted({int(t) for t in ticks})
    runs: list[list[int]] = [[numbers[0]]]
    for n in numbers[1:]:
        if n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    return ", ".join(f"{r[0]}\N{EN DASH}{r[-1]}" if len(r) > 1 else f"{r[0]}" for r in runs)


def _line_label(label: str) -> str:
    """A gas line's label in the plot: what identifies it, without the words its mark already
    says (``C2H6 Q-branch`` -> ``C2H6``, ``H2O cold diagnostic 23.8-24.0`` -> ``H2O cold``)."""
    import re

    text = re.sub(r"\s+diagnostic.*$", "", label).replace(" Q-branch", "").replace(" (gas)", "").replace("hot-band", "hot band")
    return text.strip() or label


__all__ = [
    "BAND_COLORS",
    "DETECT_SIGMA",
    "EDGE_SAMPLES",
    "GAP_STEP",
    "IDENTIFY_VELOCITY",
    "PHOTOSPHERE_COLORS",
    "SPECIES_COLORS",
    "SPECIES_GROUPS",
    "UNIDENTIFIED_DECIMALS",
    "UNIDENTIFIED_IN_BAND",
    "UNIDENTIFIED_SIGMA",
    "detect_lines",
    "species_color",
    "tex_label",
]
