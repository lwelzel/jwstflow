"""Generate the README overview figure: a simplified, annotated jwstflow workflow.

Not a render of a real workflow -- a designed advertisement figure showing the
package's capabilities on one simple DAG. Regenerate with:

    uv run python docs/make_overview_figure.py

Kept out of the installable package and of GitHub's "Download ZIP" archives
(see .gitattributes); it only exists in full git clones.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from jwstflow.dagviz import LEVEL_COLORS

GREY = "#6f6f6f"
ANN = "#555555"
NODES: dict[str, tuple[tuple[float, float], float, float]] = {}  # name -> (center, half_w, half_h)


def node(ax, name, x, y, text, *, kind="step", color=None):
    lines = text.split("\n")
    chars = max(len(line) for line in lines)
    if kind == "step":
        box = dict(boxstyle="round,pad=0.45", fc=color or "#dddddd", ec="#444444")
        font = dict(fontsize=11)
        half_w, half_h = 0.115 * chars / 2 + 0.34, 0.26 * len(lines) / 2 + 0.24
    elif kind in ("data", "ext"):
        box = dict(boxstyle="square,pad=0.38", fc="#f7f7f7" if kind == "data" else "white",
                   ec="#888888", ls="-" if kind == "data" else "--")
        font = dict(fontsize=10, family="monospace")
        half_w, half_h = 0.14 * chars / 2 + 0.28, 0.26 * len(lines) / 2 + 0.2
    else:  # source / yaml
        box = dict(boxstyle="round4,pad=0.42", fc="#f5f0e1", ec="#8a7d55")
        font = dict(fontsize=10.5)
        half_w, half_h = 0.115 * chars / 2 + 0.32, 0.26 * len(lines) / 2 + 0.24
    ax.text(x, y, text, ha="center", va="center", bbox=box, zorder=3, **font)
    NODES[name] = ((x, y), half_w, half_h)
    return name


def border(name, towards):
    """Point on the node's box border in the direction of `towards`."""
    (cx, cy), hw, hh = NODES[name]
    dx, dy = towards[0] - cx, towards[1] - cy
    if dx == dy == 0:
        return (cx, cy)
    t = min(hw / abs(dx) if dx else float("inf"), hh / abs(dy) if dy else float("inf"))
    return (cx + dx * t, cy + dy * t)


def edge(ax, a, b, *, rad=0.0, ls="-"):
    pa, pb = border(a, NODES[b][0]), border(b, NODES[a][0])
    ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle="-|>", mutation_scale=13, lw=1.15, color=GREY,
                                 linestyle=ls, shrinkA=1.5, shrinkB=1.5,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def note(ax, x, y, text, target=None, *, ha="center", rad=-0.25):
    ax.text(x, y, text, ha=ha, va="center", fontsize=9.5, style="italic", color=ANN,
            linespacing=1.4, zorder=3)
    if target is not None:
        anchor = border(target, (x, y)) if isinstance(target, str) else target
        start_y = y + (0.4 if anchor[1] > y else -0.4)
        ax.add_patch(FancyArrowPatch((x, start_y), anchor, arrowstyle="-", lw=0.75, color="#b0b0b0",
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=3, zorder=1))


def main() -> None:
    fig, ax = plt.subplots(figsize=(17.4, 5.9))
    Y = 0.55  # the main spine

    node(ax, "yaml", 0.0, Y, "workflow.yaml\n$ jwstflow run", kind="source")
    node(ax, "mast", 2.75, Y, "MAST\nprogram / obs", kind="source")
    node(ax, "uncal", 5.1, Y, "*_uncal.fits", kind="data")
    node(ax, "det1", 7.35, Y, "detector1", color=LEVEL_COLORS[1])
    node(ax, "spec2", 9.35, Y, "spec2", color=LEVEL_COLORS[2])
    node(ax, "spec3", 11.3, Y, "spec3", color=LEVEL_COLORS[3])
    node(ax, "s3d", 13.5, Y, "*_s3d.fits", kind="data")
    node(ax, "custom", 15.95, Y, "extract_source\n(your step)", color=LEVEL_COLORS[4])
    node(ax, "s1d", 18.3, Y, "*_s1d.fits", kind="data")
    node(ax, "qa", 15.95, -1.35, "compare with the\narchive's reduction", color=LEVEL_COLORS["qa"])
    node(ax, "other", 18.3, -2.6, "*_s1d.fits\nfrom another run", kind="ext")
    node(ax, "combine", 20.45, -0.85, "combine\n(level 4)", color=LEVEL_COLORS[4])
    node(ax, "spectrum", 22.55, -0.85, "1-28 um\nspectrum", kind="data")

    for a, b in [("yaml", "mast"), ("mast", "uncal"), ("uncal", "det1"), ("det1", "spec2"),
                 ("spec2", "spec3"), ("spec3", "s3d"), ("s3d", "custom"), ("custom", "s1d")]:
        edge(ax, a, b)
    edge(ax, "s3d", "qa", rad=0.18)
    edge(ax, "s1d", "combine", rad=0.12)
    edge(ax, "other", "combine", rad=-0.12)
    edge(ax, "combine", "spectrum")

    # ---- capability annotations (each anchored to what it describes)
    note(ax, 0.0, 2.15, "declarative & reproducible:\none YAML, one command", "yaml")
    note(ax, 3.7, -1.05, "verified downloads,\nCRDS pinned & prefetched", "mast", rad=0.25)
    ax.plot([6.5, 6.5, 12.15, 12.15], [1.28, 1.43, 1.43, 1.28], color="#bbbbbb", lw=1.1)
    note(ax, 9.3, 2.15, "the official STScI pipeline is the engine --\nassociations & DMS-compliant names built for you")
    note(ax, 6.15, -1.8, "checkpointed parallel execution:\ninterrupt any time, reruns resume", (8.35, 0.4), rad=0.08)
    note(ax, 16.9, 2.15, "custom steps plug in:\ndeclared, validated, unit-tested", "custom", rad=0.25)
    note(ax, 15.95, -2.2, "automatic QA against\nMAST's own products")   # directly beneath its node: no leader needed
    note(ax, 22.35, -2.45, "multi-instrument level-4 runs\nread their sibling runs", "combine", rad=-0.25)
    tree = "reductions/<target>/<run>/\n  stage1/ stage2/ stage3/ stage4/ qa/"
    ax.text(11.4, -2.5, tree, ha="center", va="center", fontsize=9.5, family="monospace", color=ANN,
            bbox=dict(boxstyle="square,pad=0.42", fc="#fbfbfb", ec="#cccccc", ls=":"), zorder=3)
    note(ax, 8.75, -2.5, "structured, self-describing\noutput tree per target & run", ha="right")
    ax.add_patch(FancyArrowPatch((11.4, -2.05), border("s3d", (11.4, -2.05)), arrowstyle="-", lw=0.75,
                                 color="#b0b0b0", linestyle=(0, (2, 3)), connectionstyle="arc3,rad=0.12",
                                 shrinkA=2, shrinkB=3, zorder=1))   # ... products land here

    ax.set_xlim(-1.7, 24.0)
    ax.set_ylim(-3.45, 2.95)
    ax.axis("off")
    fig.tight_layout(pad=0.35)
    out = Path(__file__).parent
    for name in ("overview_dag.svg", "overview_dag.pdf", "overview_dag.png"):
        fig.savefig(out / name, dpi=180)
    print("wrote", ", ".join(str(out / n) for n in ("overview_dag.svg", "overview_dag.pdf", "overview_dag.png")))


if __name__ == "__main__":
    main()
