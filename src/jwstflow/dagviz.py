"""Render a workflow's DAG as a figure: data products (per file pattern) -> steps -> products.

The graph is built from the configuration alone, so it can be drawn before any
data exists. Three node kinds: data nodes (file patterns, grouped by the stage
that produces them; the raw node carries the download query), step nodes
(coloured by calibration level, dashed when disabled), and external nodes for
inputs taken from sibling runs. Edges follow ``inputs:`` and ``depends_on:``.

Rendered with matplotlib (already a jwstflow dependency): nodes are placed by
longest-path layering with a barycenter sweep to minimise edge crossings --
pure Python, no binaries, nothing to install beyond ``uv sync``. The ``.dot``
source is written next to the figure as a portable text artifact (diffable,
and renderable with external tools if anyone wants a different look).

Used automatically at the start of every run (``workflow_graph: true``, the
default) and by ``jwstflow graph``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config.schema import RAW_STAGE, Config, StageConfig
from .steps.base import LEVEL_DIRS

log = logging.getLogger(__name__)

#: primary products of the official pipelines, for sink nodes nobody consumes
PIPELINE_PRODUCTS: dict[str, tuple[str, ...]] = {
    "calwebb_detector1": ("*_rate.fits",),
    "calwebb_dark": ("*_dark.fits",),
    "calwebb_spec2": ("*_cal.fits",),
    "calwebb_image2": ("*_cal.fits", "*_i2d.fits"),
    "calwebb_spec3": ("*_s3d.fits", "*_x1d.fits", "*_crf.fits"),
    "calwebb_image3": ("*_i2d.fits", "*_cat.ecsv"),
    "calwebb_tso3": ("*_x1dints.fits", "*_whtlt.ecsv"),
}

LEVEL_COLORS = {1: "#cfe3f5", 2: "#d3efd8", 3: "#fbe6c2", 4: "#e9dcf5", "qa": "#f2d3d8"}


@dataclass
class Graph:
    """Minimal DAG: node id -> attributes, plus (src, dst, style) edges."""

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)

    def add_node(self, nid: str, **attrs: Any) -> str:
        self.nodes.setdefault(nid, attrs)
        return nid

    def add_edge(self, src: str, dst: str, style: str = "solid") -> None:
        if (src, dst, style) not in self.edges:
            self.edges.append((src, dst, style))


def build_graph(cfg: Config) -> Graph:
    """The workflow DAG with per-pattern data nodes between the steps."""
    g = Graph()
    consumed: dict[str, set[str]] = {}  # producer stage -> patterns some consumer reads

    def data_node(producer: str, pattern: str, external: str | None = None) -> str:
        nid = f"data::{external or producer}::{pattern}"
        label = pattern if external is None else f"{external}\n{pattern}"
        g.add_node(nid, kind="data", label=label, external=external is not None)
        return nid

    if cfg.download is not None and cfg.download.enabled:
        dl = cfg.download
        query = f"MAST {dl.instrument} {'/'.join(dl.modes or [])}\nprogram {dl.program}, obs {dl.observations or 'all'}"
        g.add_node("download", kind="source", label=query)
    for stage in cfg.stages:
        g.add_node(f"step::{stage.name}", kind="step", label=stage.name, sublabel=stage.step,
                   level=stage.level, enabled=stage.enabled)
    for stage in cfg.stages:
        sid = f"step::{stage.name}"
        for spec in stage.inputs:
            pattern = spec.pattern or "*"
            if spec.path is not None:
                src = data_node("path", pattern, external=str(spec.path))
            elif spec.run is not None:
                src = data_node(spec.stage or RAW_STAGE, pattern, external=f"run {spec.run}: {spec.stage}")
            else:
                producer = spec.stage or RAW_STAGE
                src = data_node(producer, pattern)
                consumed.setdefault(producer, set()).add(pattern)
                if producer == RAW_STAGE:
                    if "download" in g.nodes:
                        g.add_edge("download", src)
                else:
                    g.add_edge(f"step::{producer}", src)
            g.add_edge(src, sid)
        for dep in stage.depends_on:
            g.add_edge(f"step::{dep}", sid, style="dotted")
    # sink products: declared outputs (custom steps) or the pipelines' primary products
    for stage in cfg.stages:
        for pattern in _declared_products(stage):
            if not any(_pattern_overlap(pattern, c) for c in consumed.get(stage.name, ())):
                g.add_edge(f"step::{stage.name}", data_node(stage.name, pattern))
    return g


def _declared_products(stage: StageConfig) -> tuple[str, ...]:
    if stage.name.split("-")[0] in PIPELINE_PRODUCTS:
        return PIPELINE_PRODUCTS[stage.name.split("-")[0]]
    try:
        from .steps.base import resolve_target

        outputs = tuple(getattr(resolve_target(stage.step), "outputs", ()) or ())
    except Exception:
        outputs = ()
    return tuple(f"*_{sfx}.fits" for sfx in outputs)


def _pattern_overlap(product: str, consumer: str) -> bool:
    """Cheap test whether a consumer glob reads the product glob (shared suffix token)."""
    token = product.replace("*", "").replace(".fits", "").strip("_")
    return token != "" and token in consumer


# --------------------------------------------------------------------------- dot


def to_dot(cfg: Config, g: Graph | None = None) -> str:
    g = g or build_graph(cfg)
    lines = [
        "digraph jwstflow {",
        '  rankdir=LR; splines=true; nodesep=0.3; ranksep=0.55; bgcolor="white";',
        '  graph [fontname="Helvetica"];',
        '  node [fontname="Helvetica", fontsize=11];',
        '  edge [color="#666666", arrowsize=0.7];',
    ]
    for nid, a in g.nodes.items():
        name = _dot_id(nid)
        if a["kind"] == "step":
            color = LEVEL_COLORS.get(a.get("level"), "#dddddd")
            style = "filled" if a.get("enabled", True) else "filled,dashed"
            extra = "" if a.get("enabled", True) else " (disabled)"
            sub = a.get("sublabel", "")
            label = f"{a['label']}{extra}" + (f"\\n{_small(sub)}" if sub and sub != a["label"] else "")
            lines.append(f'  {name} [shape=box, style="{style},rounded", fillcolor="{color}", label="{label}"];')
        elif a["kind"] == "source":
            lines.append(f'  {name} [shape=cylinder, style=filled, fillcolor="#f5f0e1", label="{_esc(a["label"])}"];')
        else:
            fill = "#f7f7f7" if not a.get("external") else "#ffffff"
            style = "filled" if not a.get("external") else "filled,dashed"
            lines.append(f'  {name} [shape=note, style="{style}", fillcolor="{fill}", fontname="Courier", label="{_esc(a["label"])}"];')
    for src, dst, style in g.edges:
        attr = "" if style == "solid" else f' [style={style}]'
        lines.append(f"  {_dot_id(src)} -> {_dot_id(dst)}{attr};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _dot_id(nid: str) -> str:
    return '"' + nid.replace('"', "'") + '"'


def _esc(text: str) -> str:
    return str(text).replace('"', "'").replace("\n", "\\n")


def _small(text: str) -> str:
    return _esc(text if len(str(text)) < 42 else str(text)[:39] + "...")


# --------------------------------------------------------------------------- rendering


def render(cfg: Config, out_dir: Path, *, formats: tuple[str, ...] = ("pdf", )) -> list[Path]:
    """Write ``<run>_dag.dot`` plus the rendered figure in each requested format
    (anything matplotlib can save: svg, pdf, png, ...); returns the files written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.run}_dag"
    g = build_graph(cfg)
    written = [out_dir / f"{stem}.dot"]
    written[0].write_text(to_dot(cfg, g))
    return written + _render_matplotlib(cfg, g, out_dir / stem, formats)


def _layers(g: Graph) -> dict[str, int]:
    """Longest-path depth of every node (the x coordinate of the fallback layout)."""
    incoming: dict[str, list[str]] = {n: [] for n in g.nodes}
    for src, dst, _ in g.edges:
        incoming[dst].append(src)
    depth: dict[str, int] = {}

    def visit(node: str, seen: tuple[str, ...] = ()) -> int:
        if node in depth:
            return depth[node]
        if node in seen:  # cycles cannot happen in a validated config; stay safe anyway
            return 0
        depth[node] = 1 + max((visit(p, seen + (node,)) for p in incoming[node]), default=-1)
        return depth[node]

    for node in g.nodes:
        visit(node)
    return depth


def _sugiyama(g: Graph) -> tuple[dict[str, tuple[float, float]], dict[tuple[str, str, str], list[tuple[float, float]]]]:
    """Layered (Sugiyama) layout: positions for the real nodes and waypoint paths per edge.

    Longest-path layering; dummy waypoints split edges spanning several layers so
    they route between rows; barycenter sweeps order every column to minimise
    crossings; median-alignment sweeps straighten chains; column x positions
    follow the widest label per column so boxes never collide.
    """
    depth = _layers(g)
    succ: dict[str, list[str]] = {}
    chains: dict[tuple[str, str, str], list[str]] = {}
    dummy_depth: dict[str, int] = dict(depth)
    for k, (src, dst, style) in enumerate(g.edges):
        path = [src]
        for step_depth in range(depth[src] + 1, depth[dst]):
            d = f"dummy::{k}::{step_depth}"
            dummy_depth[d] = step_depth
            path.append(d)
        path.append(dst)
        chains[(src, dst, style)] = path
        for a, b in zip(path, path[1:]):
            succ.setdefault(a, []).append(b)
    neighbours: dict[str, set[str]] = {n: set() for n in dummy_depth}
    for a, bs in succ.items():
        for b in bs:
            neighbours[a].add(b)
            neighbours[b].add(a)
    columns: dict[int, list[str]] = {}
    for node, d in dummy_depth.items():
        columns.setdefault(d, []).append(node)
    for d in columns:
        columns[d].sort()
    order = {n: float(i) for nodes in columns.values() for i, n in enumerate(nodes)}
    for _ in range(4):  # crossing minimisation over the expanded graph
        for d in sorted(columns) + sorted(columns, reverse=True):
            columns[d].sort(key=lambda n: _mean([order[m] for m in neighbours[n] if dummy_depth[m] != d], order[n]))
            for i, n in enumerate(columns[d]):
                order[n] = float(i)
    # vertical coordinates: pack each column symmetrically around the spine (y = 0),
    # so single-node columns -- the main chain -- sit on one straight line
    sep = {n: 0.55 if n.startswith("dummy::") else (1.25 if "\n" in str(g.nodes.get(n, {}).get("label", "")) else 1.0)
           for n in dummy_depth}
    y: dict[str, float] = {}
    for d in sorted(columns):
        offsets = [0.0]
        for a, b in zip(columns[d], columns[d][1:]):
            offsets.append(offsets[-1] + (sep[a] + sep[b]) / 2)
        centre = (offsets[0] + offsets[-1]) / 2
        for n, off in zip(columns[d], offsets):
            y[n] = off - centre
    xs: dict[int, float] = {}
    x = 0.0
    for d in sorted(columns):  # horizontal position from the widest label of each column
        widest = max((_label_width(g, n) for n in columns[d]), default=1.0)
        xs[d] = x + widest / 2
        x += widest + 0.9
    pos = {n: (xs[dummy_depth[n]], -y[n]) for n in dummy_depth}
    paths = {edge: [pos[n] for n in chain] for edge, chain in chains.items()}
    return {n: pos[n] for n in g.nodes}, paths


def _mean(values: list[float], default: float) -> float:
    return sum(values) / len(values) if values else default


def _label_width(g: Graph, node: str) -> float:
    if node.startswith("dummy::"):
        return 0.3
    label = str(g.nodes[node].get("label", ""))
    chars = max((len(line) for line in label.splitlines()), default=4)
    return 0.35 + chars * (0.135 if g.nodes[node]["kind"] == "data" else 0.115)


def _render_matplotlib(cfg: Config, g: Graph, base: Path, formats: tuple[str, ...]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, Patch

    pos, paths = _sugiyama(g)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    fig, ax = plt.subplots(figsize=(2.0 + (max(xs) - min(xs)) * 0.62, 2.4 + (max(ys) - min(ys)) * 0.78))
    for (src, dst, style), waypoints in paths.items():
        ls = ":" if style == "dotted" else "-"
        if len(waypoints) > 2:
            wx, wy = zip(*waypoints)
            ax.plot(wx[:-1], wy[:-1], color="#777777", lw=1.0, ls=ls, solid_capstyle="round", zorder=1)
        ax.add_patch(FancyArrowPatch(waypoints[-2], waypoints[-1], arrowstyle="-|>", mutation_scale=11,
                                     lw=1.0, color="#777777", linestyle=ls, shrinkA=0 if len(waypoints) > 2 else 14,
                                     shrinkB=14, zorder=1))
    for nid, a in g.nodes.items():
        x, y = pos[nid]
        if a["kind"] == "step":
            face = LEVEL_COLORS.get(a.get("level"), "#dddddd")
            label = a["label"] + ("" if a.get("enabled", True) else "\n(disabled)")
            box = dict(boxstyle="round,pad=0.45", fc=face, ec="#444444", ls="-" if a.get("enabled", True) else "--")
        elif a["kind"] == "source":
            label, box = a["label"], dict(boxstyle="round4,pad=0.4", fc="#f5f0e1", ec="#8a7d55")
        else:
            label = a["label"]
            box = dict(boxstyle="square,pad=0.35", fc="#f7f7f7" if not a.get("external") else "white",
                       ec="#888888", ls="--" if a.get("external") else "-")
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                family="monospace" if a["kind"] == "data" else "sans-serif", bbox=box, zorder=2)
    levels_used = sorted({a["level"] for a in g.nodes.values() if a["kind"] == "step"}, key=str)
    handles = [Patch(fc=LEVEL_COLORS.get(lv, "#dddddd"), ec="#444444",
                     label=f"{LEVEL_DIRS.get(lv, lv)} step") for lv in levels_used]
    handles.append(Patch(fc="#f7f7f7", ec="#888888", label="data products (file pattern)"))
    if any(a["kind"] == "source" for a in g.nodes.values()):
        handles.append(Patch(fc="#f5f0e1", ec="#8a7d55", label="MAST download"))
    if any(a.get("external") for a in g.nodes.values()):
        handles.append(Patch(fc="white", ec="#888888", ls="--", label="input from another run/path"))
    if any(not a.get("enabled", True) for a in g.nodes.values() if a["kind"] == "step"):
        handles.append(Patch(fc="#eeeeee", ec="#444444", ls="--", label="disabled stage"))
    if any(style == "dotted" for _, _, style in g.edges):
        handles.append(Line2D([], [], color="#777777", ls=":", label="ordering only (depends_on)"))
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.01))
    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
    ax.set_ylim(min(ys) - 1.1, max(ys) + 0.9)
    ax.axis("off")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    written = []
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        fig.savefig(out)
        written.append(out)
    plt.close(fig)
    return written
