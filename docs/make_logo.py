#!/usr/bin/env python3
"""
make_jwstflow_logo.py -- generate the jwstflow logo with matplotlib.

jwstflow: lightweight YAML-driven orchestrator for the official JWST
calibration pipeline (NIRSpec + MIRI).

The logo is a JWST primary-mirror segment -- a regular hexagon with flat top
and bottom, mirror gold, a black keyline and a white keyline outside it -- with
an Avatar-style waterbending emblem laid over it in the gold's complementary
colour (the hue 180 degrees away, darkened to a deep navy).

Everything is drawn from parametric curves; nothing is traced from a bitmap.
The emblem is built from three ingredients, all stroked with one constant pen
width:

  * an outer ring;
  * three "curls" -- spiral arcs r(theta) about three eye points, where the
    radial profile is a monotone cubic through a handful of anchors;
  * three "wave" lines, each a plain cosine y = c + A cos(k(x - x0)).

Every line is laid down twice, once fat and white and once at its own width in
blue, so a thin white outline separates the emblem from the gold. The black
keyline goes on last, which keeps that outline from creeping in between the blue
and the black where the lines run off the edge.

Two rules keep that outline from turning into clutter. Lines that are not meant
to touch are kept at least PEN + 2*HALO + 0.03 apart, measured centreline to
centreline, so a strip of gold always survives between them; the anchors below
are spaced for that, which opens the middle and small curls a little wider than
the emblem they are drawn from. And lines that are meant to meet something --
the rim, or a neighbouring curl -- are run well into it and cut off by the clip,
so a merge never shows the rounded end of a stroke.

The anchor values and the wave coefficients below were calibrated against the
classic Water Tribe emblem, so the output matches it closely while staying a
purely analytic construction. Coordinates are in emblem units: the outer ring
has radius 1 and is centred on the origin.

Two variants are written, each as a transparent PNG and a vector PDF:

  * jwstflow_logo_circle.*  full emblem, ring included, inside the hexagon;
  * jwstflow_logo_open.*    no ring -- the curls and waves are clipped by the
                            hexagon so the lines run out from behind its
                            black keyline.

Usage
-----
    python make_jwstflow_logo.py [--outdir DIR] [--size INCHES] [--dpi DPI]
                                 [--gold "#D9A72E"]
"""
from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
JWST_GOLD = "#D9A72E"  # gold-coated beryllium mirror segments
BLACK = "#000000"
WHITE = "#FFFFFF"


def _hex_to_rgb(color: str):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _rgb_to_hex(rgb) -> str:
    return "#" + "".join(f"{round(c * 255):02X}" for c in rgb)


def complementary(color: str, value: float, saturation: float | None = None) -> str:
    """Colour whose hue is exactly opposite `color` on the colour wheel.

    The hue is the true complement (180 degrees away, i.e. the hue of the RGB
    inverse). `value` sets the brightness explicitly so the result can be a
    *dark* shade of that hue; `saturation` optionally overrides the input's.
    """
    h, s, _ = colorsys.rgb_to_hsv(*_hex_to_rgb(color))
    if saturation is not None:
        s = saturation
    return _rgb_to_hex(colorsys.hsv_to_rgb((h + 0.5) % 1.0, s, value))


# ---------------------------------------------------------------------------
# Monotone cubic interpolation (Fritsch-Carlson), so no SciPy is needed
# ---------------------------------------------------------------------------
def monotone_cubic(xs, ys):
    """Return a smooth interpolant through (xs, ys) that adds no overshoot."""
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.empty_like(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        harmonic = 2.0 / (1.0 / d[:-1] + 1.0 / d[1:])
    m[1:-1] = np.where(d[:-1] * d[1:] > 0, harmonic, 0.0)
    m[0], m[-1] = d[0], d[-1]

    def f(q):
        q = np.asarray(q, float)
        i = np.clip(np.searchsorted(x, q) - 1, 0, len(x) - 2)
        t = (q - x[i]) / h[i]
        t2, t3 = t * t, t * t * t
        return ((2 * t3 - 3 * t2 + 1) * y[i]
                + (t3 - 2 * t2 + t) * h[i] * m[i]
                + (-2 * t3 + 3 * t2) * y[i + 1]
                + (t3 - t2) * h[i] * m[i + 1])

    return f


# ---------------------------------------------------------------------------
# The waterbending emblem, in emblem units (outer ring radius = 1)
# ---------------------------------------------------------------------------
PEN = 0.090           # stroke width, the same for every line of the emblem
HALO = 0.013          # white outline carried on each side of a blue line
RING_R = 0.951        # ring centreline radius
RING_OUT = RING_R + PEN / 2   # outer edge of the ring
MERGE_R = 0.86        # radius at which a line starts its run into the ring

EYE_BIG = (-0.570, 0.016)     # eyes of the three curls, largest first
EYE_MID = (-0.086, 0.286)
EYE_SMALL = (0.335, 0.485)

# Radial profiles r(theta): theta in degrees, measured at the curl's own eye and
# increasing anticlockwise, so theta > 360 is the second time round. Each curl
# unwinds outwards and its final sweep runs into the ring. The middle and small
# curls are interrupted where the next-larger curl crosses over them, so they
# come in two arcs whose inner ends butt into that larger stroke.
CURLS = [
    dict(eye=EYE_BIG, arcs=[[
        (-90, 0.038), (-45, 0.121), (0, 0.162), (45, 0.184), (90, 0.185),
        (135, 0.202), (180, 0.204), (225, 0.241), (270, 0.262), (315, 0.283),
        (360, 0.337), (405, 0.405), (450, 0.452), (486, 0.478),
    ]]),
    dict(eye=EYE_MID, arcs=[
        [(-77, 0.030), (-45, 0.118), (0, 0.167), (45, 0.180), (90, 0.180),
         (135, 0.194), (160, 0.217)],
        [(245, 0.262), (270, 0.252), (315, 0.281), (360, 0.337), (405, 0.400),
         (450, 0.521), (479, 0.642)],
    ]),
    dict(eye=EYE_SMALL, arcs=[
        [(-91, 0.026), (-45, 0.094), (0, 0.135), (45, 0.138), (90, 0.130),
         (135, 0.150), (137, 0.151)],
        [(256, 0.218), (270, 0.214), (315, 0.264), (360, 0.308), (389, 0.352)],
    ]),
]

# Wave lines: y = c + A cos(k (x - x0)), clipped by the ring.
WAVES = [
    dict(c=-0.1486, A=0.3212, k=2.5384, x0=0.7097),
    dict(c=-0.3602, A=0.3218, k=2.5714, x0=0.7975),
    dict(c=-0.6033, A=0.2808, k=3.1127, x0=0.8116),
]


def _polar(eye, t_deg, r):
    a = np.radians(t_deg)
    return np.c_[eye[0] + r * np.cos(a), eye[1] + r * np.sin(a)]


def spiral_arc(eye, anchors, n=400):
    """Sample one curl arc: polar r(theta) about `eye`, theta in degrees."""
    th = np.array([a[0] for a in anchors], float)
    rr = np.array([a[1] for a in anchors], float)
    t = np.linspace(th[0], th[-1], n)
    return _polar(eye, t, monotone_cubic(th, rr)(t))


def wave_arc(c, A, k, x0, inside=None, rmax=MERGE_R, n=600):
    """Sample one wave line over the stretch that stays inside `inside`.

    Only the run through the middle of the emblem is kept: a cosine wanders back
    inside a circle further along, and following it there would send the line
    round the rim into its neighbours.
    """
    if inside is None:
        def inside(P):
            return np.hypot(P[:, 0], P[:, 1]) <= rmax
    x = np.linspace(-2.0, 2.0, 8001)
    P = np.c_[x, c + A * np.cos(k * (x - x0))]
    m = inside(P)
    i = int(np.argmin(np.hypot(P[:, 0], P[:, 1])))       # deepest point inside
    lo = hi = i
    while lo > 0 and m[lo - 1]:
        lo -= 1
    while hi < len(x) - 1 and m[hi + 1]:
        hi += 1
    x = np.linspace(x[lo], x[hi], n)
    return np.c_[x, c + A * np.cos(k * (x - x0))]


def trim_to_radius(P, r=MERGE_R):  # noqa: D401
    """Cut a line back to the last point that is still within radius `r`."""
    inside = np.nonzero(np.hypot(P[:, 0], P[:, 1]) <= r)[0]
    return P[:inside[-1] + 1] if len(inside) else P


def run_to_ring(P, at_start=False, r_out=RING_OUT + PEN, dphi=14.0, arrive=0.7,
                n=140):
    """Curve a line out of the emblem and through the ring.

    It leaves along its own tangent and arrives across the ring at a steady
    angle (`arrive` = 1 is dead radial), a `dphi` degrees further round. Ending
    beyond the rim means the clip, not a round cap, decides where it stops, so
    the line melts into the ring instead of butting up against it.
    """
    if at_start:
        P = P[::-1]
    p0 = P[-1]
    t0 = p0 - P[-4]
    t0 /= np.linalg.norm(t0)
    phi0 = np.arctan2(p0[1], p0[0])
    turn = np.sign(p0[0] * t0[1] - p0[1] * t0[0]) or 1.0   # which way round
    phi1 = phi0 + turn * np.radians(dphi)
    radial = np.array([np.cos(phi1), np.sin(phi1)])
    along = turn * np.array([-np.sin(phi1), np.cos(phi1)])
    p1 = radial * r_out
    m1 = arrive * radial + (1 - arrive) * along
    m1 /= np.linalg.norm(m1)
    L = 1.15 * np.linalg.norm(p1 - p0)
    t = np.linspace(0, 1, n)[:, None]
    t2, t3 = t * t, t * t * t
    tail = ((2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * L * t0
            + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * L * m1)
    Q = np.vstack([P, tail[1:]])
    return Q[::-1] if at_start else Q


def emblem_strokes(ring=True):
    """Every line of the emblem except the ring, as polylines in emblem units.

    Each line is cut back to `r_cut` and then curved out to `r_out`, well past
    whatever will clip it -- the ring's outer edge in one variant, the hexagon
    in the other. Nothing is left to stop in mid-air.
    """
    r_cut, r_out = (MERGE_R, RING_OUT + PEN) if ring else (0.98, 1.55)
    arrive, dphi = (0.70, 14.0) if ring else (0.45, 22.0)
    out = []
    for curl in CURLS:
        last = len(curl["arcs"]) - 1
        for i, anchors in enumerate(curl["arcs"]):
            P = spiral_arc(curl["eye"], anchors)
            # only a curl's final sweep reaches the rim; the inner arcs butt
            # into the neighbouring curl and are left alone
            if i == last:
                P = run_to_ring(trim_to_radius(P, r_cut), r_out=r_out,
                                dphi=dphi, arrive=arrive)
            out.append(P)
    for w in WAVES:
        P = trim_to_radius(wave_arc(**w, rmax=r_cut), r_cut)
        P = run_to_ring(P, at_start=True, r_out=r_out, dphi=dphi, arrive=arrive)
        P = run_to_ring(P, r_out=r_out, dphi=dphi, arrive=arrive)
        out.append(P)
    return out


# ---------------------------------------------------------------------------
# Hexagon (flat top and bottom, circumradius 1) and layout
# ---------------------------------------------------------------------------
HEX_VERTS = [(np.cos(a), np.sin(a)) for a in np.deg2rad(np.arange(0, 360, 60))]

W_BLACK = 0.040       # black keyline width
W_WHITE = 0.034       # white keyline, sitting outside the black one
PAD = 0.02

SCALE_CIRCLE = 0.735  # emblem radius, in units of the hexagon circumradius
SCALE_OPEN = 0.84


LIM = 1.0 + W_BLACK / 2 + W_WHITE + PAD


def make_figure(size: float):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax, size * 72 / (2 * LIM)   # points per data unit


def draw_logo(ax, pt: float, ring: bool, gold: str, blue: str) -> None:
    # White keyline first, then the gold fill over its inner half, so the white
    # only ever shows outside the black keyline added at the end.
    ax.add_patch(Polygon(HEX_VERTS, closed=True, facecolor="none", edgecolor=WHITE,
                         linewidth=(W_BLACK + 2 * W_WHITE) * pt, joinstyle="miter",
                         zorder=1))
    fill = Polygon(HEX_VERTS, closed=True, facecolor=gold, edgecolor="none", zorder=2)
    ax.add_patch(fill)

    # The emblem, scaled onto the hexagon and clipped to it. It goes down in two
    # passes: every line first as a slightly fatter white one, then every line
    # again in blue on top. Drawing all the white before any of the blue is what
    # keeps the halo out of the joins -- where one line butts into another, its
    # halo lands inside that neighbour and the neighbour's blue covers it.
    scale = SCALE_CIRCLE if ring else SCALE_OPEN
    strokes = emblem_strokes(ring)

    # With the ring in place the lines are cut off at its outer edge; without it
    # the hexagon does the cutting.
    edge = fill
    if ring:
        edge = Circle((0, 0), RING_OUT * scale, facecolor="none", edgecolor="none")
        ax.add_patch(edge)

    def pass_(color, width, zorder):
        for P in strokes:
            line = Line2D(scale * P[:, 0], scale * P[:, 1], color=color,
                          linewidth=width * scale * pt, solid_capstyle="round",
                          solid_joinstyle="round", zorder=zorder)
            ax.add_line(line)
            line.set_clip_path(edge)
        if ring:
            circ = Circle((0, 0), RING_R * scale, facecolor="none", edgecolor=color,
                          linewidth=width * scale * pt, zorder=zorder)
            ax.add_patch(circ)
            circ.set_clip_path(fill)

    pass_(WHITE, PEN + 2 * HALO, zorder=3)
    pass_(blue, PEN, zorder=4)

    # Black keyline last: it covers the halo along the rim, so the lines run out
    # from behind it with no white between the blue and the black.
    ax.add_patch(Polygon(HEX_VERTS, closed=True, facecolor="none", edgecolor=BLACK,
                         linewidth=W_BLACK * pt, joinstyle="miter", zorder=5))


def render(stem: str, ring: bool, outdir: Path, size: float, dpi: int,
           gold: str, blue: str):
    fig, ax, pt = make_figure(size)
    draw_logo(ax, pt, ring, gold, blue)
    written = []
    for ext, kwargs in (("png", {"dpi": dpi}), ("pdf", {})):
        path = outdir / f"{stem}.{ext}"
        fig.savefig(path, transparent=True, **kwargs)
        written.append(path)
    plt.close(fig)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Generate the jwstflow logo (PNG + PDF).")
    p.add_argument("--outdir", type=Path, default=Path("./logo"), help="output directory")
    p.add_argument("--size", type=float, default=6.0, help="figure size in inches (square)")
    p.add_argument("--dpi", type=int, default=400, help="PNG resolution")
    p.add_argument("--gold", default=JWST_GOLD, help="hexagon fill colour")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    blue = complementary(args.gold, value=0.36, saturation=0.85)
    print(f"hexagon gold : {args.gold}")
    print(f"emblem blue  : {blue}  (complementary hue of the gold, darkened)")
    for stem, ring in (("jwstflow_logo_circle", True), ("jwstflow_logo_open", False)):
        for path in render(stem, ring, args.outdir, args.size, args.dpi, args.gold, blue):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()