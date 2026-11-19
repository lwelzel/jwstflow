"""jwstflow's QA figures: the standard look and the building blocks of every figure.

``docs/qa_figures.md`` is the style guide. The package is laid out by what a
figure is made of:

* :mod:`~jwstflow.qa.style` -- palette, typography, axis-label text, figure factory, legend, saving
* :mod:`~jwstflow.qa.units` -- everything flux-like in mJy
* :mod:`~jwstflow.qa.lines` -- step lines, bands, reference marks, log axes
* :mod:`~jwstflow.qa.frames` -- images with pixel and sky offsets, stretches, colorbars, overlays
* :mod:`~jwstflow.qa.annotation` -- the annotation of spectral features; its parts:
  :mod:`~jwstflow.qa.labels` (text and color), :mod:`~jwstflow.qa.placement` (geometry along one axis),
  :mod:`~jwstflow.qa.layout` (the layout, carried out when the figure is saved)
* :mod:`~jwstflow.qa.plots` -- :class:`Spectrum` and :class:`Frame`, the two kinds of figure to build on
* :mod:`~jwstflow.qa.io` -- locating and reading products for a figure
* :mod:`~jwstflow.qa.steps` -- the QA steps shipped with jwstflow (imported on demand)

Everything a QA step needs is importable from here (``from jwstflow import qa``).
Matplotlib is imported lazily, so importing this package stays cheap.
"""

from . import annotation, frames, io, labels, layout, lines, plots, style, units
from .annotation import *
from .frames import *
from .io import *
from .labels import *
from .layout import *
from .lines import *
from .plots import *
from .style import *
from .units import *

# every module lists its public names; a helper added to one of them is exported from here as well
__all__ = [  # noqa: PLE0604
    *annotation.__all__,
    *frames.__all__,
    *io.__all__,
    *labels.__all__,
    *layout.__all__,
    *lines.__all__,
    *plots.__all__,
    *style.__all__,
    *units.__all__,
]
