"""The QA steps shipped with jwstflow.

Reference them in YAML by entry-point name (``plot_spectrum``) or dotted path
(``jwstflow.qa.steps.spectra:PlotSpectrum``). They double as examples of the
:class:`jwstflow.Step` API and of figures built on :class:`jwstflow.qa.Spectrum`
and :class:`jwstflow.qa.Frame`:

* :mod:`.spectra` -- 1-D spectra: ``plot_spectrum``, ``plot_snr``, ``plot_spectrum_stages``
* :mod:`.stitch` -- stitched spectra: ``plot_stitch``, ``plot_stitch_overlaps``, ``plot_stitch_background``
* :mod:`.images` -- ``quicklook_image``
* :mod:`.cubes` -- IFU cubes: ``qa_cube_summary``, ``qa_cube_slices``
* :mod:`.summary` -- ``header_summary`` (a table, not a figure)
"""

from .cubes import QaCubeSlices, QaCubeSummary
from .images import QuicklookImage
from .spectra import PlotSnr, PlotSpectrum, PlotSpectrumStages
from .stitch import PlotStitch, PlotStitchBackground, PlotStitchOverlaps
from .summary import HeaderSummary, copy_inputs

__all__ = [
    "HeaderSummary",
    "PlotSnr",
    "PlotSpectrum",
    "PlotSpectrumStages",
    "PlotStitch",
    "PlotStitchBackground",
    "PlotStitchOverlaps",
    "QaCubeSlices",
    "QaCubeSummary",
    "QuicklookImage",
    "copy_inputs",
]
