"""The QA steps shipped with jwstflow.

Reference them in YAML by entry-point name (``plot_spectrum``) or dotted path
(``jwstflow.qa.steps.spectra:PlotSpectrum``). They double as examples of the
:class:`jwstflow.Step` API and of figures built on :class:`jwstflow.qa.Spectrum`
and :class:`jwstflow.qa.Frame`:

* :mod:`.spectra` -- 1-D spectra: ``plot_spectrum``, ``plot_snr``, ``plot_spectrum_stages``
* :mod:`.stitch` -- stitched spectra: ``plot_stitch``, ``plot_stitch_overlaps``, ``plot_stitch_background``
* :mod:`.images` -- ``quicklook_image``
* :mod:`.cubes` -- IFU cubes: ``qa_cube_summary``, ``qa_cube_slices``
* :mod:`.psf` -- PSF cubes (:mod:`jwstflow.psf`): ``qa_psf_cube``
* :mod:`.clusters` -- bad spaxel clusters (:mod:`jwstflow.clusters`): ``qa_spaxel_clusters``
* :mod:`.mast_compare` -- ``mast_compare``: jwstflow's products against the archive's own
* :mod:`.summary` -- ``header_summary`` (a table, not a figure)
"""

from .clusters import QaSpaxelClusters
from .cubes import QaCubeSlices, QaCubeSummary
from .images import QuicklookImage
from .mast_compare import MastCompare
from .psf import QaPsfCube
from .spectra import PlotSnr, PlotSpectrum, PlotSpectrumStages
from .stitch import PlotStitch, PlotStitchBackground, PlotStitchOverlaps
from .summary import HeaderSummary, copy_inputs

__all__ = [
    "HeaderSummary",
    "MastCompare",
    "PlotSnr",
    "PlotSpectrum",
    "PlotSpectrumStages",
    "PlotStitch",
    "PlotStitchBackground",
    "PlotStitchOverlaps",
    "QaCubeSlices",
    "QaCubeSummary",
    "QaPsfCube",
    "QaSpaxelClusters",
    "QuicklookImage",
    "copy_inputs",
]
