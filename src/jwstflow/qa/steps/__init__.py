"""The QA steps shipped with jwstflow.

Reference them in YAML by entry-point name (``plot_spectrum``) or dotted path
(``jwstflow.qa.steps.spectra:PlotSpectrum``). They double as examples of the
:class:`jwstflow.Step` API and of figures built on :class:`jwstflow.qa.Spectrum`
and :class:`jwstflow.qa.Frame`:

* :mod:`.spectra` -- 1-D spectra: ``plot_spectrum``, ``plot_spectrum_apertures``, ``plot_snr``,
  ``plot_spectrum_stages``
* :mod:`.archival` -- archival photometry and spectra over the spectra: ``plot_archival``
* :mod:`.stitch` -- stitched spectra: ``plot_stitch``, ``plot_stitch_overlaps``, ``plot_stitch_background``
* :mod:`.images` -- ``quicklook_image``
* :mod:`.cubes` -- IFU cubes: ``qa_cube_summary``, ``qa_cube_slices``
* :mod:`.psf` -- PSF cubes (:mod:`jwstflow.cubes.psf`): ``qa_psf_cube``
* :mod:`.clusters` -- bad spaxel clusters (:mod:`jwstflow.cubes.clusters`): ``qa_spaxel_clusters``
* :mod:`.gaps` -- the NIRSpec detector gap (:mod:`jwstflow.cubes.gaps`): ``qa_gap_coverage``
* :mod:`.cube_stitch` -- joint cubes (:mod:`jwstflow.cubes.stitching`): ``qa_cube_stitch``
* :mod:`.mast_compare` -- ``mast_compare``: jwstflow's products against the archive's own
* :mod:`.summary` -- ``header_summary`` (a table, not a figure)
"""

from .clusters import QaSpaxelClusters
from .cube_stitch import QaCubeStitch
from .cubes import QaCubeSlices, QaCubeSummary
from .gaps import QaGapCoverage
from .images import QuicklookImage
from .mast_compare import MastCompare
from .psf import QaPsfCube
from .archival import PlotArchival
from .spectra import PlotSnr, PlotSpectrum, PlotSpectrumApertures, PlotSpectrumStages
from .stitch import PlotStitch, PlotStitchBackground, PlotStitchOverlaps
from .summary import HeaderSummary

__all__ = [
    "HeaderSummary",
    "MastCompare",
    "PlotSnr",
    "PlotArchival",
    "PlotSpectrum",
    "PlotSpectrumApertures",
    "PlotSpectrumStages",
    "PlotStitch",
    "PlotStitchBackground",
    "PlotStitchOverlaps",
    "QaCubeSlices",
    "QaCubeStitch",
    "QaCubeSummary",
    "QaGapCoverage",
    "QaPsfCube",
    "QaSpaxelClusters",
    "QuicklookImage",
]
