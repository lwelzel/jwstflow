"""Mock observations and stub pipelines: run a *complete* workflow in seconds.

The unit-level tools in :mod:`jwstflow.testing` execute one step on synthetic
products. This module tests the level above: a whole workflow YAML through the
real engine (config loading, input discovery, associations, task planning,
checkpointing, product naming), with only the expensive official pipeline
stages replaced by fast, scene-faithful stand-ins. No network, no CRDS cache,
no MAST -- a full NIRSpec-IFU or MIRI-MRS reduction runs in seconds.

Three pieces:

* **Mock observations** -- :func:`nirspec_ifu_observation` /
  :func:`miri_mrs_observation` describe an observation exactly like a real one
  (DMS file names, header keywords, dither/detector/grating structure, MAST
  download log), modelled on the MIDAS ESO-Ha 569 program (jw01751), but with
  strongly reduced data sizes: a few dithers, tens of spectral planes instead
  of thousands, ~30 px fields. The astrophysical scene (:class:`MockScene`:
  an edge-on disk and/or point source, flat sky, emission-line halos, a hot
  pixel) travels in ``JWFMK*`` header keywords from the ``_uncal`` files all
  the way to the cubes, so every product is deterministic and assertable.
  :func:`write_observation` materialises the ``_uncal`` files plus the MAST
  observation log (the source of the DMS target id).

* **Stub pipelines** -- :class:`StubDetector1`, :class:`StubSpec2`,
  :class:`StubSpec3` and :class:`StubCubeBuild` mimic the official pipelines' *interfaces*: same stage
  names (``calwebb_detector1`` ...), same product names (``_rate``, ``_cal``,
  DMS level-3 cube names with the band appended), consuming the same
  association files, honouring the parameters that shape the data flow
  (``steps.extract_1d.skip``, background members, ``background_dir``).
  The data they write is synthesised from the scene keywords: ``spec2``
  subtracts the sky of associated background members like ``bkg_subtract``
  would, ``spec3`` renders the scene into IFU cubes, and
  ``spec3_with_background`` subtracts the measured ``*_bkgspec.fits`` sky
  like ``master_background`` would. NIRSpec cal files carry a tiny but
  faithful IFU *detector layout* with a real ``gwcs``: one detector pixel
  per (spaxel, plane) of the detector's half of the band (rows = spaxels,
  columns = planes; :func:`mock_cal_wcs`), so DQ flags set on a cal file
  (``OUTLIER``, what ``propagate_cluster_flags`` writes) are honoured when the
  cubes are rendered -- the bad-cluster chain of :mod:`jwstflow.clusters`
  runs end to end offline. Custom (jwstflow / contributed) steps then run
  *for real* on those cubes.

* **Runners** -- :func:`stub_pipelines` shadows the official step aliases in
  the in-process registry (use the serial backend / ``workers=1``, where
  tasks run in this process; by dotted path,
  ``jwstflow.testing.mock:StubDetector1`` etc. also work in spawned workers),
  :func:`offline_overrides` pins the CRDS context and disables everything
  that would touch the network, and :func:`run_mock_workflow` bundles both
  around ``load_config`` + ``Runner`` for one-line integration tests::

      write_observation(work / "reductions/eso-ha-569/raw", nirspec_ifu_observation())
      cfg, summary = run_mock_workflow("reductions/eso-ha-569/nirspec_ifu.yaml", work)
      assert summary.ok

The package, by module: :mod:`.instruments` (band tables), :mod:`.scene`
(:class:`MockScene`), :mod:`.observations` (the mock observations and the
imager star field), :mod:`.detector` (the mock IFU detector layout and WCS),
:mod:`.stubs` (the stub pipelines) and :mod:`.workflow` (the runners).
Everything is importable from here, and the stubs are addressed in YAML as
``jwstflow.testing.mock:StubSpec3``.

Everything here is deliberately import-light at module import time (numpy
only); astropy / stdatamodels load inside the functions that need them, like
the rest of jwstflow.
"""

from .detector import mock_cal_layout, mock_cal_wcs
from .instruments import MRS_BANDS, MRS_DETECTOR_CHANNELS, MRS_PIX_ARCSEC, NRS_GRATINGS
from .observations import (
    DEFAULT_NIRSPEC_SCENE,
    MockExposure,
    MockObservation,
    miri_mrs_observation,
    nirspec_ifu_observation,
    star_field,
    write_gaia_catalog,
    write_observation,
)
from .scene import MockScene
from .stubs import (
    DEFAULT_STUBS,
    StubCubeBuild,
    StubDetector1,
    StubImage2,
    StubImage3,
    StubSpec2,
    StubSpec3,
    StubSpec3WithBackground,
    stub_pipelines,
)
from .workflow import offline_overrides, run_mock_workflow

__all__ = [
    "MockScene", "MockExposure", "MockObservation",
    "nirspec_ifu_observation", "miri_mrs_observation", "write_observation",
    "star_field", "write_gaia_catalog",
    "StubDetector1", "StubSpec2", "StubSpec3", "StubSpec3WithBackground",
    "StubCubeBuild", "StubImage2", "StubImage3", "mock_cal_layout", "mock_cal_wcs",
    "DEFAULT_NIRSPEC_SCENE",
    "stub_pipelines", "offline_overrides", "run_mock_workflow",
    "NRS_GRATINGS", "MRS_BANDS", "MRS_DETECTOR_CHANNELS", "MRS_PIX_ARCSEC", "DEFAULT_STUBS",
]
