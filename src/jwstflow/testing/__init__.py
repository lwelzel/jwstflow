"""Tools for developing and testing custom steps without real data or a full run.

* :func:`run_step` executes a step exactly as a worker would (RunContext,
  parameter validation, output checks) on files of your choice;
* :func:`synthetic_cube`, :func:`synthetic_x1d`, :func:`synthetic_image`, :func:`synthetic_i2d` write
  small JWST-like products (IFU cube, extracted spectrum, rate/cal image, resampled image)
  with the keywords jwstflow steps typically read;
* :func:`check_step` audits a step's declaration and signature without running
  it (what ``jwstflow check-step`` prints);
* :mod:`jwstflow.testing.mock` runs a *whole workflow* offline: mock
  observations and stand-ins for the official pipelines.

The names above are importable from here; the code is in :mod:`.run`,
:mod:`.synthetic` and :mod:`.audit`.
"""

from .audit import check_step
from .run import make_context, run_step
from .synthetic import synthetic_cube, synthetic_i2d, synthetic_image, synthetic_x1d

__all__ = [
    "check_step",
    "make_context",
    "run_step",
    "synthetic_cube",
    "synthetic_i2d",
    "synthetic_image",
    "synthetic_x1d",
]
