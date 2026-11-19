"""Scaffolding: the boilerplate jwstflow can write for you.

* :func:`scaffold_step` -- one custom step and a passing test (``jwstflow new-step``);
* :func:`scaffold_package` -- a whole contributed step package: pyproject with entry
  points, src layout, step stubs, tests, README, CI, git (``jwstflow new-package``).

The templates live inside the core package, so generated boilerplate always matches
the step contract of the installed jwstflow version.
"""

from .package import scaffold_package
from .step import scaffold_step

__all__ = ["scaffold_package", "scaffold_step"]
