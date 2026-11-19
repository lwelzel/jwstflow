"""The QA figure standard under its first name: everything lives in :mod:`jwstflow.qa`.

``from jwstflow import qafig`` keeps working for step packages written against
it; new code imports :mod:`jwstflow.qa`.
"""

from .qa import *
from .qa import __all__  # noqa: F401
from .qa.frames import _robust_max  # noqa: F401
from .qa.labels import _line_label, _series  # noqa: F401
from .qa.placement import _declutter, _spread  # noqa: F401
