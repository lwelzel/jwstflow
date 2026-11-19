"""1-D spectra: what steps that read or write spectra build on.

* :mod:`.x1d` -- the jwst ``x1d`` table layout: :class:`Spectrum1D`, :func:`read_x1d`,
  :func:`write_x1d` and the columns and keywords that travel through 1-D processing;
* :mod:`.features` -- the line and band datasets shipped with jwstflow (``features:`` in YAML);
* :mod:`.stitching` -- the stitching contract: the ``stitch_segments`` step that splices
  overlapping segments, and that step packages subclass.

The x1d names are importable from here (``from jwstflow.spectra import Spectrum1D, write_x1d``).
"""

from .x1d import (
    X1D_COLUMNS,
    X1D_DTYPE,
    X1D_PASSTHROUGH_COLUMNS,
    X1D_PROVENANCE_KEYS,
    X1D_UNITS,
    Spectrum1D,
    passthrough_columns,
    read_x1d,
    write_x1d,
    x1d_table,
)

__all__ = [
    "X1D_COLUMNS",
    "X1D_DTYPE",
    "X1D_PASSTHROUGH_COLUMNS",
    "X1D_PROVENANCE_KEYS",
    "X1D_UNITS",
    "Spectrum1D",
    "passthrough_columns",
    "read_x1d",
    "write_x1d",
    "x1d_table",
]
