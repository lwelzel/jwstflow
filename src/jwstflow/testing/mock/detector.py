"""The mock NIRSpec IFU detector: the layout of a mock cal file and its real ``gwcs``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .instruments import NRS_GRATINGS


def mock_cal_layout(hdr: Any) -> dict[str, Any] | None:
    """The detector layout of a mock NIRSpec cal file, or None for other exposures.

    Rows are the flattened spaxels (``row = iy * size + ix``), columns the
    planes of *this detector's half* of the band (NRS1: the blue half, NRS2:
    the red half -- ``planes = (lo, hi)`` in the band's plane numbering).
    Returns ``size``, ``pix``, the band ``wave`` grid [um], ``planes`` and the
    detector ``shape``.
    """
    if str(hdr.get("EXP_TYPE", "")).upper() != "NRS_IFU":
        return None
    grating, filt = str(hdr["GRATING"]).upper(), str(hdr["FILTER"]).upper()
    lo_um, hi_um = NRS_GRATINGS[(grating, filt)]
    nwave, size, pix = int(hdr["JWFMKNW"]), int(hdr["JWFMKSZ"]), float(hdr["JWFMKPX"])
    half = nwave // 2
    planes = (0, half) if str(hdr["DETECTOR"]).upper() == "NRS1" else (half, nwave)
    return {"size": size, "pix": pix, "wave": np.linspace(lo_um, hi_um, nwave), "planes": planes,
            "shape": (size * size, planes[1] - planes[0])}


def mock_cal_wcs(hdr: Any) -> Any:
    """A ``gwcs`` for the mock detector layout: (x, y) -> (RA, Dec, wavelength [um]).

    The sky part is the very TAN projection the stub cubes carry (centre
    TARG_RA/TARG_DEC, ``JWFMKPX`` arcsec per spaxel), so a detector pixel
    maps back onto exactly the cube spaxel it renders; the bounding box
    covers the detector. Serialises with the datamodel (ASDF extension).
    """
    from astropy import units as u
    from astropy.coordinates import ICRS
    from astropy.modeling import models
    from gwcs import WCS
    from gwcs import coordinate_frames as cf

    layout = mock_cal_layout(hdr)
    if layout is None:
        raise ValueError("mock_cal_wcs: not a mock NIRSpec IFU exposure")
    size, pix, wave = layout["size"], layout["pix"], layout["wave"]
    lo, hi = layout["planes"]
    n_spax = size * size
    idx = np.arange(n_spax)
    iy, ix = np.divmod(idx, size)
    y_to_ix = models.Tabular1D(points=idx.astype(float), lookup_table=ix.astype(float), bounds_error=False,
                               fill_value=np.nan, name="y_to_ix")
    y_to_iy = models.Tabular1D(points=idx.astype(float), lookup_table=iy.astype(float), bounds_error=False,
                               fill_value=np.nan, name="y_to_iy")
    step = float(wave[1] - wave[0])
    x_to_wave = models.Shift(lo) | models.Scale(step) | models.Shift(float(wave[0]))
    centre = (size - 1) / 2.0
    sky = ((models.Shift(-centre) & models.Shift(-centre)) | (models.Scale(-pix / 3600.0) & models.Scale(pix / 3600.0))
           | models.Pix2Sky_TAN() | models.RotateNative2Celestial(float(hdr["TARG_RA"]), float(hdr["TARG_DEC"]), 180.0))
    forward = models.Mapping((1, 1, 0)) | (y_to_ix & y_to_iy & x_to_wave) | (sky & models.Identity(1))
    detector = cf.Frame2D(name="detector", axes_order=(0, 1), unit=(u.pix, u.pix))
    world = cf.CompositeFrame([cf.CelestialFrame(name="sky", reference_frame=ICRS(), axes_order=(0, 1)),
                               cf.SpectralFrame(name="spectral", axes_order=(2,), unit=(u.micron,),
                                                axes_names=("wavelength",))], name="world")
    wcs = WCS([(detector, forward), (world, None)])
    wcs.bounding_box = ((-0.5, hi - lo - 0.5), (-0.5, n_spax - 0.5))
    return wcs


def _mock_flagged_voxels(cal: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(plane, iy, ix) of the OUTLIER-flagged detector pixels of a mock cal file (None: no layout).

    Only the OUTLIER bit (what ``propagate_cluster_flags`` sets) is honoured:
    ``DO_NOT_USE`` alone (the gap edges of ``flag_frames``) is ignored on
    purpose, so the mock cubes stay fully sampled.
    """
    from astropy.io import fits

    with fits.open(cal) as hdul:
        hdr = hdul[0].header
        layout = mock_cal_layout(hdr)
        if layout is None or "DQ" not in hdul or hdul["DQ"].data.shape != layout["shape"]:
            return None
        dq = np.asarray(hdul["DQ"].data)
    rows, cols = np.nonzero(dq & 16)   # OUTLIER
    size, (lo, _) = layout["size"], layout["planes"]
    iy, ix = np.divmod(rows, size)
    return cols + lo, iy, ix
