"""QA of PSF-cube products: ``qa_psf_cube``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ... import qa
from ...psf import (
    PSFCUBE_SUFFIX,
    IfuConfig,
    PsfCubeProduct,
    approx_fwhm_arcsec,
    diffraction_fwhm_arcsec,
)
from ...steps.base import RunContext, Step


class QaPsfCube(Step):
    """Figure of a PSF-cube product: slices across wavelength, FWHM and in-field energy.

    For each ``*_psfcube.fits``, ``n_slices`` PSF slices spread over the
    stored wavelengths are imaged (log stretch, axes in arcsec offsets), next
    to the measured FWHM against wavelength -- with the diffraction limit
    1.025 lambda/D and, for MIRI MRS, the Law et al. 2023 empirical relation
    as references -- and the per-slice in-field energy fraction. Everything is
    read from the product itself (the psfcube contract is self-describing);
    one PNG per input, named after it, following the jwstflow QA figure
    standard (``docs/qa_figures.md``).
    """

    level = "qa"
    inputs = (f"*_{PSFCUBE_SUFFIX}.fits",)
    version = "4"   # 3 -> 4: PSF slices as frames (pixel and arcsec offsets from the centre)

    def run(self, inputs: list[Path], ctx: RunContext, *, n_slices: int = 4, dpi: int = 150,
            fmt: str = "png", xscale: str = "log", **params: Any) -> list[Path]:
        out: list[Path] = []
        for inp in inputs:
            product = PsfCubeProduct.read(inp)
            try:
                config = IfuConfig.from_header(product.header)
            except ValueError:
                config = None
            n = max(2, min(int(n_slices), len(product.wavelengths)))
            picks = np.unique(np.linspace(0, len(product.wavelengths) - 1, n).round().astype(int))
            fig = qa.figure(figsize=(3.0 * len(picks) + 0.8, 6.4))
            gs = fig.add_gridspec(2, len(picks))
            sky = qa.SkyFrame.from_scale(product.pixelscale_arcsec, product.data.shape[1:], reference="PSF centre")
            for col, k in enumerate(picks):
                last = col == len(picks) - 1
                frame = qa.Frame(fig.add_subplot(gs[0, col]))
                frame.image(product.data[k], sky=sky, stretch="log", cbar=last, cbar_label="PSF (normalised to unit sum)",
                            sides="bt" + ("l" if col == 0 else "") + ("r" if last else ""))
                qa.tag(frame.ax, qa.um(float(product.wavelengths[k])))
            ax_fwhm = fig.add_subplot(gs[1, : max(1, len(picks) // 2)])
            ax_sum = fig.add_subplot(gs[1, max(1, len(picks) // 2):])
            wave, fwhm = product.wavelengths, product.fwhm_arcsec()
            qa.step(ax_fwhm, wave, fwhm, color=qa.MAIN_COLOR, label="measured")
            fine = np.geomspace(wave[0], wave[-1], 200)
            ax_fwhm.plot(fine, diffraction_fwhm_arcsec(fine), color=qa.REF_COLOR, lw=qa.LW["main"], ls="--",
                         label=r"diffraction limit 1.025 $\lambda/D$")
            if config is not None and config.instrument == "MIRI":
                ax_fwhm.plot(fine, approx_fwhm_arcsec(config, fine), color=qa.REF_COLOR, lw=qa.LW["main"], ls=":",
                             label="Law et al. (2023), empirical")
            ax_fwhm.set_xlabel(qa.WAVE_LABEL)
            ax_fwhm.set_ylabel(qa.label("FWHM", "arcsec"))
            qa.set_wave_scale(ax_fwhm, xscale)
            qa.step(ax_sum, wave, product.sums(), color=qa.MAIN_COLOR)
            ax_sum.set_xlabel(qa.WAVE_LABEL)
            ax_sum.set_ylabel("in-field energy fraction")
            ax_sum.set_ylim(0, 1.05)
            qa.set_wave_scale(ax_sum, xscale)
            pa = product.position_angle_deg
            qa.annotate(ax_fwhm, inp.name,
                           f"{product.data.shape[2]} pix of {product.pixelscale_arcsec:.4f} arcsec, "
                           f"frame {product.frame or '?'}, PA {f'{pa:.1f}' if pa is not None else 'n/a'}",
                           f"engine {product.meta_value('JWFPSFEN') or '?'} "
                           f"({product.meta_value('JWFPSFMD') or '?'})")
            qa.figlegend(fig)
            out.append(qa.save(fig, ctx.output_dir / f"{inp.stem}.{fmt}", dpi=dpi))
        return out
