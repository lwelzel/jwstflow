"""QA of PSF-cube products: ``qa_psf_cube``."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from ... import qa
from ...cubes.psf import (
    PSFCUBE_SUFFIX,
    IfuConfig,
    PsfCubeProduct,
    approx_fwhm_arcsec,
    diffraction_fwhm_arcsec,
)
from ...steps import RunContext, Step


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


class QaPsfNeighbour(Step):
    """What ``subtract_psf_neighbour`` took out of a cube, and what it left.

    One figure per ``*_nbrfit.fits`` (read alone; neither cube is needed).
    Top, a grid of frames on the cube's grid, one column per block of planes
    (its wavelength range tagged): the data, the neighbour's model as
    subtracted (both on the data's asinh scale), and the residual -- data
    minus model minus the fitted background -- on a signed scale set by
    what is left in the fitted region off the neighbour's blanked core (the
    target, never fitted, and that core run off it). Each frame marks the target (+) and the
    neighbour (x) and outlines the target's protected region (dashed), the
    control ring and the blanked core of the neighbour. Pixel and arcsec
    offsets share one zero, the target.

    Below, three panels: the neighbour's total flux from the fit against
    wavelength, with its 1-sigma band; the control ring against wavelength
    in bins (mean residual in the annulus about the neighbour at the
    target's distance, outside the protected region; held out of the fit
    unless the notes say otherwise) next to the same for the half of the
    ring facing away from the target and for the field beyond the ring --
    zero is what a correct subtraction leaves; and the azimuthal mean
    profile about the neighbour, outside the protected region, in the middle
    block: before (data minus background), the model, and the residual,
    on a symmetric log axis (negative values stay visible).
    """

    level = "qa"
    inputs = ("*_nbrfit.fits",)
    version = "1"

    def run(self, inputs: list[Path], ctx: RunContext, *, panel: float = 2.6, bins: int = 60, dpi: int = 150,
            fmt: str = "png", **params: Any) -> list[Path]:
        from ...cubes.psf import PsfSubtraction
        from ...cubes.psf.subtraction import BLANKED, PROTECTED, RING

        out: list[Path] = []
        for inp in inputs:
            rec = PsfSubtraction.read(inp)
            hdr = rec.header
            unit = str(hdr.get("BUNIT", "MJy/sr"))
            data, model, resid = rec.images
            nb, ny, nx = data.shape
            sky = qa.SkyFrame.from_header(rec.image_header, (ny, nx), primary=hdr,
                                          ref_world=(float(hdr["JWFNBTRA"]), float(hdr["JWFNBTDE"])),
                                          reference="target")
            fig = qa.figure(figsize=(panel * nb + 1.9, panel * ny / nx * 3 + 3.6))
            outer = fig.add_gridspec(2, 1, height_ratios=[panel * ny / nx * 3, 2.4], hspace=0.28)
            grid = outer[0].subgridspec(3, nb, wspace=0.08, hspace=0.08)
            lower = outer[1].subgridspec(1, 3, wspace=0.42)
            data_mjy = qa.to_mjy(data, unit)[0]
            norm = qa.stretch_norm(data_mjy, "asinh", (1.0, 99.9))
            # the residual's scale: what is left where the neighbour was fitted, off its blanked core
            judged = rec.mask(1) & ~rec.mask(BLANKED)
            res_mjy = qa.to_mjy(resid, unit)[0][:, judged]
            res_mjy = res_mjy[np.isfinite(res_mjy)]
            lim = float(np.percentile(np.abs(res_mjy), 95.0)) if res_mjy.size else 1.0
            labels = {"target": "target", "neighbour": "neighbour"}
            for b in range(nb):
                w_lo, w_hi = (float(rec.planes["WAVELENGTH"][int(rec.blocks[k][b])]) for k in ("FIRST", "LAST"))
                for row, (img, kind) in enumerate(((data[b], "data"), (model[b], "model"), (resid[b], "residual"))):
                    ax = fig.add_subplot(grid[row, b])
                    last = b == nb - 1
                    sides = ("l" if b == 0 else "") + ("r" if last else "") + ("t" if row == 0 else "") \
                        + ("b" if row == 2 else "")
                    frame = qa.Frame(ax)
                    if kind == "residual":
                        frame.image(img, unit=unit, sky=sky, stretch="linear", center=0.0, vmin=-lim, vmax=lim,
                                    cbar=last, sides=sides)
                    else:
                        frame.image(img, unit=unit, sky=sky, norm=norm, cbar=last, sides=sides)
                    first = b == 0 and row == 0
                    frame.mask(rec.mask(RING), ls="solid", color=qa.OVERLAY_COLORS[1],
                               label="control ring" if first else None)
                    frame.mask(rec.mask(PROTECTED), ls="dashed", color=qa.OVERLAY_COLORS[0],
                               label="target's protected region" if first else None)
                    frame.mask(rec.mask(BLANKED), ls="dotted", color=qa.OVERLAY_COLORS[2],
                               label="neighbour's core, blanked in the cube" if first else None)
                    frame.mark(float(hdr["JWFNBTX"]), float(hdr["JWFNBTY"]), label=labels.pop("target", None))
                    frame.mark(float(hdr["JWFNBX"]), float(hdr["JWFNBY"]), symbol="x",
                               label=labels.pop("neighbour", None))
                    ax.set_xlim(-0.5, nx - 0.5)          # the frame is the cube's grid, also when the
                    ax.set_ylim(-0.5, ny - 0.5)          # neighbour lies beyond it (the notes give its pixel)
                    qa.tag(ax, f"{kind}, {qa.um_range(w_lo, w_hi, '.2f')}" if row == 0 else kind)
            self._flux(fig.add_subplot(lower[0, 0]), rec)
            self._ring(fig.add_subplot(lower[0, 1]), rec, unit, bins)
            self._profile(fig.add_subplot(lower[0, 2]), rec, unit)
            ax0 = fig.axes[0]
            target_core = float(hdr.get("JWFNBTSB", np.nan))
            ring_med = float(hdr.get("JWFNBRMD", np.nan))
            qa.annotate(
                ax0, inp.name,
                f"neighbour at ({float(hdr['JWFNBX']):.2f}, {float(hdr['JWFNBY']):.2f}) pix, "
                + ("fitted on the cube" if hdr.get("JWFNBREF") else "as given (core outside the field)")
                + f", {float(hdr['JWFNBSEP']):.2f} arcsec from the target; library {hdr.get('JWFNBLIB') or '?'}, "
                f"{float(hdr['JWFNBFOV']):.1f} arcsec field ({float(hdr['JWFNBNED']):.1f} needed)",
                f"fitted: PSF{' + halo ' + str(hdr['JWFNBHAL']) + ' arcsec' if hdr.get('JWFNBHAL', 'none') != 'none' else ''}"
                f" + {hdr.get('JWFNBBKG')} background (not subtracted), model error {float(hdr['JWFNBMER']):g}, "
                f"clip {float(hdr['JWFNBCLP']):g}" + r"$\sigma$" + f", {hdr.get('JWFNBSMP')} sampling; protected radius "
                f"{float(hdr['JWFNBPRO']):.2f} arcsec; control ring {float(hdr['JWFNBRNG']):.2f} "
                + r"$\pm$" + f" {float(hdr['JWFNBRWD']):.2f} arcsec, "
                + ("held out of the fit" if hdr.get("JWFNBHLD") else "inside the fit")
                + "; dotted vertical line in the profile: the target's distance",
                "median: reduced " + r"$\chi^2$" + f" {float(hdr['JWFNBCHI']):.2f}; control ring "
                f"{_signed(ring_med / target_core * 100 if target_core else np.nan, '.2f')} % of the target's core "
                f"(before: {_signed(float(hdr.get('JWFNBRBF', np.nan)) / target_core * 100 if target_core else np.nan, '.1f')} %); "
                f"{int(hdr.get('JWFNBNBL', 0))} spaxels of the neighbour's core blanked",
                sky.note() if sky is not None else "")
            qa.figlegend(fig)
            out.append(qa.save(fig, ctx.output_dir / f"{inp.stem}.{fmt}", dpi=dpi))
        return out

    @staticmethod
    def _flux(ax: Any, rec: Any) -> None:
        wave = rec.planes["WAVELENGTH"]
        flux, label = qa.to_mjy(rec.planes["FLUX"], "Jy")
        error = qa.to_mjy(rec.planes["FLUX_ERR"], "Jy")[0]
        qa.step(ax, wave, flux, color=qa.MAIN_COLOR, lw=qa.LW["main"], label="left: the neighbour's flux, PSF fit")
        qa.error_band(ax, wave, flux, error, label="left: " + qa.SIGMA_BAND_LABEL)
        ax.set_xlabel(qa.WAVE_LABEL)
        ax.set_ylabel(label)
        qa.set_wave_scale(ax, "log")
        qa.set_flux_scale(ax, "log")

    @staticmethod
    def _ring(ax: Any, rec: Any, unit: str, bins: int) -> None:
        wave = rec.planes["WAVELENGTH"]
        edges = np.linspace(0, len(wave), max(2, min(int(bins), len(wave))) + 1).round().astype(int)
        centres = np.array([np.nanmedian(wave[a:b]) for a, b in pairwise(edges)])

        def binned(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            v = qa.to_mjy(values, unit)[0]
            means = np.array([np.nanmean(v[a:b]) if np.isfinite(v[a:b]).any() else np.nan
                              for a, b in pairwise(edges)])
            errs = np.array([np.nanstd(v[a:b]) / np.sqrt(max(np.isfinite(v[a:b]).sum(), 1))
                             for a, b in pairwise(edges)])
            return means, errs

        ring, ring_err = binned(rec.planes["RING"])
        back, _ = binned(rec.planes["RING_BACK"])
        far, _ = binned(rec.planes["SKY_FAR"])
        qa.refline(ax, y=0.0)
        qa.step(ax, centres, far, color=qa.ROLE_COLORS["background"], lw=qa.LW["thin"], ls="-.",
                label="middle: field beyond the ring, residual")
        if np.isfinite(back).any():
            qa.step(ax, centres, back, color=qa.ROLE_COLORS["context"], lw=qa.LW["thin"], ls="--",
                    label="middle: half of the ring facing away")
        qa.step(ax, centres, ring, color=qa.MAIN_COLOR, lw=qa.LW["main"], label="middle: control ring, residual")
        qa.error_band(ax, centres, ring, ring_err, label=r"middle: scatter of the planes in a bin / $\sqrt{n}$")
        ax.set_xlabel(qa.WAVE_LABEL)
        ax.set_ylabel(qa.label(r"$\Delta I_\nu$", r"mJy arcsec$^{-2}$"))
        qa.set_wave_scale(ax, "log")

    @staticmethod
    def _profile(ax: Any, rec: Any, unit: str) -> None:
        nb = len(rec.blocks["FIRST"])
        b = nb // 2
        r = rec.profile["R_ARCSEC"]
        before = qa.to_mjy(rec.profile[f"BEFORE_{b}"], unit)[0]
        model = qa.to_mjy(rec.profile[f"MODEL_{b}"], unit)[0]
        after = qa.to_mjy(rec.profile[f"AFTER_{b}"], unit)[0]
        lam = float(rec.blocks["WAVELENGTH"][b])
        ax.plot(r, before, color=qa.ROLE_COLORS["context"], lw=qa.LW["main"],
                label=f"right, about the neighbour at {qa.um(lam)}: data − background")
        ax.plot(r, model, color=qa.ROLE_COLORS["model"], lw=qa.LW["main"], ls="--", label="right: the neighbour's model")
        ax.plot(r, after, color=qa.MAIN_COLOR, lw=qa.LW["main"], ls=":", label="right: residual")
        qa.refline(ax, x=float(rec.header["JWFNBRNG"]), ls="dotted")
        finite = np.abs(np.concatenate([before, model, after]))
        finite = finite[np.isfinite(finite) & (finite > 0)]
        ax.set_yscale("symlog", linthresh=float(np.percentile(finite, 10)) if finite.size else 1e-3)
        ax.set_xlabel(qa.label("$r$", "arcsec"))
        ax.set_ylabel(qa.SB_LABEL)


def _signed(value: float, fmt: str) -> str:
    """A signed number with a real minus sign (rule 17)."""
    return format(float(value), "+" + fmt).replace("-", "\N{MINUS SIGN}")
