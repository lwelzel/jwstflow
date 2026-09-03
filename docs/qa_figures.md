# The QA figure standard

Every QA figure -- from jwstflow's own steps and from contributed step
packages (`jwstflow-midas`, `jwstflow-joys`, ...) -- follows one standard, so
a reduction's `qa/` tree reads as a single, consistent report. The standard
is *set here*, in the core package, and *implemented* by `jwstflow.qafig`:
QA steps build their figures through that module instead of using matplotlib
directly.

## The rules

1. **No titles.** Figures and subplots never carry a title. Whatever a title
   would have said (the product name, spaxel counts, parameters) goes into
   the legend: `qafig.annotate(ax, text, ...)` adds hand-less legend entries
   and `qafig.figlegend(fig)` places one combined legend *outside* the axes,
   above the figure.
2. **Flux-like quantities are plotted in mJy**, surface brightness in
   mJy arcsec⁻². `qafig.to_mjy(values, unit)` converts from whatever unit a
   product carries (`Jy`, `MJy`, `MJy/sr`, ... -- pass the FITS
   `TUNIT`/`BUNIT` string) and returns the finished axis label.
3. **Line-style plots use `ax.step(..., where="mid")`** -- spectra are
   histograms over wavelength bins, not smooth curves. `qafig.step(ax, x, y)`
   wraps it.
4. **Every `imshow` panel has a colorbar**, its height matched *exactly* to
   the image axes (`qafig.imshow` / `qafig.colorbar` -- an inset axes glued
   to the parent, exact for any aspect ratio), labelled with the data unit.
   Automatic display limits are robust: the upper percentile is capped at the
   brightest 3x3 *neighbourhood* median, so isolated hot pixels never set
   `vmax` (explicit `vmin`/`vmax`/`norm` override everything).
5. **Collapsing a datacube along the spectral axis is always nan-aware**
   (`qafig.collapse` uses `nanmedian`/`nanmean`, never `median`/`mean`;
   `min_coverage` blanks spaxels finite in too few planes -- footprint-edge
   medians of a handful of values would otherwise dominate the limits).
6. **One palette everywhere.** Images use cmasher's `torch` colormap
   (`qafig.CMAP`). The only line of a plot -- or the main product among
   several (e.g. the stitched spectrum over its segments) -- is black
   (`qafig.MAIN_COLOR`). Families of equivalent lines get
   `qafig.line_colors(n)`, sampled from the same colormap. Overlays on
   images (contours, aperture circles, markers) cycle through
   `qafig.OVERLAY_COLORS`, chosen for contrast on `torch`.
7. **Figures come from QA steps only** (`level = "qa"`), never from data
   steps. Data steps record what a figure would need *inside their products*
   (the mask contract, extra QA extensions, JSON keys); a QA step reads the
   product and draws. Because every QA stage writes into
   `qa/<step name>/`, the subdirectory name always says which step produced
   a figure.
8. **Axis labels always carry their unit in square brackets**:
   `wavelength [um]`, `flux density [mJy]`, `dRA [arcsec]`, `x [pix]`.
   `qafig` provides the common ones (`WAVE_LABEL`, `FLUX_LABEL`, `SB_LABEL`).
9. **Spectra use a logarithmic wavelength axis by default**
   (`qafig.set_wave_scale(ax)` -- plain numbers as tick labels, not powers of
   ten). Steps that plot spectra expose an `xscale` parameter so a single
   figure can be switched back to `linear` in the workflow.
10. **Spectral features are annotated through one helper**
    (`qafig.annotate_features(ax, features, wave_min=..., wave_max=...)`),
    driven by the same `features:` grammar the data steps use
    (`jwstflow.features`): gas lines become short labelled ticks in a lane at
    the top of the axes, emission bands (PAH & co.) and ice bands become
    shaded wavelength spans in two lanes below, every label hanging from a
    common line beneath the lanes, color-coded by class
    (`qafig.FEATURE_COLORS`). Steps that plot spectra expose a `features`
    parameter (default off; `all` = every bundled dataset in range).

## Writing a QA step

```python
from jwstflow import RunContext, Step, qafig

class PlotMyProduct(Step):
    """One figure per *_myprod.fits: the spectrum over its collapsed cube."""

    level = "qa"                       # -> qa/plot_my_product/
    inputs = ("*_myprod.fits",)

    def run(self, inputs, ctx: RunContext, *, dpi: int = 150, **params):
        out = []
        for path in inputs:
            spec = ...                                   # wavelength [um], flux + TUNIT
            flux, ylabel = qafig.to_mjy(spec.flux, spec.unit)          # rule 2
            fig, (a, b) = qafig.subplots(1, 2, figsize=(10, 4))
            qafig.imshow(a, qafig.collapse(cube), unit="MJy/sr",       # rules 4 + 5
                         stretch="log")
            a.set(xlabel="x [pix]", ylabel="y [pix]")                  # rule 8
            qafig.step(b, spec.wavelength, flux, color=qafig.MAIN_COLOR)   # rules 3 + 6
            b.set(xlabel=qafig.WAVE_LABEL, ylabel=ylabel)
            qafig.annotate(b, path.name)                               # rule 1: no titles
            qafig.figlegend(fig)
            out.append(qafig.save(fig, ctx.output_dir / f"{path.stem}.png", dpi=dpi))
        return out
```

Guidelines that follow from the rules:

* **Never plot from a data step.** If a data step computes something worth
  seeing (a sky mask, per-star offsets), write it into the product (an extra
  FITS extension, a JSON key) and give it a QA step. `jwstflow` itself
  follows this: `stitch_segments` writes the contract ECSV and the
  `plot_stitch` QA step draws it.
* **Identify the product in the legend**, not in a title:
  `qafig.annotate(ax, path.name)`.
* Use `qafig.contour_proxy(ax, color, label=...)` to give contours a legend
  handle.
* Finish with `qafig.save(fig, path)` -- it uses a tight bounding box so
  outside legends and colorbar labels are never clipped, and closes the
  figure.

## QA steps shipped with jwstflow

| step | inputs | figure |
|---|---|---|
| `plot_spectrum` | `*_x1d` / `*_c1d` / `*_s1d(comb)` | every EXTRACT1D/COMBINE1D spectrum, in mJy; with 2+ spectra also a combined log-log overview (`<stage>_all.png`) |
| `quicklook_image` | anything with a SCI extension | nan-median collapsed image with colorbar |
| `plot_stitch` | `*_s1dcomb.ecsv` | stitched spectrum (black) over its rescaled segments; when the stitch rescaled anything, also the segments as extracted (`*_unscaled.png`); with 2+ stitched spectra a combined log-log overview (`<stage>_all.png`) |
| `plot_stitch_overlaps` | `*_s1dcomb.ecsv` | one panel per neighbouring segment pair, zoomed into their overlap (`*_overlaps.png`): both segments with their error bands, the crossover wavelength (dotted), the shaded window around it from which the multiply factor was measured, the bluer segment times that factor (dashed), and the factor with its 1-sigma uncertainty in the legend |
| `plot_stitch_background` | `*_s1dcomb.ecsv` | source / background / difference: the background the extraction subtracted (BACKGROUND columns of the segment files), reassembled with the stitch's scales and crossovers, under the stitched spectrum with and without it (`*_bkgcomp.png`); feature lanes annotated by default |
| `mast_compare` | `*_s3d` / `*_x1d` | jwstflow vs. MAST spectra and their ratio |
| `qa_spaxel_clusters` | `*_clustermask.fits` | per band and region: one row per dither cube, one column per wavelength plane around the window (`*_<region>_clusters.png`); the search aperture dashed on every panel, the flagged spaxels solid on included dithers; the legend states each dither's decision with its numbers |
