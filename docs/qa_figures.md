# QA figures: the style guide

Every QA figure -- from jwstflow's own steps and from contributed step
packages (`jwstflow-midas`, `jwstflow-joys`, ...) -- follows one standard, so
a reduction's `qa/` tree reads as a single, consistent report. The standard
is *set here* and *implemented* by the `jwstflow.qa` package: QA steps build
their figures through it instead of using matplotlib directly. If a rule
cannot be followed with what `jwstflow.qa` offers, extend the package -- do
not work around it in a step.

Almost every QA figure is one of **two kinds**, and the package has a class
for each -- build on them (or subclass them) rather than repeating the calls:

* a **spectrum-like** figure -- a flux-like quantity over wavelength, both
  axes logarithmic, the 1σ band of its lines, the spectral features
  annotated: `qa.Spectrum` (rules 5-10);
* a **frame-like** panel -- an image on its pixel grid with pixel and sky
  offsets from one zero, a colorbar, masks and markers on top: `qa.Frame`
  (rules 11-15).

This guide has four parts: the rules, the palette and type, how to write a
new QA step (with the layout of the package), and the checklist to run
before a figure is committed.

## 1. The rules

### Structure

1. **No titles.** Figures and subplots never carry a title. Whatever a title
   would have said goes into the legend: `qa.annotate(ax, text, ...)`
   records note lines (the product name first, then counts and parameters)
   and `qa.figlegend(fig)` places one combined legend *outside* the axes,
   above the figure -- the notes as its first column in grey, then the
   marks. The legend chooses its number of columns so that it fits the
   figure width; long notes are wrapped (never inside a file name).
2. **What is common goes into the legend, what belongs to one panel goes
   into the panel.** In a multi-panel figure the legend explains the marks
   all panels share; a panel's own label (which wavelength, which segment
   pair, a measured value) is `qa.tag(ax, text)` on an image and
   `qa.panel_note(ax, text)` on a line plot.
3. **Figures come from QA steps only** (`level = "qa"`), never from data
   steps. Data steps record what a figure needs *inside their products*
   (the mask contract, extra extensions, JSON keys, the mask bits of a
   table); a QA step reads the product and draws. Every QA stage writes
   into `qa/<step name>/`, so the directory names the step.
4. **One figure, one question.** Two quantities of different scale get two
   panels sharing the wavelength axis, never a second y axis.

### Spectra and other line plots

5. **Flux-like quantities are plotted in mJy**, surface brightness in
   mJy arcsec⁻². `qa.to_mjy(values, unit)` converts from whatever unit a
   product carries (pass the FITS `TUNIT`/`BUNIT` string) and returns the
   finished axis label.
6. **Line-style plots use `qa.step`** (`ax.step(..., where="mid")`):
   spectra are histograms over wavelength bins. The line is **broken at
   gaps**: non-finite samples and missing stretches of the wavelength grid
   (detector gaps, masked ranges that were dropped from a table) are never
   bridged. A change of sampling -- the junction of two gratings -- is not a
   gap.
7. **Uncertainties are shaded bands** in the color of their line:
   `qa.error_band(ax, x, y, error)` (or `qa.band` for an arbitrary
   range). Labelled once, as `qa.SIGMA_BAND_LABEL`.
8. **Spectrum-like figures have logarithmic axes by default**, wavelength
   (`qa.set_wave_scale(ax)`) and flux (`qa.set_flux_scale(ax, scale)`): a
   log flux axis has no negative range to waste. A quantity that scatters
   around zero (a background, a residual, a ratio) keeps a linear axis.
   Log axes carry plain numbers as tick
   labels (`0.3`, `1`, `3` -- not `3 × 10⁰`), placed so that they never
   crowd; a log axis spanning less than a factor of five gets evenly spaced
   ticks; an axis leaving 0.001-10⁵ switches to powers of ten throughout.
   The automatic limits of a log flux axis ignore single near-zero samples.
   Steps that plot spectra expose `xscale` (and `yscale`) parameters.
9. **Reference lines and windows are recessive**: `qa.refline` (zero,
   unity, thresholds, crossovers) and `qa.span` (a shaded wavelength
   window in the color of its role) are drawn behind the data.

### Spectral features

10. **Features are annotated through one helper**,
    `qa.annotate_features(ax, features, wave_min=..., wave_max=...)`,
    driven by the `features:` grammar of `jwstflow.spectra.features`. The spectrum
    stays the main element of the figure, and a peak is identified by the
    label standing over it:

    * **Every line in range is marked, whether the spectrum shows it or
      not** -- the figure tells where lines *would* be. No label is dropped
      to make room: a figure with more labels gets wider.
    * **Each gas label stands at its own line**, a constant factor
      (`label_offset`) above the spectrum there, with a stem to the line and
      in the **color of the species** (`qa.SPECIES_COLORS`) -- the labels
      follow the spectrum instead of forming a row. A line that towers over
      its neighbours keeps its label on top of itself and the neighbours'
      labels **dodge left or right** of it. Labels that would touch are
      moved apart; where lines crowd so much that one row of labels would
      have to move far from them, the labels alternate between two tiers.
      Narrow bands of the gas catalogue (Q-branches) count as lines. Strong
      peaks that no catalogue line or gas band explains are marked `?`
      (`qa.detect_lines`).
    * **Series** (the H I series, the branches of the H₂ bands) are
      labelled line by line too. Only where consecutive members lie closer
      than a label is wide do they share one label for the range
      (`H I Pf 17–25`, `H₂ 1-0 Q(1–8)`), with a stem from every member --
      no brackets, no combs in the overview.
    * **Broad gas emission** (the CO fundamental, OH and water forests) is
      marked above the plot: outlined rectangles over the wavelength range,
      in the species color.
    * **Photospheric lines** of the star (the opt-in `photospheric_lines`
      dataset; `photospheric: false` leaves them out) mirror the gas lines:
      dashed, labelled right *below* their absorption line, in one family
      of greens (`qa.PHOTOSPHERE_COLORS`). Lines of one species that
      stand side by side are one multiplet: a single label (`Na I doublet`,
      `Ca I triplet`, `Ca I (×4)`) with a stem from every line.
    * **Solid-state bands** (ices, silicates, PAH) are rectangles in compact
      rows at the foot of the plot area, colored by group
      (`qa.BAND_COLORS`). **All bands of one species share a row.** The
      label is the typeset formula of the species (H₂O, ¹³CO₂, NH₄⁺ -- no
      "ice"), beside its rectangle where the row has room. The rows share
      the room under the spectrum with the photospheric labels: a
      photospheric label stands clear of the bands (moved aside on a curve),
      and one with no room beside them is put above the spectrum instead.
    * **One frame, one wavelength axis**, and **both axes logarithmic** by
      default: a log flux axis has no negative range to spend on labels.
    * **No legend entries** for the annotation: every mark carries its own
      label in the color of its class.
    * **Colors come from the house colormaps**: the gas species from
      cmasher *torch* (H₂, noble gases, other atoms and ions, He I, H I)
      with *ember* and *flamingo* for the molecules; the photosphere from
      the greens (*emerald*, *jungle*, *swamp*, *toxic*); the ices from the
      cool maps (*freeze*, *amethyst*, *lavender*, *sapphire*), silicates
      grey, PAH from *sunburst*.
    * **Nothing crosses, nothing overlaps.** A label over its own line
      stands right above the spectrum there. Labels that would touch are
      moved apart and each is joined to its line by one smooth curve that
      leaves the line vertically -- no kinks; the labels of such a run
      share one level and keep the order of their lines, so the curves
      cannot cross.
    * `series="combs"` is the other layout, for a separate figure: every
      series with several members in view is one named row above the plot,
      a tick and a number per member -- the compact inventory
      (`plot_stitch` writes it as `*_series.png`).
    * The y range gains only the room the labels need; the data area never
      falls below its designed height, and the figure stays **wider than
      tall**.

    The line catalogues are generated from primary lists by
    `scripts/build_line_tables.py` (H I: Rydberg formula; H₂ and CO
    bandheads: HITRAN; atoms and ions: NIST ASD Ritz wavelengths; all
    vacuum). Add lines there, not by hand. `gas_lines` is the set data
    steps mask and is not extended lightly; lines that are only worth
    labelling go to `gas_lines_extended`, which figures annotate together
    with it (and `photospheric_lines`) but no data step uses unless named.

    Points of a curve (band peaks, measured positions) are labelled with
    `qa.callouts(ax, x, y, texts)`: rotated labels hanging below the
    curve, spread and tied to their points. Never place label text by hand
    with fixed offsets.

### Images

11. **Every image is `qa.imshow`**: `origin="lower"`, nearest-neighbour
    pixels, robust display limits (`qa.robust_limits`: percentiles of
    the finite pixels, the upper one capped at the brightest 3×3
    neighbourhood so that isolated hot pixels never set it), and one of
    four stretches. `asinh` is the default choice for sky images -- linear
    around the background, logarithmic above, and zero or negative pixels
    stay visible; `log` for images that are positive throughout (zero and
    negative pixels take the lowest color: they are data, not gaps);
    `linear` and `sqrt` for low dynamic range. Signed images (differences,
    residuals) use `center=0` and the diverging map.
12. **Missing data are shown as missing.** NaN pixels are drawn in a neutral
    grey (`qa.BAD_COLOR`) that no colormap contains -- never in the
    lowest color. Spectral collapses are nan-aware (`qa.collapse`;
    `min_coverage` blanks spaxels finite in too few planes).
13. **Every image has a colorbar** as tall as the image and of one fixed
    physical width (`qa.colorbar`, called by `imshow`), labelled with
    the quantity and unit. It gets a pointed end wherever data lie beyond
    the display limits, and plain-number ticks under the log and asinh
    stretches. Panels that share one norm (`qa.stretch_norm`) share one
    colorbar per row.
14. **Every frame carries two sets of axes with one common zero.** Offsets
    in pixels at the bottom/left (`Δx [pix]`, `Δy [pix]`) and offsets on the
    sky in arcsec at the top/right, both measured from the same reference
    position -- the target, else the image centre -- so the zeros of the
    two sets coincide: `qa.Frame().image(..., sky=qa.SkyFrame.from_header(
    header, shape, primary=primary))`. Sky-aligned images are labelled
    Δα cos δ / Δδ with East to the left; any other orientation gets Δx / Δy
    and a compass. The legend states the reference position and the array
    pixel the zero sits at (`frame.note()`), so an offset is turned back into
    an array index by adding it. The image stays in pixel coordinates, so
    overlays are drawn in pixels. In a grid every panel keeps its ticks on
    all four sides and the tick labels are written once, on the outer panels
    (`sides=`). Only an image without a celestial WCS (a detector image)
    has plain pixel coordinates, and only panels that are not frames on the
    sky (thumbnails) leave the axes to the step (`sides=None`).
15. **Overlays**: spaxel masks are outlined (`Frame.mask`: solid for the
    primary mask, dashed for the secondary), positions marked
    (`Frame.mark`), both in `qa.OVERLAY_COLORS`.

### Labels and type

16. **Axis labels are `symbol [unit]`**: a quantity is named by its symbol,
    not in words -- `$F_\nu$ [mJy]`, `$\lambda$ [µm]`, `$I_\nu$
    [mJy arcsec⁻²]`, `$\tau$`, `$F_\nu / \sigma_{F_\nu}$` -- taken from the
    constants of `jwstflow.qa` (`FLUX_LABEL`, `WAVE_LABEL`, `SB_LABEL`,
    `TAU_LABEL`, `SNR_LABEL`, `X_PIX_LABEL`, `DRA_LABEL`, ...) or built
    with `qa.label(symbol, unit)`. Words only where no symbol exists
    (a fraction of planes, a count). Dimensionless quantities carry no
    brackets.
17. **Symbols and units are mathtext**, never ASCII stand-ins: `qa.UM`
    (µm), `qa.um(4.05)`, `qa.um_range(lo, hi)`, `qa.times(1.02)`
    (× 1.020), `qa.pm(value, error)`, `$\tau$`, `$\sigma$`, a real minus
    sign. This holds for legend entries and notes as well as for axes.
18. **One set of font sizes and line widths** (`qa.FONT`, `qa.LW`,
    applied through `qa.STYLE` by `qa.subplots` / `qa.figure`):
    axis labels 9 pt, tick labels and legend 8 pt, in-panel notes 7 pt,
    feature labels 5.5 pt; ticks point inwards on all four sides. Steps
    never pass `fontsize`, and take line widths from `qa.LW`.
19. **Standard figure sizes** (`qa.FIGSIZE`): a spectrum is 11 in wide
    (`spectrum`, `spectrum2` with a second panel), a single image panel is
    `image`. Grids state their panel size as a parameter.

## 2. Palette

| use | color | constant |
|---|---|---|
| images | cmasher `torch` | `CMAP` |
| signed images | cmasher `fusion_r` | `DIVERGING_CMAP` |
| no data | grey 0.80 | `BAD_COLOR` |
| the product / main line | black | `MAIN_COLOR` |
| its input, drawn underneath | grey 0.62 | `CONTEXT_COLOR` |
| reference lines | grey 0.55 | `REF_COLOR` |
| an *ordered* family of lines (bands, segments, slits) | sampled from `torch` | `line_colors(n)` |
| marks with a role | see below | `ROLE_COLORS` |
| other categorical marks | Okabe-Ito, in fixed order | `ACCENTS` |
| overlays on images | cyan, lime, magenta, white | `OVERLAY_COLORS` |
| spectral feature classes | grey / orange / violet / green | `FEATURE_COLORS` |

Color follows meaning, not order of appearance. A mark with a role has the
same color in every figure (`ROLE_COLORS`): `model` (a model or fit over the
data), `background` (sky estimates; the plum of the middle of `torch`), `excluded` (samples left out of a fit),
`censored`, `bridged` (interpolated stretches), `window` (a measurement
window), `anchor` (fit stretches), `ice`, `emission`, `line`, `photosphere`.
A package with further roles maps them onto these once, in one table
(`jwstflow_midas.qa.continuum.COLORS`), and never writes a color literal in a step.
Never use `OVERLAY_COLORS` on a white background or `ACCENTS` on an image,
and never rely on color alone: marks that differ in role also differ in
marker or line style.

## 3. Writing a QA step

### The package

| module | what it holds |
|---|---|
| `qa.style` | palette, role colors, font sizes, axis-label constants, `subplots` / `figure`, `annotate`, `figlegend`, `save` |
| `qa.units` | `to_mjy` |
| `qa.lines` | `step`, `band`, `error_band`, `refline`, `span`, `set_wave_scale`, `set_flux_scale` |
| `qa.frames` | `SkyFrame`, `imshow`, `sky_axes`, `colorbar`, `collapse`, `robust_limits`, `stretch_norm`, `contour_mask`, `marker`, `tag`, `panel_note` |
| `qa.annotation` | `annotate_features` (records what to mark) |
| `qa.labels`, `qa.placement`, `qa.layout` | the parts of the annotation: label text and species colors and the line search; geometry along one axis; `FeatureLayout`, which places everything when the figure is saved, one method per phase |
| `qa.plots` | **`Spectrum`** and **`Frame`** |
| `qa.io` | `find_file`, `read_spectrum`, `read_background`, `read_cube`, `ancestor`, `short_labels` |
| `qa.steps` | the steps shipped with jwstflow: `spectra`, `stitch`, `images`, `cubes`, `cube_stitch`, `psf`, `clusters`, `gaps`, `mast_compare`, `summary` |

Everything public is importable from `jwstflow.qa` itself
(`from jwstflow import qa`).

### A spectrum-like figure

```python
from jwstflow import RunContext, Step, qa

class PlotMyProduct(Step):
    """One figure per *_myprod.fits: the spectrum over the model it was fitted with."""

    level = "qa"                       # -> qa/plot_my_product/
    inputs = ("*_myprod.fits",)
    version = "1"                      # bump whenever the figure changes

    def run(self, inputs, ctx: RunContext, *, xscale: str = "log", yscale: str = "log",
            features="all", dpi: int = 150, **params):
        out = []
        for path in inputs:
            spec = ...                                                  # read the product, nothing else
            flux, ylabel = qa.to_mjy(spec.flux, spec.unit)              # rule 5
            plot = qa.Spectrum(xscale=xscale, yscale=yscale)            # rule 8
            plot.line(spec.wavelength, qa.to_mjy(spec.model, spec.unit)[0],
                      color=qa.ROLE_COLORS["model"], context=True, label="model")
            plot.line(spec.wavelength, flux, error=qa.to_mjy(spec.error, spec.unit)[0],
                      label="spectrum", error_label=qa.SIGMA_BAND_LABEL)   # rules 6, 7
            out.append(plot.finish(ctx.output_dir / f"{path.stem}.png", ylabel=ylabel,
                                   notes=[path.name], features=features, dpi=dpi))   # rules 1, 10, 16
        return out
```

`Spectrum.line` draws a gap-aware step line with its 1σ band; a `context`
line (an input, a segment, a model) is thin and does not count for the
annotated range. `Spectrum(panels=2, height_ratios=[2, 1])` stacks a second
panel under the spectrum on the same wavelength axis (`plot.axes[1]`: a
ratio, a residual); `finish(..., wave_range=(lo, hi))` zooms. The feature
labels follow everything plotted and the lines of the product are searched
for peaks; a figure whose product is not the curve that carries the lines
names that curve with `follow=(wavelength, values)`.

### A frame-like figure

```python
            frame = qa.Frame()
            frame.image(qa.collapse(cube, min_coverage=0.5), unit="MJy/sr",          # rules 11-13
                        sky=qa.SkyFrame.from_header(header, cube.shape, primary=primary))   # rule 14
            frame.mask(aperture, label=f"aperture ({aperture.sum()} spaxels)")       # rule 15
            frame.mark(x_source, y_source)
            out.append(frame.finish(ctx.output_dir / f"{path.stem}.png", notes=[path.name], dpi=dpi))
```

`Frame.finish` closes the notes with the reference position of the offsets.
Both classes also draw onto an axes of a larger figure --
`qa.Frame(ax)`, `qa.Spectrum(ax)`, then `qa.annotate`, `qa.figlegend` and
`qa.save` for the figure as a whole (`qa_infield_background` in
jwstflow-midas: a frame next to a spectrum; `qa_cube_summary`: three frames
over two spectra; `qa_cube_slices`: a grid of frames with
`sides=` naming the labelled sides of each).

### A figure of your own kind

A package whose products need the same treatment in several figures
subclasses once and builds its steps on the subclass:

```python
class DiskFrame(qa.Frame):
    """The collapsed disk image under a log stretch."""

    def image(self, image, *, sky=None, **kwargs):
        kwargs.setdefault("unit", "MJy/sr")
        kwargs.setdefault("stretch", "log")
        return super().image(image, sky=sky, **kwargs)
```

(`jwstflow_midas.qa.disk.DiskFrame`; `jwstflow_midas.qa.continuum.
ContinuumSpectrum` does the same for a continuum under its spectrum.)

Guidelines that follow from the rules:

* **Never plot from a data step.** If a data step computes something worth
  seeing (a sky mask, per-star offsets), write it into the product and give
  it a QA step. Give image extensions the 2-D WCS of their cube
  (`jwstflow.cubes.masks.strip_wave_axis`), so the QA step can draw the sky axes
  from the product alone.
* **Read self-describing products, not code.** A figure should not need the
  module that wrote its product: take bit tables, units and parameters from
  the product's own header or metadata.
* **Always finish with `qa.save(fig, path)`.** It carries out the layout
  that needs the final geometry (label spreading, feature rows, tick
  placement), writes with a tight bounding box so that outside legends and
  colorbars are never clipped, and closes the figure.
* **Expose only what a user may need to change** as parameters (`xscale`,
  `yscale`, `stretch`, `features`, `dpi`), with the standard's defaults.
* **Skip, with a log line, what cannot be drawn** (a missing extension, a
  product without the needed column) instead of failing the run; return
  only the files written.
* **Test the step on real products before committing** and look at the
  result: `jwstflow run <workflow> --only <step>` renders one QA stage from
  existing products without touching any data stage.

## 4. Checklist for a new or changed figure

- [ ] No title anywhere; the product name is the first note of the legend.
- [ ] Every axis is labelled `symbol [unit]` from `jwstflow.qa`'s constants; no
      `um`, `sigma`, `x` or `+-` in any text.
- [ ] Fluxes are in mJy (mJy arcsec⁻²); lines are `qa.step`; errors are
      bands.
- [ ] Gaps in the data are gaps in the plot; NaN pixels are grey.
- [ ] The figure is built on `qa.Spectrum` or `qa.Frame` (or says why it
      is neither).
- [ ] Frames have pixel and arcsec offsets from one zero (or state why
      not), a colorbar with label, and a stretch chosen for their dynamic
      range.
- [ ] Colors come from `jwstflow.qa` by meaning; no color literals, no `fontsize`,
      no hand-placed label offsets in the step.
- [ ] Nothing overlaps: legend, labels, annotations, colorbar. Checked on a
      real product, at the narrowest and the widest wavelength range the
      step will see.
- [ ] The step's `version` is bumped and its docstring says what the figure
      shows; `docs/qa_figures.md` lists it.

## QA steps shipped with jwstflow

| step | kind | inputs | figure |
|---|---|---|---|
| `plot_spectrum` | spectrum | `*_x1d` / `*_c1d` / `*_s1d(comb)` | every EXTRACT1D/COMBINE1D spectrum in mJy with its 1σ band; with 2+ spectra also a combined log-log overview (`<stage>_all.png`); optional feature annotation |
| `plot_snr` | spectrum | `*_x1d` / `*_c1d` / `*_s1d(comb)` | signal-to-noise ratio per sample against wavelength (`*_snr.png`), median S/N in the legend, features annotated by default |
| `plot_spectrum_stages` | spectrum | the spectra of the last 1-D processing stage | the same spectrum at each named earlier stage, and the ratio of every stage to the one before it (`*_stages.png`): what defringing, cleaning, ... did |
| `plot_stitch` | spectrum | `*_s1dcomb.ecsv` | stitched spectrum (black) over its rescaled segments; with features also the series as combs (`*_series.png`); when the stitch rescaled anything, the segments as extracted (`*_unscaled.png`); with 2+ stitched spectra a combined overview |
| `plot_stitch_overlaps` | line panels | `*_s1dcomb.ecsv` | one panel per neighbouring segment pair, zoomed into their overlap (`*_overlaps.png`): both segments with error bands, the crossover, the window the factor was measured in, the bluer segment times that factor; pair and factor noted in each panel |
| `plot_stitch_background` | spectrum | `*_s1dcomb.ecsv` | source / background / difference (`*_bkgcomp.png`), features annotated by default |
| `quicklook_image` | frame | anything with a SCI extension | nan-median collapsed image (asinh), pixel and sky offsets from the target |
| `qa_cube_summary` | frames + spectra | `*_s3d.fits`, `*_s3dcomb.fits` | median image, coverage and signal-to-noise maps; under them, over the full width, the field-integrated flux and the filled fraction of the footprint against wavelength (`*_summary.png`) |
| `qa_cube_slices` | frames | `*_s3d.fits`, `*_s3dcomb.fits` | channel maps: the cube in `n_slices` wavelength bins on one display scale (`*_slices.png`) |
| `qa_psf_cube` | frames + lines | `*_psfcube.fits` | PSF slices across wavelength, FWHM against the diffraction limit, in-field energy |
| `mast_compare` | spectrum | `*_s3d` / `*_x1d` | jwstflow vs. MAST spectra and their ratio |
| `qa_spaxel_clusters` | thumbnails | `*_clustermask.fits` | per band and region: one row per dither cube, one column per wavelength plane (`*_<region>_clusters.png`) |
| `qa_gap_coverage` | frames | `*_gapcov.fits` | the coverage next to the NIRSpec detector gap on the sky: where all exposures stop before and after it, and how long the partially covered stretch beyond is (nm), per group of exposures |
| `qa_cube_stitch` | spectrum, line panels, frames | `*_s3dcomb.fits` | the figures of a joint cube (in development: not a science-ready product), from the records in the product: the flux over the field all segments cover -- the cube itself over its segments as joined, crossovers marked, features annotated (`*_spectrum.png`); one panel per neighbouring pair zoomed into the overlap, drawn like `plot_stitch_overlaps` (`*_overlaps.png`); one frame per pair, zoomed onto the source, with the bright core in both segments at the same wavelengths, the region the ratio was measured over and the measured shift between the segments tagged (`*_positions.png`); with two or more pairs those shifts against wavelength (`*_offsets.png`) |

All of them live in `jwstflow.qa.steps`, one module per kind of product
(`spectra`, `stitch`, `images`, `cubes`, `cube_stitch`, `psf`, `clusters`,
`gaps`, `mast_compare`, `summary`); workflows name them by their registered names.
