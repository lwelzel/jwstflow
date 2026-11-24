# Archival data next to a reduction

Measurements from outside JWST -- broadband photometry from the literature, archival spectra -- are not
fetched by a recipe. They are curated by hand, once per target, and stored as ECSV files in

    <data root>/<target>/archival/*.ecsv

next to `raw/` and `mast_reference/` (the target directory is the parent of a run's `root`), so a
re-reduction never removes them. The QA step `plot_archival` draws them over a run's spectra.

## File layout (`jwstflow.spectra.archival`)

Every file says what it is in `meta['kind']` and where its numbers come from (`description`,
`source_file`, `sources` with one entry per reference).

**Photometry** (`kind: photometry`), one row per measurement:

| column        | unit      | meaning |
|---------------|-----------|---------|
| `WAVELENGTH`  | um        | reference wavelength of the band |
| `BANDWIDTH`   | um        | full width of the band (NaN where unknown) |
| `FLUX`        | Jy (F_nu) | flux density; for an upper limit the limit itself |
| `FLUX_ERROR`  | Jy        | 1-sigma uncertainty (for an upper limit the 1-sigma noise) |
| `UPPER_LIMIT` | bool      | `True`: `FLUX` is an upper limit at `meta['upper_limit_sigma']` sigma |
| `REFERENCE`   |           | citation key of the source (e.g. the bibtex key) |
| `CITATION`    |           | the source as text, `Author et al. Year` -- the legend groups by it |
| `NOTE`        |           | free text (caveats); optional |

Units are read from the columns: any F_nu (or F_lambda) unit astropy converts to Jy, any length for the
wavelength. A file with upper limits must state their significance in `meta['upper_limit_sigma']`.

**Spectrum** (`kind: spectrum`): `WAVELENGTH`, `FLUX` and optionally `FLUX_ERROR`, with
`meta['citation']` / `meta['reference']` (and the epoch, where known).

`jwstflow.spectra.archival.photometry_table(...)` builds a valid photometry table from plain arrays
(fluxes in mJy); `read_archival(path)` reads and checks a file.

## Adding data for a target

1. Collect the measurements with their references; keep the original table (e.g. the CSV) in
   `archival/source/` for provenance.
2. Convert it with `photometry_table` (or write the ECSV by hand) into `archival/<target>_<what>.ecsv`;
   record in the metadata what was left out and why.
3. Add the step to a recipe, e.g. the combination of a target:

   ```yaml
   - step: plot_archival
     inputs:
       - {stage: stitch_segments-epochs, pattern: "*_s1dcomb.ecsv"}
     parameters:
       archival: ["archival/*.ecsv"]     # default; relative to the target directory
     tags: [qa]
   ```

The archival files are parameters, not inputs of the step: after editing them, re-run it with
`jwstflow run <recipe> --tag qa --force` (or `--only plot_archival --force`).

## Figures

`<spectrum>_archival.png` over the spectrum's range and `<spectrum>_archival_sed.png` over everything the
archival data cover. The spectrum is the black line; each source of photometry has one marker and color
(no circles: those are kept for JWST photometry hexagons in paper figures); 1-sigma errors are I-shaped
bars, band widths thin horizontal bars; upper limits are hollow markers with a downward arrow at the
limit, their significance in the legend.
