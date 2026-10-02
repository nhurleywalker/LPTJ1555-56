# MKTJ1555-56 analysis code

Code for the analysis and figures in the MKTJ1555-56 paper: ASKAP (EMU) +
MeerKAT observations of the source, plus the archival non-detection light
curves and upper limits from every other observation of the field. 
It runs through a single entry point, `run_analysis.py` 
The dynamic spectra need to be downloaded separately from [Zenodo](https://doi.org/doi:10.5281/zenodo.22822831) as they
are too large for GitHub to host.

## Layout

```
run_analysis.py         single CLI entry point for everything
mktj1555/
  config.py              every path, constant, and analysis on/off switch
  timing.py              ephemeris / phase-folding helpers
  plotting.py            shared plot helpers (dynamic spectra, light curves)
  fitting.py              power-law spectral fitting
  detrend.py              MeerKAT baseline-ripple removal (shared by the
                           main analysis and the archival stage)
  data_io.py              loads the pickles, applies RFI flags, forms Stokes
                           I/Q/U/V, does primary-beam and parallactic-angle
                           correction for the two detection observations
  preprocessing.py        turns raw per-beam pickles into the primary-beam-
                           corrected/averaged ones everything else reads
  primary_beams.py        Gaussian/cosine primary-beam models + beam lookup
                           tables
  stages/                 one module per analysis stage (see below),
                           including `archival.py` (the former
                           archival_lightcurve.py)
```

## Requirements

```
numpy, scipy, pandas, matplotlib, astropy, astroplan
```

## Running the main analysis

```
python run_analysis.py --stage all
```

or a subset, e.g.:

```
python run_analysis.py --stage dynspec lightcurves fold
```

Available stages (see `run_analysis.py --help` or `STAGE_CHOICES` in that
file): `dynspec`, `lightcurves`, `paper-plots`, `fold`, `polarization-angle`,
`fold-main-pulse-spectrum`, `phase-scan`, `interpulse-spectrum`,
`joint-iquv`, `joint-spectrum`, `joint-spectrum-v`, `spiky-spectrum`,
`second-ip-v`, `second-ip-i`, `band-split`, `acf`, `archival`.

By default (no `--stage` given) only `acf` runs, matching the state the
original script was last committed in (`makeACF = True`, everything else
`False`). The folded MeerKAT light curve and the per-pulse ephemeris-check
plot always run, since they were unconditional in the original script too.
**`--stage all` includes `archival`** (the full non-detection pipeline over
every other observation of the field), which needs the raw per-observation
pickles for the whole archive -- if you just want the two detection
observations, name the stages you want explicitly instead of using `all`.

The main analysis reads a single primary-beam-corrected MeerKAT pickle
(`averaged_dynspec/CB1652551867.pkl`). `data_io.load_meerkat_data`
checks whether that file exists and, if not, generates it on the spot from
the raw pickle in `--dynspec-dir` (via `preprocessing.ensure_meerkat_file`).
You'll see a line like:

```
averaged_dynspec/CB1652551867.pkl not found -- running MeerKAT primary-beam
correction for CB1652551867 first...
```

the first time you run any stage that needs it. After that, the file
exists and subsequent runs load it directly.

Useful flags:

```
--askap-beam09 PATH     override the beam09 ASKAP pickle
--askap-beam15 PATH     override the beam15 ASKAP pickle
--meerkat PATH          override the MeerKAT pickle
--dynspec-dir DIR       raw per-observation pickles (default: dynspec)
--averaged-dir DIR      primary-beam-corrected/averaged pickles (default: averaged_dynspec)
--output-dir DIR        base directory for figures/tables (default: .)
--png-dir DIR           subdirectory (under --output-dir) for PNGs / sanity-check
                        plots (default: test_plots)
--pdf-dir DIR           subdirectory (under --output-dir) for PDFs / paper-quality
                        plots (default: paper_plots)
--beam-info-dir DIR     directory with *_beam_table.fits / *_pointings.txt
                        (default: ./beam_info, or set BEAM_INFO_DIR env var)
--askap-pb-corr         force-regenerate ASKAP beam averaging (archival stage), even
                        for SBIDs that already have an averaged pickle
--mkt-pb-corr           force-regenerate MeerKAT beam correction (archival stage), even
                        for observations that already have a corrected pickle
--spec-index, --cal-freq-ghz   flux-density scaling for the archival stage
```

All the paths, the ephemeris, the source/beam coordinates, the RFI flags,
and the phase windows used to isolate the main pulse/interpulse live in
`mktj1555/config.py` -- that's the file to edit if any of those need to
change for a re-analysis.

## Running the archival (non-detection) light curves

```
python run_analysis.py --stage archival
```

The `archival` stage always runs the beam-averaging/correction
preprocessing first, but skips any observation that's already been
processed (i.e. already has a pickle in `--averaged-dir`), so repeat runs
are fast. Pass `--askap-pb-corr`/`--mkt-pb-corr` to force everything to be
regenerated -- e.g. after the raw `dynspec/` pickles change.

Produces `observations_stacked.{png,pdf}` (per-observation folded light
curves), `Archival_upper_limits.pdf`, `Archival_measurements.csv`, and
`obs_table.tex`.

## Notes from the Claude refactor

- **Output files are now split by type**: PNGs go to `test_plots/`, PDFs go
  to `paper_plots/` (both under `--output-dir`), everything else (text
  tables, CSVs) stays directly under `--output-dir`. Override the
  subdirectory names with `--png-dir`/`--pdf-dir` if you'd rather call them
  something else.
- **The main-pulse-width calculation in `fold_askap` was fixed**: it now
  reads `len_mp = (1 + phase_end - phase_start) * 2 * P * 24 * 3600`
  (previously `phase_start - phase_end`, which gave an implausibly large
  ~7000s width close to the whole rotation period). This only affects the
  printed "Main pulse is about ...s wide" line and the numbers derived from
  it -- it doesn't change any of the folded data itself.

- **The original `makeRM` flag never did RM synthesis.** The code behind it
  was the folded main-pulse Stokes I/Q/U spectrum and power-law fit -- the
  flag name was simply stale. It's been renamed to `fold-main-pulse-spectrum`
  here. RM Synthesis is performed separately by RM-Tools.

- A handful of near-identical ~130-line blocks in the original main script
  (the main-pulse "spiky" spectrum, the second interpulse fit two ways, the
  joint ASKAP+MeerKAT spectrum in Stokes I and V) have been consolidated
  around shared helpers in `fitting.py`. The numerical recipe for each is
  unchanged; only the boilerplate around it was deduplicated.

- **The MeerKAT baseline-ripple removal (`detrend.py`) used to be
  implemented twice** -- once in the main script (for the discovery
  observation) and again, separately, in `archival_lightcurve.py` (for
  every other MeerKAT observation) -- both hardcoding "exactly 3 time gaps,
  4 segments". They're now one function, generalised to however many
  segments a given observation actually has. This is a behaviour change
  for any archival MeerKAT observation that doesn't have exactly 4
  segments: the original would have raised an `IndexError` on those; this
  version detrends them instead. Worth checking whether any of your
  archival observations previously silently relied on/worked around that.
- All commented-out historical code (abandoned calculations, alternative
  beam combinations, one-off sanity-check plots that were never re-enabled)
  has been deleted, per your instruction -- it's still in this codebase's
  git history if you need to recover any of it. Two unused imports
  (`MKCosBeam`, `get_beam_pos_mkt` in the main analysis script) and one
  unused variable (`a = 6`, an ASKAP aperture that was never actually used
  -- `primary_beams.GaussianPB` defaults to 12 m) were also removed.
- `stages/archival.py`'s per-file sanity-check figure is now closed
  (`plt.close(fig)`) after saving instead of left open -- across ~130
  files the original would accumulate open matplotlib figures in memory
  over a single run.

