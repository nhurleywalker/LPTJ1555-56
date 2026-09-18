"""
Turn raw per-beam pickles into the primary-beam-corrected, averaged pickles
the rest of the analysis actually reads from `Paths.averaged_dir`. Slow (it
re-averages every beam of every observation), so:

- `average_askap_beams` / `correct_meerkat_beams` do the full batch job,
  skipping any output file that already exists unless `force=True` -- call
  these explicitly (or via `--askap-pb-corr` / `--mkt-pb-corr` /
  `--stage archival`) when the raw pickles change.
- `ensure_meerkat_file` generates *just* the one MeerKAT file the main
  analysis needs (`Paths.meerkat`), on demand, the first time it's missing
  -- `data_io.load_meerkat_data` calls it automatically.
"""
from __future__ import annotations

import glob
import os
import pickle
from pathlib import Path

import numpy as np

from .config import AnalysisConfig
from .primary_beams import GaussianPB, MKCosBeam, get_beam_pos, get_beam_pos_mkt


def average_askap_beams(cfg: AnalysisConfig, force: bool = False):
    """For every ASKAP SBID with multiple beam pickles in `dynspec_dir`,
    primary-beam-weight-average them together into a single
    `averaged_dir/SB{sbid:05d}.pkl`."""
    dynspec_dir, averaged_dir = cfg.paths.dynspec_dir, cfg.paths.averaged_dir
    os.makedirs(averaged_dir, exist_ok=True)

    all_pkls = sorted(glob.glob(f"{dynspec_dir}/science*.pkl"))
    sbids = np.unique([int(f.split("SB")[1][:5].replace("_", "")) for f in all_pkls])

    for sbid in sbids:
        outname = f"{averaged_dir}/SB{sbid:05d}.pkl"
        if os.path.exists(outname) and not force:
            continue
        print(f"Correcting SB{sbid}")
        dynspecs = glob.glob(f"{dynspec_dir}/*SB{sbid}*pkl")

        ds_example = np.load(dynspecs[0], allow_pickle=True)
        shape = ds_example["DS"].shape
        final_array = np.empty((*shape, len(dynspecs)))
        weights_array = np.empty((shape[1], len(dynspecs)))

        for i, dsfile in enumerate(dynspecs):
            ds = np.load(dsfile, allow_pickle=True)
            beam = dsfile.split("beam")[1][:2]
            # Survey name is always right after the SBID, e.g.
            # scienceData.VAST_1552-56.SB81799.VAST_1552-56.beam01_averaged_cal.leakage.pkl
            # scienceData_SB8676_RACS_1552-56A.beam01_averaged_cal.pkl
            # ...but sometimes separated by "_" and sometimes ".", hence the replace.
            survey = dsfile.split("SB")[1].replace("_", ".").split(".")[1]
            sep = get_beam_pos(survey, beam).separation(cfg.source.coord)
            pb_vals = [GaussianPB(frequency=f * 1e9).evaluate(sep.rad, freq=f) for f in ds["FREQS"]]
            weights_array[:, i] = pb_vals
            final_array[..., i] = ds["DS"]

        # Correcting for the primary beam means multiplying by 1/pb (bigger
        # for beams further from the source); weighting by the primary beam
        # means dividing by that same factor (big numbers are bad weights).
        # The two cancel, leaving a simple weighted average by the raw pb.
        weights_tiled = np.tile(weights_array[None, :, None, :], (shape[0], 1, shape[2], 1))
        ds_example["DS"][:, :, :] = np.nansum(final_array, axis=3) / np.nansum(weights_tiled, axis=3)

        # SBID 40625 was run through FixMS but the others weren't, so they
        # need a factor of 2 to match the FixMS Stokes convention. (Position
        # angles will still be wrong for those, but there's only noise in
        # this archival data, so it doesn't matter for upper limits.)
        if sbid != 40625:
            ds_example["DS"] *= 2

        with open(outname, "wb") as f:
            pickle.dump(ds_example, f)


def correct_meerkat_beams(cfg: AnalysisConfig, cbids=None, force: bool = False):
    """Primary-beam correct (cosine beam model) MeerKAT observations,
    writing `averaged_dir/CB{cbid:010d}.pkl`. If `cbids` is given, restrict
    to those coherent-beam IDs; otherwise process everything found in
    `dynspec_dir`."""
    dynspec_dir, averaged_dir = cfg.paths.dynspec_dir, cfg.paths.averaged_dir
    os.makedirs(averaged_dir, exist_ok=True)

    for dsfile in sorted(glob.glob(f"{dynspec_dir}/1*pkl")):
        cbid = int(Path(dsfile).stem[:10])
        if cbids is not None and cbid not in cbids:
            continue
        outname = f"{averaged_dir}/CB{cbid:010d}.pkl"
        if os.path.exists(outname) and not force:
            continue
        print(f"Correcting CB{cbid}")

        ds = np.load(dsfile, allow_pickle=True)
        sep_deg = get_beam_pos_mkt(cbid).separation(cfg.source.coord).deg
        pb_vals = np.array([MKCosBeam(sep_deg, f) for f in ds["FREQS"]])
        ds["DS"] /= np.tile(pb_vals[None, :, None], (ds["DS"].shape[0], 1, ds["DS"].shape[2]))

        with open(outname, "wb") as f:
            pickle.dump(ds, f)


def ensure_meerkat_file(cfg: AnalysisConfig):
    """If `cfg.paths.meerkat` doesn't exist yet, generate it by running the
    MeerKAT primary-beam correction for just that one observation. Raises a
    clear error if the raw pickle isn't there either."""
    path = cfg.paths.meerkat
    if os.path.exists(path):
        return

    stem = Path(path).stem  # e.g. "CB1652551867"
    if not stem.startswith("CB"):
        raise FileNotFoundError(
            f"{path} doesn't exist, and I don't know how to generate it "
            f"(expected a 'CB<cbid>.pkl'-style filename)."
        )
    cbid = int(stem[2:])
    print(f"{path} not found -- running MeerKAT primary-beam correction for "
          f"CB{cbid} first...")
    correct_meerkat_beams(cfg, cbids=[cbid])

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not generate {path}. Is the raw pickle for CB{cbid} "
            f"present in {cfg.paths.dynspec_dir}?"
        )
