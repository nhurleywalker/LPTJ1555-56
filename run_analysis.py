#!/usr/bin/env python
"""
Analysis of the ASKAP (EMU) + MeerKAT observations of MKTJ1555-56.

Run with no arguments for just the ACF figure and the two always-on
overview plots; pass --stage all to run everything, or name individual
stages.

Examples:
    python run_analysis.py --stage all
    python run_analysis.py --stage dynspec lightcurves
    python run_analysis.py --stage fold --output-dir figures/
"""
from __future__ import annotations

import argparse
import warnings

import matplotlib.pyplot as plt

from mktj1555.config import AnalysisConfig

STAGE_CHOICES = [
    "dynspec", "lightcurves", "paper-plots", "fold", "polarization-angle",
    "fold-main-pulse-spectrum", "phase-scan", "interpulse-spectrum",
    "joint-iquv", "joint-spectrum", "joint-spectrum-v",
    "spiky-spectrum", "second-ip-v", "second-ip-i", "band-split", "acf",
    "archival",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", nargs="+", choices=STAGE_CHOICES + ["all"], default=["acf"],
                    help="Which analysis stage(s) to run (default: acf). Pass 'all' to run "
                         "everything.")
    p.add_argument("--askap-beam09", help="Path to the beam09 ASKAP pickle")
    p.add_argument("--askap-beam15", help="Path to the beam15 ASKAP pickle")
    p.add_argument("--meerkat", help="Path to the MeerKAT pickle")
    p.add_argument("--dynspec-dir", help="Directory containing raw per-observation pickles "
                    "(used by preprocessing and the 'archival' stage)")
    p.add_argument("--averaged-dir", help="Directory for primary-beam-corrected/averaged pickles")
    p.add_argument("--output-dir", default=".", help="Base directory to write figures/tables into")
    p.add_argument("--png-dir", help="Subdirectory (under --output-dir) for PNGs / sanity-check plots "
                    "(default: test_plots)")
    p.add_argument("--pdf-dir", help="Subdirectory (under --output-dir) for PDFs / paper-quality plots "
                    "(default: paper_plots)")
    p.add_argument("--beam-info-dir", help="Directory containing *_beam_table.fits / *_pointings.txt")
    p.add_argument("--askap-pb-corr", action="store_true",
                    help="Force-regenerate ASKAP beam averaging for the 'archival' stage, even "
                         "for SBIDs that already have an averaged pickle (use after the raw "
                         "pickles change; otherwise already-processed SBIDs are skipped).")
    p.add_argument("--mkt-pb-corr", action="store_true",
                    help="Force-regenerate MeerKAT beam correction for the 'archival' stage, "
                         "even for observations that already have a corrected pickle. Doesn't "
                         "affect the automatic single-file generation for the main analysis "
                         "(that always fills in a missing file, but never overwrites an "
                         "existing one).")
    p.add_argument("--spec-index", type=float, help="Spectral index for the archival stage's "
                    "flux-density scaling")
    p.add_argument("--cal-freq-ghz", type=float, help="Reference frequency (GHz) for the "
                    "archival stage's flux-density scaling")
    return p.parse_args()


def build_config(args) -> AnalysisConfig:
    cfg = AnalysisConfig()
    if args.askap_beam09:
        cfg.paths.askap_beam09 = args.askap_beam09
    if args.askap_beam15:
        cfg.paths.askap_beam15 = args.askap_beam15
    if args.meerkat:
        cfg.paths.meerkat = args.meerkat
    if args.dynspec_dir:
        cfg.paths.dynspec_dir = args.dynspec_dir
    if args.averaged_dir:
        cfg.paths.averaged_dir = args.averaged_dir
    cfg.paths.output_dir = args.output_dir
    if args.png_dir:
        cfg.paths.png_dir = args.png_dir
    if args.pdf_dir:
        cfg.paths.pdf_dir = args.pdf_dir
    if args.beam_info_dir:
        from mktj1555 import primary_beams
        primary_beams.BEAM_INFO_DIR = args.beam_info_dir
    cfg.stages.askap_pb_corr = args.askap_pb_corr
    cfg.stages.mkt_pb_corr = args.mkt_pb_corr
    if args.spec_index is not None:
        cfg.archival.spec_index = args.spec_index
    if args.cal_freq_ghz is not None:
        cfg.archival.cal_freq_hz = args.cal_freq_ghz * 1e9
    return cfg


def main():
    args = parse_args()
    cfg = build_config(args)
    stages = set(STAGE_CHOICES) if "all" in args.stage else set(args.stage)

    # Headless-friendly plotting; every stage saves its own figures.
    plt.rcParams.update({"font.size": 6})

    if "archival" in stages:
        from mktj1555.stages import archival
        print("Running archival (non-detection) light-curve pipeline...")
        archival.run_archival_pipeline(cfg)

    needs_main_data = stages - {"archival"}
    if not needs_main_data:
        print("Done.")
        return

    from mktj1555 import data_io
    print("Loading ASKAP + MeerKAT data...")
    # If cfg.paths.meerkat isn't found on disk, this generates it
    # automatically (see preprocessing.ensure_meerkat_file).
    data = data_io.load_all(cfg)

    from mktj1555.stages import overview
    numbering = overview.compute_pulse_numbering(data, cfg)
    if cfg.stages.overview:
        overview.plot_folded_meerkat_lightcurve(data, cfg)
        overview.plot_ephemeris_overview(data, cfg, numbering)

    if "dynspec" in stages:
        from mktj1555.stages import dynspec
        dynspec.plot_askap_dynspec(data, cfg)
        dynspec.plot_meerkat_dynspec(data, cfg)

    if "lightcurves" in stages:
        from mktj1555.stages import lightcurves
        lightcurves.plot_askap_lightcurves(data, cfg)
        lightcurves.save_askap_lightcurve_table(data, cfg)
        lightcurves.plot_meerkat_lightcurves(data, cfg)
        lightcurves.save_meerkat_lightcurve_table(data, cfg)

    if "paper-plots" in stages:
        from mktj1555.stages import paper_plots
        paper_plots.plot_askap_paper_figure(data, cfg)
        paper_plots.plot_meerkat_paper_figure(data, cfg)

    # Folding stages share a `phase` array, so compute it once if needed.
    fold_result = None
    needs_fold_phase = {"fold", "polarization-angle", "fold-main-pulse-spectrum", "phase-scan"} & stages
    if needs_fold_phase:
        from mktj1555.stages import fold
        fold_result = fold.fold_askap(data, cfg) if "fold" in stages else None
        # Even if the figure itself isn't requested, later stages need the
        # folded `phase` array -- recompute it cheaply if `fold` wasn't run.
        if fold_result is None:
            from mktj1555.timing import phase_of
            phase = phase_of(data.askap.times / (24 * 3600), cfg.ephemeris)
            fold_result = {"phase": phase}

        if "polarization-angle" in stages:
            from mktj1555.stages import polarization
            polarization.plot_polarization_angle(data, cfg, fold_result["phase"])

        main_pulse_spectrum = None
        if "fold-main-pulse-spectrum" in stages:
            from mktj1555.stages import fold as fold_mod
            main_pulse_spectrum = fold_mod.fold_main_pulse_spectrum(data, cfg, fold_result["phase"])

        if "phase-scan" in stages:
            from mktj1555.stages import phase_scan
            rms_arr = main_pulse_spectrum["rms_arr"] if main_pulse_spectrum else None
            phase_scan.scan_phase_bins(data, cfg, fold_result["phase"], rms_arr=rms_arr)

    if "interpulse-spectrum" in stages:
        from mktj1555.stages import fold as fold_mod
        from mktj1555.timing import phase_of
        phase = fold_result["phase"] if fold_result else phase_of(data.askap.times / (24 * 3600), cfg.ephemeris)
        fold_mod.fold_interpulse_spectrum(data, cfg, phase)

    if {"joint-iquv", "joint-spectrum", "joint-spectrum-v"} & stages:
        from mktj1555.stages import joint
        if "joint-iquv" in stages:
            joint.plot_joint_lightcurves(data, cfg)
        if "joint-spectrum" in stages:
            joint.fit_joint_spectrum(data, cfg, use_stokes="I")
        if "joint-spectrum-v" in stages:
            joint.fit_joint_spectrum(data, cfg, use_stokes="V")

    if {"spiky-spectrum", "second-ip-v", "second-ip-i", "band-split"} & stages:
        from mktj1555.stages import pulse_spectra
        if "spiky-spectrum" in stages:
            pulse_spectra.fit_main_pulse_spectrum(data, cfg)
        if "second-ip-v" in stages:
            pulse_spectra.fit_second_interpulse_stokes_v(data, cfg)
        if "second-ip-i" in stages:
            pulse_spectra.fit_second_interpulse_stokes_i(data, cfg)
        if "band-split" in stages:
            pulse_spectra.compare_band_split_lightcurve(data, cfg)

    if "acf" in stages:
        from mktj1555.stages import acf
        acf.run_acf_stage(data, cfg, numbering)

    print("Done.")


if __name__ == "__main__":
    main()
