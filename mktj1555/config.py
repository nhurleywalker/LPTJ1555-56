"""
Every path, constant, and analysis on/off switch, gathered into one place.
Anything you're likely to want to tweak between runs lives here. File paths
are relative to the current working directory by default (run from a
directory containing `dynspec/` and `averaged_dynspec/` subfolders).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from astropy.coordinates import SkyCoord


@dataclass
class Paths:
    """Input data files and where plots/tables get written."""

    # Directory holding the raw per-observation pickles, and the directory
    # for (and, once populated, source of) the primary-beam-corrected/
    # averaged pickles derived from them. Used by the preprocessing step and
    # by the archival (non-detection) light-curve stage.
    dynspec_dir: str = "dynspec"
    averaged_dir: str = "averaged_dynspec"

    # ASKAP (EMU) beam data, after running through FixMS (ASKAP's polarisation
    # leakage correction). Beam 09 and beam 15 are the two beams nearest the source.
    askap_beam09: str = (
        "dynspec/scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2."
        "beam09_averaged_cal.leakage_fixms.pkl"
    )
    askap_beam15: str = (
        "dynspec/scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2."
        "beam15_averaged_cal.leakage_fixms.pkl"
    )

    # MeerKAT data, already primary-beam corrected and averaged (see
    # preprocessing.py / the 'archival' stage for how these get produced).
    meerkat: str = "averaged_dynspec/CB1652551867.pkl"

    # Where figures and text tables are written. PNGs (quick-look/sanity-
    # check plots) go under `output_dir/png_dir`; PDFs (paper-quality
    # figures) go under `output_dir/pdf_dir`; everything else (text tables,
    # CSVs) goes directly under `output_dir`.
    output_dir: str = "."
    png_dir: str = "test_plots"
    pdf_dir: str = "paper_plots"

    def out(self, filename: str) -> str:
        """Resolve an output filename against the right output directory
        for its type (see `output_dir`/`png_dir`/`pdf_dir` above), creating
        that directory if it doesn't exist yet."""
        if filename.endswith(".png"):
            directory = os.path.join(self.output_dir, self.png_dir)
        elif filename.endswith(".pdf"):
            directory = os.path.join(self.output_dir, self.pdf_dir)
        else:
            directory = self.output_dir
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, filename)


@dataclass
class Ephemeris:
    """Timing solution for the source (MJD reference epoch + period in days)."""

    # Most up-to-date values, from the paper.
    T0: float = 59713.42655646942
    P: float = 62.205893 / (2 * 24 * 60)  # minutes -> days, half-period


@dataclass
class RFIFlags:
    """Manually-identified bad channels/times to blank before analysis.

    Each entry is (time_slice, freq_slice, pol_slice) applied to the MeerKAT
    DS array as ds[time_slice, freq_slice, pol_slice] = nan, plus a couple of
    ASKAP single-channel/edge flags applied after Stokes conversion.
    """

    meerkat_time_freq_pol: tuple = (
        (slice(None), slice(908, 925), slice(None)),
        (slice(96, 142), slice(186, 207), slice(None)),
        (slice(96, 142), slice(952, 974), slice(None)),
        (slice(870, 876), slice(899, 940), slice(None)),
    )
    # Single bad ASKAP channel, and a flagged low-frequency edge, applied to
    # each Stokes plane (I, Q, U, V) after forming them.
    askap_bad_channel: int = 124
    askap_edge_channels: slice = field(default_factory=lambda: slice(0, 16))


@dataclass
class Source:
    """Sky position of the source and the two nearest ASKAP beams."""

    coord: SkyCoord = field(
        default_factory=lambda: SkyCoord("15h55m43.69s -56d31m02.4s", frame="fk5")
    )
    beam_names: tuple = ("09", "15")
    beam_coords: SkyCoord = field(
        default_factory=lambda: SkyCoord(
            ["15h58m56.563 -56d53m16.761", "15h55m36.850 -56d06m49.968"], frame="fk5"
        )
    )


@dataclass
class PhaseWindows:
    """Pulse-phase windows used to isolate the main pulse / interpulse.

    Phase is defined by Ephemeris.T0/P, folded on twice the rotation period.
    The main pulse window is centred at phase 0 (i.e. wraps around); the
    interpulse windows are separate narrow ranges used for spectral fitting.
    """

    main_pulse_start: float = 0.44 + 0.5
    main_pulse_end: float = 0.55 + 0.5 - 1
    interpulse_start: float = 0.02 + 0.5
    interpulse_end: float = 0.055 + 0.5
    # Narrower windows used specifically by the interpulse-spectrum fit
    ip_spectrum_start: float = 0.04
    ip_spectrum_end: float = 0.06


@dataclass
class ArchivalParams:
    """Parameters for the archival (non-detection) light-curve stage."""

    # Spectral index used to scale every flux density to a common frequency,
    # and that reference frequency.
    spec_index: float = -1.5
    cal_freq_hz: float = 1.0e9


@dataclass
class Stages:
    """Which analysis stages to run."""

    dynspec: bool = False
    lightcurves: bool = False
    paper_askap_ds: bool = False
    paper_meerkat_ds: bool = False
    fold: bool = False
    polarization_angle: bool = False
    # There is no rotation-measure-synthesis code anywhere in this codebase;
    # if RM synthesis is needed for the paper it needs to be added
    # separately. This flag controls the folded main-pulse Stokes I/Q/U
    # spectrum and power-law fit.
    fold_main_pulse_spectrum: bool = False
    phase_bin_scan: bool = False
    interpulse_spectrum: bool = False
    debug_poly: bool = False
    joint_iquv: bool = False
    joint_spectrum: bool = False
    joint_spectrum_v: bool = False
    spiky_spectrum: bool = False
    second_ip_spectrum_v: bool = False
    second_ip_spectrum_i: bool = False
    try_band_split: bool = False
    acf: bool = True
    # The folded MeerKAT light curve and the per-pulse ephemeris-accuracy
    # overview plot always run.
    overview: bool = True
    # Non-detection light curves + upper limits for every other observation
    # of the field.
    archival: bool = False

    # Preprocessing: force-regenerate the primary-beam-corrected/averaged
    # pickles in `Paths.averaged_dir` from the raw ones in `Paths.dynspec_dir`,
    # even if they already exist. The 'archival' stage always runs this
    # preprocessing, but skips any file that's already there unless one of
    # these is True -- set True after the raw pickles change. Independent of
    # `preprocessing.ensure_meerkat_file`, which always generates just the
    # one file the main analysis stages need if it's missing, regardless of
    # these flags.
    askap_pb_corr: bool = False
    mkt_pb_corr: bool = False


# Plot styling shared across every stage.
STOKES_COLOR = {"I": "black", "Q": "green", "U": "orange", "V": "blue", "L": "red", "T": "darkgrey"}
STOKES_CMAP = {"I": "plasma", "Q": "bwr_r", "U": "bwr_r", "V": "bwr_r"}
CM_PER_INCH = 1 / 2.54
BOX_PROPERTIES = dict(boxstyle="round", facecolor="red", alpha=0.3, edgecolor="black")

# Fixed indices used to zoom in on specific features of the MeerKAT data
# (identified by eye from the dynamic spectra).
MAIN_PULSE_ZOOM = (460, 510)          # "spiky" main-pulse microstructure
SECOND_INTERPULSE_ZOOM = (763, 863)   # second MeerKAT interpulse
JOINT_TIME_RANGE = (5159271500.0, 5159272500.0)
JOINT_SPEC_RANGE = (5159271875.0, 5159271970.0)


@dataclass
class AnalysisConfig:
    paths: Paths = field(default_factory=Paths)
    ephemeris: Ephemeris = field(default_factory=Ephemeris)
    rfi: RFIFlags = field(default_factory=RFIFlags)
    source: Source = field(default_factory=Source)
    phases: PhaseWindows = field(default_factory=PhaseWindows)
    stages: Stages = field(default_factory=Stages)
    archival: ArchivalParams = field(default_factory=ArchivalParams)
