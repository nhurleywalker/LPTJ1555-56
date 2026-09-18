"""Stage: per-pulse autocorrelation functions, to look for characteristic
microstructure timescales in three example pulses (2 and 4 from ASKAP, 13
from MeerKAT -- picked by eye as the most interesting)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..config import AnalysisConfig, STOKES_COLOR, CM_PER_INCH
from ..data_io import AnalysisData
from .overview import PulseNumbering, compute_pulse_numbering


def plot_pulse_previews(data: AnalysisData, cfg: AnalysisConfig, numbering: PulseNumbering):
    """One quick-look plot per pulse (phase 0.4-0.6), to pick out which
    pulses are worth an ACF -- purely a by-eye diagnostic step."""
    a = data.askap
    for n in range(numbering.minpulsenum, numbering.maxpulsenum):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        sel = numbering.pulsenums == n
        ax.plot(numbering.phase[sel], 1000 * a.ilc[sel], color=STOKES_COLOR["I"], alpha=0.8, lw=0.5)
        ax.set_ylabel("brightness (mJy/beam)")
        ax.set_xlabel("Phase")
        ax.set_xlim(0.4, 0.6)
        fig.savefig(cfg.paths.out(f"ASKAP_pulse{n}.png"), bbox_inches="tight")
        plt.close(fig)


def _acf_panel(ax_ts, ax_acf, times, ilc_stack, colors, sample_time_s, nsec, error_bar=None):
    for ts, color in zip(ilc_stack, colors):
        ax_ts.plot(times, 1000 * ts, color=color, alpha=0.8, lw=0.5)
    if error_bar:
        (x, y, yerr) = error_bar
        ax_ts.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.5, capsize=4, capthick=1.5)

    ilc = ilc_stack[0]
    acorr = np.correlate(ilc, ilc, "full")[len(ilc) - 1:]
    t = sample_time_s * np.arange(len(acorr))
    ax_acf.plot(t, acorr / np.nanmax(acorr), alpha=1, lw=0.5, color="darkblue")
    ax_acf.set_xlim([0, nsec / 2])
    peak = np.argmax(acorr[1:])
    ax_acf.axvline(t[1:][peak], color="darkred", lw=0.5, alpha=0.8)
    ax_acf.set_ylim(0.2, 1)
    ax_acf.set_xlabel("Time / s")


def plot_acf_examples(data: AnalysisData, cfg: AnalysisConfig, numbering: PulseNumbering):
    a, m = data.askap, data.meerkat
    ph = cfg.phases
    phase_start_acf = ph.main_pulse_start - 0.5
    phase_end_acf = ph.main_pulse_end + 1 - 0.5

    fig = plt.figure(figsize=(17.9 * CM_PER_INCH, 8 * CM_PER_INCH))

    # ASKAP pulse 2
    ax1 = fig.add_axes([0.1, 0.5, 0.25, 0.6])
    ax2 = fig.add_axes([0.1, 0.1, 0.25, 0.3])
    sel = np.logical_and(np.logical_and(numbering.phase > phase_start_acf, numbering.phase < phase_end_acf),
                          numbering.pulsenums == 2)
    _acf_panel(ax1, ax2, a.times_z[sel], [a.ilc[sel], a.qlc[sel], a.ulc[sel], a.vlc[sel]],
               [STOKES_COLOR[s] for s in "IQUV"], sample_time_s=10, nsec=200,
               error_bar=(3775, 27.5, 2))
    ax1.set_ylabel("Flux density / mJy")
    ax1.set_xlabel("Time / s")
    ax1.set_title("Pulse 2: ASKAP")

    # ASKAP pulse 4
    ax3 = fig.add_axes([0.4, 0.5, 0.25, 0.6])
    ax4 = fig.add_axes([0.4, 0.1, 0.25, 0.3])
    sel = np.logical_and(np.logical_and(numbering.phase > phase_start_acf, numbering.phase < phase_end_acf),
                          numbering.pulsenums == 4)
    _acf_panel(ax3, ax4, a.times_z[sel], [a.ilc[sel], a.qlc[sel], a.ulc[sel], a.vlc[sel]],
               [STOKES_COLOR[s] for s in "IQUV"], sample_time_s=10, nsec=200,
               error_bar=(11270, 17.5, 2))
    ax3.set_xlabel("Time / s")
    ax3.set_title("Pulse 4: ASKAP")

    # MeerKAT pulse 13
    ax5 = fig.add_axes([0.7, 0.5, 0.25, 0.6])
    ax6 = fig.add_axes([0.7, 0.1, 0.25, 0.3])
    sel = np.logical_and(
        np.logical_and(numbering.phase_m > phase_start_acf, numbering.phase_m < phase_end_acf),
        np.logical_and(numbering.pulsenums_m == 13, ~np.isnan(m.ilc)),
    )
    _acf_panel(ax5, ax6, m.times_z[sel], [m.ilc[sel], m.qlc_corr[sel], m.ulc_corr[sel], m.vlc[sel]],
               [STOKES_COLOR[s] for s in "IQUV"], sample_time_s=8, nsec=150,
               error_bar=(16915, 11.75, 0.44))
    ax5.set_xlabel("Time / s")
    ax5.set_title("Pulse 13: MeerKAT")
    ax5.legend(["I", "Q", "U", "V"])

    fig.savefig(cfg.paths.out("ACF.png"), bbox_inches="tight")
    fig.savefig(cfg.paths.out("ACF.pdf"), bbox_inches="tight")
    plt.close(fig)


def run_acf_stage(data: AnalysisData, cfg: AnalysisConfig, base_numbering: PulseNumbering):
    """Re-derives phase/pulse numbering with the main pulse re-centred at
    phase 0.5 (period_offset=1) specifically for the ACF plots, rather than
    reusing the phase-0-centred numbering from the ephemeris-overview
    stage."""
    numbering = compute_pulse_numbering(data, cfg, period_offset=1.0)
    plot_pulse_previews(data, cfg, base_numbering)
    plot_acf_examples(data, cfg, numbering)
