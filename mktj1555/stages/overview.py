"""Two sanity-check plots that always run, unconditionally: the folded
MeerKAT light curve, and a per-pulse panel plot checking the ephemeris
stays accurate across the whole campaign."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from ..config import AnalysisConfig, STOKES_COLOR
from ..data_io import AnalysisData
from ..timing import pulsenum, phase_of
from ..plotting import get_weighted_sum


def plot_folded_meerkat_lightcurve(data: AnalysisData, cfg: AnalysisConfig, num_bins=150):
    """Fold the MeerKAT light curve on the rotation ephemeris. No
    parallactic-angle correction is applied to the fold itself -- Q/U here
    need a full polarisation calibration before they're fully meaningful."""
    m = data.meerkat
    eph = cfg.ephemeris
    trange = m.times / (24 * 3600)
    phase = np.mod(trange, 2 * eph.P) / (2 * eph.P)
    idx = np.argsort(phase)

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(phase[idx], bin_edges) - 1
    counts = np.bincount(bin_indices, minlength=num_bins)

    def _binned(lc):
        sums = get_weighted_sum(bin_indices, 1000 * lc[idx])
        return sums / np.where(counts == 0, 1, counts)

    I_avg = _binned(m.ilc)
    Q_avg = _binned(m.qlc_corr)
    U_avg = _binned(m.ulc_corr)
    V_avg = _binned(m.vlc)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    for name, y in (("I", I_avg), ("Q", Q_avg), ("U", U_avg), ("V", V_avg)):
        ax.plot(bin_centers, y, color=STOKES_COLOR[name], alpha=0.8, lw=0.5, label=f"Stokes {name}")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Mean brightness (mJy)")
    ax.legend(loc=1)
    fig.savefig(cfg.paths.out("Folded_MeerKAT_light_curve.png"), bbox_inches="tight")
    plt.close(fig)


@dataclass
class PulseNumbering:
    phase: np.ndarray
    pulsenums: np.ndarray
    minpulsenum: int
    maxpulsenum: int
    phase_m: np.ndarray
    pulsenums_m: np.ndarray
    trange: np.ndarray
    trange_m: np.ndarray


def compute_pulse_numbering(data: AnalysisData, cfg: AnalysisConfig, period_offset=0.0) -> PulseNumbering:
    """Phase and integer pulse-number for every ASKAP/MeerKAT sample,
    relative to the ephemeris. `period_offset` (in units of P) shifts which
    part of the rotation is centred at phase 0 -- the ACF stage uses +1 to
    centre the main pulse at phase 0.5 instead of wrapping across phase 0."""
    a, m, eph = data.askap, data.meerkat, cfg.ephemeris
    trange = a.times / (24 * 3600)
    phase = phase_of(trange, eph, period_offset=period_offset)
    pulsenums = (pulsenum(trange, eph) + period_offset / 2).astype(int)

    trange_m = m.times / (24 * 3600)
    phase_m = phase_of(trange_m, eph, period_offset=period_offset)
    pulsenums_m = (pulsenum(trange_m, eph) + period_offset / 2).astype(int)

    return PulseNumbering(
        phase=phase, pulsenums=pulsenums, minpulsenum=pulsenums[0], maxpulsenum=pulsenums[-1],
        phase_m=phase_m, pulsenums_m=pulsenums_m, trange=trange, trange_m=trange_m,
    )


def plot_ephemeris_overview(data: AnalysisData, cfg: AnalysisConfig, numbering: PulseNumbering):
    """One panel per pulse, ASKAP then MeerKAT, checking the ephemeris lines
    up with the pulse in every epoch."""
    a, m = data.askap, data.meerkat
    n = numbering
    num_extra_mkt_panels = len(np.unique(n.pulsenums_m))
    num_panels = n.maxpulsenum - n.minpulsenum + num_extra_mkt_panels

    fig = plt.figure(figsize=(5, 20))
    ind = 1
    for pn in range(n.minpulsenum, n.maxpulsenum):
        ax = fig.add_subplot(num_panels, 1, ind)
        ax.plot(n.phase[n.pulsenums == pn], 1000 * a.ilc[n.pulsenums == pn], color=STOKES_COLOR["I"],
                alpha=0.8, label=f"{pn}")
        ax.set_ylim(-3, 20)
        ax.set_xlim(-0.05, 1.05)
        ax.axvline(0.05, alpha=0.4, color="orange")
        ax.axvline(0.52, alpha=0.8, color="orange")
        ax.legend()
        ax.tick_params(axis="x", labelbottom=False)
        ind += 1
    for pn in np.unique(n.pulsenums_m):
        ax = fig.add_subplot(num_panels, 1, ind)
        ax.plot(n.phase_m[n.pulsenums_m == pn], 1000 * m.ilc[n.pulsenums_m == pn], color="purple",
                alpha=0.8, label=f"{pn}")
        ax.set_ylim(-3, 20)
        ax.set_xlim(-0.05, 1.05)
        ax.axvline(0.05, alpha=0.4, color="orange")
        ax.axvline(0.52, alpha=0.8, color="orange")
        ax.legend()
        if ind != num_panels:
            ax.tick_params(axis="x", labelbottom=False)
        ind += 1
        ax.set_xlabel("Phase")
    fig.savefig(cfg.paths.out("Ephemeris_lightcurve.png"), bbox_inches="tight")
    plt.close(fig)
