"""Stage: polarisation angle across the main pulse (per time sample, not
phase-binned -- a check for PA structure across the pulse)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..config import AnalysisConfig, STOKES_COLOR
from ..data_io import AnalysisData
from ..timing import phase_window_mask


def plot_polarization_angle(data: AnalysisData, cfg: AnalysisConfig, phase):
    a = data.askap
    ph = cfg.phases
    ind = phase_window_mask(phase, ph.main_pulse_start, ph.main_pulse_end)

    I, Q, U = a.ilc[ind], a.qlc[ind], a.ulc[ind]
    pa = 180.0 + np.degrees(0.5 * np.arctan2(U, Q))
    L = np.sqrt(U ** 2 + Q ** 2)
    rms = np.nanstd(Q)  # less signal here than in I
    err_pa = np.degrees(rms / (2 * L))

    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(211)
    ax1.plot(1000 * I, lw=0.5, alpha=0.8, color=STOKES_COLOR["I"], label="I")
    ax1.plot(1000 * Q, lw=0.5, alpha=0.8, color=STOKES_COLOR["Q"], label="Q")
    ax1.plot(1000 * U, lw=0.5, alpha=0.8, color=STOKES_COLOR["U"], label="U")
    ax1.set_ylabel("Flux density / mJy")
    ax1.axhline(0, lw=0.5, color="black", alpha=0.5)
    ax1.axhspan(-1000 * rms, 1000 * rms, color="blue", alpha=0.1, label=r"$\sigma_\mathrm{off}$")
    ax1.legend(loc=1)

    ax2 = fig.add_subplot(212)
    ok = err_pa < np.degrees(0.15)
    idx = np.arange(len(pa))
    ax2.errorbar(x=idx[ok], y=pa[ok], yerr=err_pa[ok], lw=0, elinewidth=0.5)
    ax2.scatter(x=idx[ok], y=pa[ok], s=1)
    ax2.axhline(142.0, color="red", alpha=0.5)
    ax2.set_ylabel(r"Polarization angle ($^\circ$)")
    ax2.set_xlabel("Time index")
    ax2.set_xlim(ax1.get_xlim())

    fig.savefig(cfg.paths.out("phase_wrt_index.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    out = np.array([a.times[ind], I, Q, U, L, pa, err_pa, rms * np.ones(len(I))])
    np.savetxt(cfg.paths.out("EMU_IQU_light_curves.txt"), out.T,
               header="MJDsec I Q U L PA err_PA err_S")
