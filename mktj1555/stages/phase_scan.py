"""Stage: sweep a narrow phase window across the main pulse, saving the
weighted Stokes spectrum for each step. An attempt to see a
polarisation-angle sweep across the pulse -- it doesn't show one, but this
is kept for completeness / in case future data resolves it."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..config import AnalysisConfig, STOKES_COLOR
from ..data_io import AnalysisData
from ..timing import phase_window_mask


def scan_phase_bins(data: AnalysisData, cfg: AnalysisConfig, phase, step=0.01, rms_arr=None):
    """`rms_arr`: a representative per-channel noise estimate to write into
    the output tables, e.g. the one returned by
    `fold.fold_main_pulse_spectrum(...)["rms_arr"]` -- reused across all
    phase bins rather than recomputed for each one."""
    a = data.askap
    ph = cfg.phases

    # The main pulse window wraps through phase 0 (main_pulse_start >
    # main_pulse_end), so scan across it in "unwrapped" phase space and
    # fold each step's edges back into [0, 1) via `% 1`.
    span = ph.main_pulse_end - ph.main_pulse_start
    if span <= 0:
        span += 1
    for phase_start_unwrapped in np.arange(ph.main_pulse_start, ph.main_pulse_start + span, step):
        phase_start = phase_start_unwrapped % 1
        phase_end = (phase_start_unwrapped + step) % 1
        ind = phase_window_mask(phase, phase_start, phase_end)
        Ilc_pulse = a.ilc[ind]

        weights = np.tile(Ilc_pulse, (a.Qt.shape[1], 1)).T
        weights[weights < 0] = 0.0
        weights /= np.nanmax(weights)

        def _collapse(plane):
            spec = np.nansum(plane[ind, :] * weights, axis=0) / np.nansum(weights[:, 0])
            return np.where(spec == 0.0, np.nan, spec)

        I_pulse, Q_pulse, U_pulse, V_pulse = (_collapse(p) for p in (a.It, a.Qt, a.Ut, a.Vt))

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        for name, spec in (("I", I_pulse), ("Q", Q_pulse), ("U", U_pulse), ("V", V_pulse)):
            ax.scatter(a.freqs, 1000 * spec, color=STOKES_COLOR[name], alpha=0.8, label=name)
            ax.axhline(np.nanmean(1000 * spec), color=STOKES_COLOR[name])
        ax.set_ylabel("Weighted brightness (mJy)")
        ax.set_xlabel("Frequency / GHz")
        ax.legend()
        fig.savefig(cfg.paths.out(f"weighted_EMU_Stokes_phasebin{phase_start}.png"), bbox_inches="tight")
        plt.close(fig)

        ok = ~np.isnan(I_pulse)
        rms_this_bin = rms_arr[ok] if rms_arr is not None else np.full(ok.sum(), np.nan)
        out = np.array([a.freqs[ok] * 1e9, I_pulse[ok], Q_pulse[ok], U_pulse[ok],
                        rms_this_bin, rms_this_bin, rms_this_bin])
        np.savetxt(cfg.paths.out(f"EMU_folded_IQU_spectrum_phasebin{phase_start:1.2f}.txt"), out.T)
