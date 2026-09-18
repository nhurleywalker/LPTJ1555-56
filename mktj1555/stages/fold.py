"""Stage: fold the ASKAP light curve on the rotation ephemeris.

Produces the phase-binned Stokes light curves and fractional-polarisation
figure, plus (via `weighted_pulse_spectrum`) the frequency spectrum of
whichever phase window you point it at -- used for both the main pulse and
the interpulse.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from ..config import AnalysisConfig, STOKES_COLOR
from ..data_io import AnalysisData
from ..timing import phase_of, phase_window_mask
from ..plotting import get_weighted_sum, pwrap
from ..fitting import pl, fit_powerlaw, bin_by_split


def fold_askap(data: AnalysisData, cfg: AnalysisConfig, num_bins=200):
    """Fold the ASKAP Stokes I/Q/U/V light curves on 2x the rotation period
    and bin by phase. Returns the folded arrays for reuse by other stages
    (e.g. polarisation-angle / RM fitting), and writes the folded-light-curve
    figure."""
    a = data.askap
    eph, ph = cfg.ephemeris, cfg.phases

    trange = a.times / (24 * 3600)
    phase = phase_of(trange, eph)
    idx = np.argsort(phase)

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(phase[idx], bin_edges) - 1
    counts = np.bincount(bin_indices, minlength=num_bins)

    def _binned(lc):
        sums = get_weighted_sum(bin_indices, 1000 * lc[idx])
        return sums / np.where(counts == 0, 1, counts)

    I_avg, Q_avg, U_avg, V_avg = (_binned(lc) for lc in (a.ilc, a.qlc, a.ulc, a.vlc))
    L_avg = np.sqrt(Q_avg ** 2 + U_avg ** 2)
    T_avg = np.sqrt(Q_avg ** 2 + U_avg ** 2 + V_avg ** 2)

    rms_start, rms_end = 0.1, 0.4
    in_rms_window = np.logical_and(bin_centers > rms_start, bin_centers < rms_end)
    rms = np.nanstd(U_avg[in_rms_window])

    L_frac = 100 * L_avg / I_avg
    V_frac = 100 * np.abs(V_avg) / I_avg
    err_V_frac = 100 * np.sqrt((rms / I_avg) ** 2 + (V_avg * rms / I_avg ** 2) ** 2)
    T_frac = 100 * np.sqrt((Q_avg ** 2 + U_avg ** 2 + V_avg ** 2) / I_avg ** 2)
    err_L = np.sqrt((rms * Q_avg / L_avg) ** 2 + (rms * U_avg / L_avg) ** 2)
    err_L_frac = np.abs(L_frac) * np.sqrt((err_L / L_avg) ** 2 + (rms / I_avg) ** 2)
    err_T = np.sqrt((rms * Q_avg / T_avg) ** 2 + (rms * U_avg / T_avg) ** 2 + (rms * T_avg) ** 2)
    err_T_frac = T_frac * np.sqrt((err_T / T_avg) ** 2 + (rms / I_avg) ** 2)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.scatter(T_frac, err_T_frac, color=STOKES_COLOR["T"])
    ax.scatter(L_frac, err_L_frac, color=STOKES_COLOR["L"])
    ax.scatter(V_frac, err_V_frac, color=STOKES_COLOR["V"])
    ax.set_xlabel("value")
    ax.set_ylabel("error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.savefig(cfg.paths.out("error_test.png"), bbox_inches="tight")
    plt.close(fig)

    _plot_folded_lightcurve(cfg, bin_centers, I_avg, Q_avg, U_avg, V_avg, T_frac, L_frac, V_frac,
                             err_L_frac, err_V_frac, err_T_frac, rms, num_bins, a.times_z[-1])

    return dict(phase=phase, bin_centers=bin_centers, I_avg=I_avg, Q_avg=Q_avg, U_avg=U_avg,
                V_avg=V_avg, L_frac=L_frac, V_frac=V_frac, T_frac=T_frac, rms=rms)


def _plot_folded_lightcurve(cfg, bin_centers, I_avg, Q_avg, U_avg, V_avg, T_frac, L_frac, V_frac,
                             err_L_frac, err_V_frac, err_T_frac, rms, num_bins, obs_length_s):
    ph = cfg.phases
    phase_start, phase_end = ph.main_pulse_start, ph.main_pulse_end
    phase_start_ip, phase_end_ip = ph.interpulse_start, ph.interpulse_end
    cm = 1 / 2.54

    fig = plt.figure(figsize=(16 * cm, 8 * cm))
    ax1 = fig.add_axes([0.1, 0.5, 0.5, 0.35])
    for ax in (ax1,):
        ax.axvspan(0, phase_end, alpha=0.1, color="grey")
        ax.axvspan(phase_start, phase_end + 1, alpha=0.1, color="grey")
        ax.axvspan(phase_start + 1, 2, alpha=0.1, color="grey")
        ax.axvspan(phase_start_ip, phase_end_ip, alpha=0.1, color="yellow")
        ax.axvspan(phase_start_ip + 1, phase_end_ip + 1, alpha=0.1, color="yellow")

    x = np.hstack([bin_centers, bin_centers + 1])
    for name, y in (("I", I_avg), ("Q", Q_avg), ("U", U_avg), ("V", V_avg)):
        ax1.plot(x, np.hstack([y, y]), color=STOKES_COLOR[name], alpha=0.8, lw=0.5, label=name)
    ax1.set_xticklabels([])
    ax1.set_xlabel("Phase")
    ax1.set_ylabel("Mean brightness (mJy)")
    ax1.set_xlim(0, 2)
    ax1.legend(loc=1)

    ax2 = fig.add_axes([0.1, 0.1, 0.5, 0.35])
    ax2.axvspan(0, phase_end, alpha=0.1, color="grey")
    ax2.axvspan(phase_start, phase_end + 1, alpha=0.1, color="grey")
    ax2.axvspan(phase_start + 1, 2, alpha=0.1, color="grey")
    ax2.axvspan(phase_start_ip, phase_end_ip, alpha=0.1, color="yellow")
    ax2.axvspan(phase_start_ip + 1, phase_end_ip + 1, alpha=0.1, color="yellow")
    I_cut = 1  # mJy
    ind_mp = np.logical_or(bin_centers > phase_start, bin_centers < phase_end)
    ind1 = np.logical_and(np.abs(I_avg) > I_cut, ind_mp)
    ind_ip = np.logical_and(bin_centers > phase_start_ip, bin_centers < phase_end_ip)
    ind_ip_wide = np.logical_and(bin_centers > phase_start_ip - 0.05, bin_centers < phase_end_ip + 0.05)
    ind2 = np.logical_and(np.abs(I_avg) > I_cut, ind_ip)

    def _plot_frac(ax, sel, offset=0):
        ax.scatter(bin_centers[sel] + offset, T_frac[sel], color=STOKES_COLOR["T"], alpha=0.8, lw=0, s=5)
        ax.scatter(bin_centers[sel] + offset, L_frac[sel], color=STOKES_COLOR["L"], alpha=0.8, lw=0, s=5, marker="s")
        ax.errorbar(bin_centers[sel] + offset, L_frac[sel], yerr=err_L_frac[sel], color=STOKES_COLOR["L"],
                    alpha=0.8, lw=0, elinewidth=0.5, capsize=1, capthick=0.5)
        ax.scatter(bin_centers[sel] + offset, V_frac[sel], color=STOKES_COLOR["V"], alpha=0.8, lw=0, s=8, marker="*")
        ax.errorbar(bin_centers[sel] + offset, V_frac[sel], yerr=err_V_frac[sel], color=STOKES_COLOR["V"],
                    alpha=0.8, lw=0, elinewidth=0.5, capsize=1, capthick=0.5)

    ax2.scatter(bin_centers[ind1], T_frac[ind1], color=STOKES_COLOR["T"], alpha=0.8, lw=0, s=5, label="Total")
    ax2.scatter(bin_centers[ind1], L_frac[ind1], color=STOKES_COLOR["L"], alpha=0.8, lw=0, s=5, marker="s", label="Linear")
    ax2.scatter(bin_centers[ind1], V_frac[ind1], color=STOKES_COLOR["V"], alpha=0.8, lw=0, s=8, marker="*", label="Circular")
    _plot_frac(ax2, ind1)
    _plot_frac(ax2, ind1, offset=1)
    _plot_frac(ax2, ind2)
    _plot_frac(ax2, ind2, offset=1)
    ax2.set_xlabel("Phase")
    ax2.set_ylim(0, 120)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylabel("$|$Fractional polarisation$|$ (%)")
    ax2.legend(loc=1)

    ax3 = fig.add_axes([0.60, 0.5, 0.15, 0.35])
    ax3.set_ylim(ax1.get_ylim())
    for name, y in (("I", I_avg), ("Q", Q_avg), ("U", U_avg), ("V", V_avg)):
        ax3.plot(bin_centers[ind_ip_wide], y[ind_ip_wide], color=STOKES_COLOR[name], alpha=0.8, lw=0.5)
    ax3.axvspan(phase_start_ip, phase_end_ip, alpha=0.1, color="yellow")
    ax3.tick_params(axis="y", length=0)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])

    ax4 = fig.add_axes([0.75, 0.5, 0.25, 0.35])
    ax4.set_ylim(ax1.get_ylim())
    x_w = pwrap(bin_centers, num_bins, add=True)
    ind_mp_wide = np.logical_and(x_w > phase_start - 0.03, x_w < phase_end + 1.03)
    ind_mp_rb = np.logical_and(np.logical_and(x_w > phase_start, x_w < phase_end + 1), np.abs(pwrap(I_avg, num_bins) > I_cut))
    for name, y in (("I", I_avg), ("Q", Q_avg), ("U", U_avg), ("V", V_avg)):
        yw = pwrap(y, num_bins)
        ax4.plot(x_w[ind_mp_wide], yw[ind_mp_wide], color=STOKES_COLOR[name], alpha=0.8, lw=0.5)
    ax4.axvspan(phase_start, phase_end + 1, alpha=0.1, color="grey")
    ax4.tick_params(axis="y", direction="inout")
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    ax4.errorbar(0.93, 8, yerr=rms, fmt="none", ecolor="black", elinewidth=1, capsize=1, capthick=1)

    ax5 = fig.add_axes([0.60, 0.1, 0.15, 0.35])
    ax5.set_ylim(ax2.get_ylim())
    ax5.set_xlim(ax3.get_xlim())
    ax5.scatter(bin_centers[ind2], T_frac[ind2], color=STOKES_COLOR["T"], alpha=0.8, lw=0, s=5, marker="o")
    ax5.errorbar(bin_centers[ind2], T_frac[ind2], yerr=err_T_frac[ind2], color=STOKES_COLOR["T"], alpha=0.8, lw=0, elinewidth=0.5, capsize=1, capthick=0.5)
    ax5.scatter(bin_centers[ind2], L_frac[ind2], color=STOKES_COLOR["L"], alpha=0.8, lw=0, s=5, marker="s")
    ax5.errorbar(bin_centers[ind2], L_frac[ind2], yerr=err_L_frac[ind2], color=STOKES_COLOR["L"], alpha=0.8, lw=0, elinewidth=0.5, capsize=1, capthick=0.5)
    ax5.scatter(bin_centers[ind2], V_frac[ind2], color=STOKES_COLOR["V"], alpha=0.8, lw=0, s=5, marker="*")
    ax5.errorbar(bin_centers[ind2], V_frac[ind2], yerr=err_V_frac[ind2], color=STOKES_COLOR["V"], alpha=0.8, lw=0, elinewidth=0.5, capsize=1, capthick=0.5)
    ax5.axvspan(phase_start_ip, phase_end_ip, alpha=0.1, color="yellow")
    ax5.tick_params(axis="y", length=0)
    ax5.set_yticklabels([])

    yL, yV, yT = pwrap(L_frac, num_bins), pwrap(V_frac, num_bins), pwrap(T_frac, num_bins)
    eL, eV, eT = pwrap(err_L_frac, num_bins), pwrap(err_V_frac, num_bins), pwrap(err_T_frac, num_bins)
    ax6 = fig.add_axes([0.75, 0.1, 0.25, 0.35])
    ax6.set_ylim(ax2.get_ylim())
    ax6.set_xlim(ax4.get_xlim())
    ax6.scatter(x_w[ind_mp_rb], yT[ind_mp_rb], color=STOKES_COLOR["T"], alpha=0.8, lw=0, s=5, marker="o")
    ax6.errorbar(x_w[ind_mp_rb], yT[ind_mp_rb], yerr=eT[ind_mp_rb], color=STOKES_COLOR["T"], alpha=0.8, lw=0, elinewidth=0.5, capsize=1, capthick=0.5)
    ax6.scatter(x_w[ind_mp_rb], yL[ind_mp_rb], color=STOKES_COLOR["L"], alpha=0.8, lw=0, s=5, marker="s")
    ax6.errorbar(x_w[ind_mp_rb], yL[ind_mp_rb], yerr=eL[ind_mp_rb], color=STOKES_COLOR["L"], alpha=0.8, lw=0, elinewidth=0.5, capsize=1, capthick=0.5)
    ax6.scatter(x_w[ind_mp_rb], yV[ind_mp_rb], color=STOKES_COLOR["V"], alpha=0.8, lw=0, s=8, marker="*")
    ax6.errorbar(x_w[ind_mp_rb], yV[ind_mp_rb], yerr=eV[ind_mp_rb], color=STOKES_COLOR["V"], alpha=0.8, lw=0, elinewidth=0.5, capsize=1, capthick=0.5)
    ax6.axvspan(phase_start, phase_end + 1, alpha=0.1, color="grey")
    ax6.tick_params(axis="y", direction="inout")
    ax6.set_yticklabels([])

    P = cfg.ephemeris.P
    n_p = obs_length_s / (2 * P * 24 * 3600)
    len_ip = (phase_end_ip - phase_start_ip) * 2 * P * 24 * 3600
    len_mp = (1 + phase_end - phase_start) * 2 * P * 24 * 3600
    print(f"Successfully stacked {n_p:2.2f} periods, boosting S/N by {np.sqrt(n_p):2.2f}")
    print(f"RMS of light curve is {rms * 1000:2.0f} uJy/beam")
    print(f"Main pulse is about {len_mp:2.0f}s wide.")
    print(f"Inter-pulse is about {len_ip:2.0f}s wide.")
    print(f"Maximum linear polarisation of main (broad) pulse is {np.nanmax(L_frac[ind1]):3.0f}%")
    print(f"Maximum linear polarisation of inter (narrow) pulse is {np.nanmax(L_frac[ind2]):3.0f}%")
    print(f"Maximum absolute circular polarisation of main (broad) pulse is {np.nanmax(np.abs(V_frac[ind1])):3.0f}%")
    print(f"Maximum absolute circular polarisation of inter (narrow) pulse is {np.nanmax(np.abs(V_frac[ind2])):3.0f}%")
    print(f"Mean linear polarisation of main (broad) pulse is {np.nanmean(L_frac[ind1]):3.0f}%")
    print(f"Mean linear polarisation of inter (narrow) pulse is {np.nanmean(L_frac[ind2]):3.0f}%")
    print(f"Mean absolute circular polarisation of main (broad) pulse is {np.nanmean(np.abs(V_frac[ind1])):3.0f}%")
    print(f"Mean absolute circular polarisation of inter (narrow) pulse is {np.nanmean(np.abs(V_frac[ind2])):3.0f}%")
    fig.savefig(cfg.paths.out("Folded_EMU_light_curve.pdf"), bbox_inches="tight")
    plt.close(fig)


def weighted_pulse_spectrum(data: AnalysisData, cfg: AnalysisConfig, phase, phase_start, phase_end,
                             out_prefix, weighted_qu_name, iqu_txt_name, final_plot_name,
                             fit_label, nu_range=(1.3, 1.45), n_splits=5, sanity_check_path=None):
    """Weight the ASKAP dynamic spectrum by the Stokes-I light curve within
    a phase window, collapse to a spectrum, and (if there's enough S/N) fit
    a power law. Used for both the main-pulse fold spectrum and the
    interpulse spectrum -- they're the same recipe over different windows.
    """
    a = data.askap
    ind = phase_window_mask(phase, phase_start, phase_end)
    Ilc_pulse = a.ilc[ind]

    if sanity_check_path:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.scatter(phase[ind], Ilc_pulse)
        ax.set_xlabel("Phase")
        ax.set_ylabel(f"Stokes I light curve just of {sanity_check_path[1]}")
        fig.savefig(cfg.paths.out(sanity_check_path[0]), bbox_inches="tight")
        plt.close(fig)

    weights = np.tile(Ilc_pulse, (a.Qt.shape[1], 1)).T
    weights[weights < 0] = 0.0
    weights /= np.nanmax(weights)

    def _collapse(plane):
        spec = np.nansum(plane[ind, :] * weights, axis=0) / np.nansum(weights, axis=0)
        return np.where(spec == 0.0, np.nan, spec)

    I_pulse, Q_pulse, U_pulse, V_pulse = (_collapse(p) for p in (a.It, a.Qt, a.Ut, a.Vt))

    rms = np.nanstd(a.It[np.logical_and(phase > 0.2, phase < 0.3), :])
    rms /= np.sqrt(np.nansum(weights, axis=0))
    rms_arr = rms * np.ones(len(I_pulse))

    fig = plt.figure(figsize=(5 / 2.54, 4 / 2.54))
    ax = fig.add_subplot(111)
    ax.scatter(a.freqs, 1000 * Q_pulse, color=STOKES_COLOR["Q"], alpha=0.8, marker=".", s=4, lw=0.5, zorder=10, label="Stokes Q")
    ax.errorbar(a.freqs, 1000 * Q_pulse, yerr=1000 * rms_arr, color=STOKES_COLOR["Q"], alpha=0.8, elinewidth=0.5, lw=0, zorder=5)
    ax.axhline(np.nanmean(1000 * Q_pulse), color=STOKES_COLOR["Q"], lw=0.5)
    ax.scatter(a.freqs, 1000 * U_pulse, color=STOKES_COLOR["U"], alpha=0.8, marker=".", s=4, lw=0.5, zorder=10, label="Stokes U")
    ax.errorbar(a.freqs, 1000 * U_pulse, yerr=1000 * rms_arr, color=STOKES_COLOR["U"], alpha=0.8, elinewidth=0.5, lw=0, zorder=5)
    ax.axhline(np.nanmean(1000 * U_pulse), color=STOKES_COLOR["U"], lw=0.5)
    ax.set_ylabel("Weighted brightness (mJy)")
    ax.set_xlabel("Frequency / GHz")
    ax.set_ylim(-10, 10)
    ax.legend()
    fig.savefig(cfg.paths.out(f"{weighted_qu_name}.pdf"), bbox_inches="tight")
    fig.savefig(cfg.paths.out(f"{weighted_qu_name}.png"), bbox_inches="tight")
    plt.close(fig)

    ok = ~np.isnan(I_pulse)
    out = np.array([a.freqs[ok] * 1e9, I_pulse[ok], Q_pulse[ok], U_pulse[ok],
                    rms_arr[ok], rms_arr[ok], rms_arr[ok]])
    np.savetxt(cfg.paths.out(iqu_txt_name), out.T)

    freqs_b, I_b, err_b = bin_by_split(a.freqs[ok], I_pulse[ok], rms_arr[ok], n_splits)
    fit = fit_powerlaw(freqs_b, 1000 * I_b, 1000 * err_b, nu_range=nu_range)
    print(fit.report(f"just EMU {fit_label} (low bandwidth!)"))

    fig = plt.figure(figsize=(8 / 2.54, 8 / 2.54))
    ax = fig.add_subplot(111)
    ax.scatter(a.freqs[ok], 1000 * I_pulse[ok], color=STOKES_COLOR["I"], alpha=0.2, marker=".", s=4, lw=0.5, zorder=10, label="Stokes I")
    ax.scatter(freqs_b, 1000 * I_b, color=STOKES_COLOR["I"], alpha=0.8, marker="s", s=6, lw=0.5, zorder=10, label="Stokes I")
    ax.errorbar(freqs_b, 1000 * I_b, yerr=1000 * err_b, color=STOKES_COLOR["I"], alpha=0.8, elinewidth=0.5, lw=0, zorder=5)
    ax.plot(fit.nu, fit.q50, lw=0.5, color="red")
    ax.fill_between(fit.nu, fit.q16, fit.q84, alpha=0.3, color="red")
    ax.set_ylabel("Weighted brightness (mJy)")
    ax.set_xlabel("Frequency / GHz")
    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import StrMethodFormatter, MultipleLocator
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax.yaxis.set_minor_formatter(StrMethodFormatter("{x:.0f}"))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    ax.xaxis.set_minor_formatter(StrMethodFormatter("{x:.2f}"))
    ax.set_ylim(8, 16)
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    fig.savefig(cfg.paths.out(final_plot_name), bbox_inches="tight", dpi=300)
    plt.close(fig)

    return dict(I_pulse=I_pulse, Q_pulse=Q_pulse, U_pulse=U_pulse, V_pulse=V_pulse, rms_arr=rms_arr)


def fold_main_pulse_spectrum(data: AnalysisData, cfg: AnalysisConfig, phase):
    ph = cfg.phases
    return weighted_pulse_spectrum(
        data, cfg, phase, ph.main_pulse_start, ph.main_pulse_end, out_prefix="EMU_folded",
        weighted_qu_name="EMU_weighted_Stokes_spectra",
        iqu_txt_name="EMU_folded_IQU_spectrum.txt",
        final_plot_name="EMU_folded_Stokes_I_spectrum_mainpulse.png",
        fit_label="Pulse",
        sanity_check_path=("test_pulse_capture.png", "main pulse"),
    )


def fold_interpulse_spectrum(data: AnalysisData, cfg: AnalysisConfig, phase):
    ph = cfg.phases
    return weighted_pulse_spectrum(
        data, cfg, phase, ph.ip_spectrum_start, ph.ip_spectrum_end, out_prefix="EMU_IP_folded",
        weighted_qu_name="EMU_IP_weighted_Stokes_spectra",
        iqu_txt_name="EMU_IP_folded_IQU_spectrum.txt",
        final_plot_name="EMU_folded_Stokes_I_spectrum_interpulse.png",
        fit_label="IP",
        sanity_check_path=("test_ipulse_capture.png", "inter pulse"),
    )
