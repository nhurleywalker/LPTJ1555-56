"""Stage: compare ASKAP and MeerKAT directly, around the joint pulse
detection both telescopes captured at the same time."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..config import AnalysisConfig, STOKES_COLOR, CM_PER_INCH, JOINT_TIME_RANGE, JOINT_SPEC_RANGE
from ..data_io import AnalysisData
from ..timing import ephem
from ..plotting import make_dynspec
from ..fitting import (
    normalized_weights, collapse_with_weights, bin_by_edges, fit_powerlaw,
    bin_shading_spans, SpectrumSeries, plot_powerlaw_spectrum,
)

STANDARD_BIN_EDGES = [110, 240, 470, 700, 860]


def plot_joint_lightcurves(data: AnalysisData, cfg: AnalysisConfig):
    a, m = data.askap, data.meerkat
    eph = cfg.ephemeris
    tstart, tend = JOINT_TIME_RANGE
    spec_start, spec_end = JOINT_SPEC_RANGE

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_ylabel("brightness (mJy/beam)")
    ax.set_xlabel("time / s")
    ax.plot(a.times, 1000 * a.ilc, lw=2, color="darkgrey", alpha=0.8, label="ASKAP Stokes I")
    ax.plot(m.times, 1000 * m.ilc, lw=0.5, color="black", alpha=0.8, label="MeerKAT Stokes I")
    for i in range(-3, 30):
        ax.axvline(ephem(i, eph) * 24 * 3600, alpha=0.4, color="orange")
    ax.set_xlim(tstart, tend)
    ax.set_ylim(-10, 30)
    ax.axvspan(spec_start, spec_end, color="blue", alpha=0.1)
    ax.legend(loc=1)
    fig.savefig(cfg.paths.out("Joint_StokesI_lightcurve.png"), bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_ylabel("brightness (mJy/beam)")
    ax.set_xlabel("time / s")
    for name, lw_a, lw_m, col_a, col_m in [
        ("Q", 2, 0.5, "red", "darkred"), ("U", 2, 0.5, "blue", "darkblue"), ("V", 2, 0.5, "green", "darkgreen"),
    ]:
        lc_a = getattr(a, f"{name.lower()}lc")
        lc_m = getattr(m, f"{name.lower()}lc")
        ax.plot(a.times, 1000 * lc_a, lw=lw_a, color=col_a, alpha=0.5, label=f"ASKAP Stokes {name}")
        ax.plot(m.times, 1000 * lc_m, lw=lw_m, color=col_m, alpha=0.8, label=f"MeerKAT Stokes {name}")
    for i in range(-3, 30):
        ax.axvline(ephem(i, eph) * 24 * 3600, alpha=0.4, color="orange")
    ax.set_xlim(tstart, tend)
    ax.legend(loc=1)
    fig.savefig(cfg.paths.out("Joint_StokesQUV_lightcurve.png"), bbox_inches="tight")
    plt.close(fig)


def _pulse_window_indices(times, spec_start, spec_end):
    return np.argwhere(np.logical_and(times < spec_end, times > spec_start)).ravel()


def fit_joint_spectrum(data: AnalysisData, cfg: AnalysisConfig, use_stokes="I", weight_floor=None):
    """Weight ASKAP + MeerKAT around the joint pulse detection by its own
    light curve, collapse each to a single-epoch spectrum, and fit one
    power law across both instruments' bands.

    use_stokes="I" does this with weights thresholded at zero and produces
    background-subtraction diagnostics only (see below); use_stokes="V"
    skips background subtraction and thresholds -Stokes V weighting at
    0.006, since Stokes V is cleaner than Stokes I for this particular
    pulse.
    """
    a, m = data.askap, data.meerkat
    spec_start, spec_end = JOINT_SPEC_RANGE
    ind_a = _pulse_window_indices(a.times, spec_start, spec_end)
    ind_m = _pulse_window_indices(m.times, spec_start, spec_end)
    weight_floor = weight_floor if weight_floor is not None else (0.0 if use_stokes == "I" else 0.006)

    if use_stokes == "I":
        plane_a, plane_m, lc_a, lc_m = a.It, m.It, a.ilc, m.ilc
        sign = 1.0
    else:
        plane_a, plane_m, lc_a, lc_m = -a.Vt, -m.Vt, -a.vlc, -m.vlc
        sign = -1.0

    make_dynspec(plane_m[ind_m[0] - 30:ind_m[-1] + 30].T, -0.005, 0.03, "plasma" if use_stokes == "I" else "bwr_r",
                 [m.times_z[ind_m[0] - 30], m.times_z[ind_m[-1] + 30], m.freqs[0], m.freqs[-1]],
                 cfg.paths.out(f"MeerKAT_Stokes{use_stokes}_joint_pulse_zoom.png"), imwidth=5)
    make_dynspec(plane_a[ind_a[0] - 24:ind_a[-1] + 24].T, -0.005, 0.03, "plasma" if use_stokes == "I" else "bwr_r",
                 [a.times_z[ind_a[0] - 24], a.times_z[ind_a[-1] + 24], a.freqs[0], a.freqs[-1]],
                 cfg.paths.out(f"EMU_Stokes{use_stokes}_joint_pulse_zoom.png"), imwidth=5)

    if use_stokes == "I":
        # Background diagnostics only (kept purely for the debug PNGs --
        # the actual fit below uses the raw, non-background-subtracted
        # data, since background subtraction turns out to be unnecessary
        # here).
        bkg = np.nanmean([np.nanmean(m.It[ind_m[0] - 10:ind_m[0] - 1], axis=0),
                          np.nanmean(m.It[ind_m[-1] + 1:ind_m[-1] + 10], axis=0)], axis=0)
        bkg_for_plot = np.tile(bkg, (len(ind_m) + 59, 1))
        make_dynspec(bkg_for_plot.T, -0.005, 0.03, "plasma",
                     [m.times_z[ind_m[0] - 30], m.times_z[ind_m[-1] + 30], m.freqs[0], m.freqs[-1]],
                     cfg.paths.out("MeerKAT_StokesI_joint_pulse_zoom_bkg.png"), imwidth=5)
        make_dynspec((m.It[ind_m[0] - 30:ind_m[-1] + 30] - bkg_for_plot).T, -0.005, 0.03, "plasma",
                     [m.times_z[ind_m[0] - 30], m.times_z[ind_m[-1] + 30], m.freqs[0], m.freqs[-1]],
                     cfg.paths.out("MeerKAT_StokesI_joint_pulse_zoom_bkg_subtracted.png"), imwidth=5)
    else:
        for label, weights_raw, out_name in [
            ("EMU", lc_a[ind_a], "EMU_test_ip_weights.png"),
            ("MeerKAT", lc_m[ind_m], "MeerKAT_test_ip_weights.png"),
        ]:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.plot(weights_raw)
            ax.axhline(weight_floor)
            ax.set_xlabel("index")
            ax.set_ylabel(f"Stokes V light curve just of joint inter pulse ({label})")
            fig.savefig(cfg.paths.out(out_name), bbox_inches="tight")
            plt.close(fig)

    weights_a = normalized_weights(lc_a[ind_a], weight_floor)
    weights_m = normalized_weights(lc_m[ind_m], weight_floor)
    flux_a = collapse_with_weights(plane_a[ind_a], weights_a)
    flux_m = collapse_with_weights(plane_m[ind_m], weights_m)

    freqs_m_b, flux_m_b, err_m_b = bin_by_edges(m.freqs, flux_m, STANDARD_BIN_EDGES)
    ok_a = ~np.isnan(flux_a)
    flux_a_b = np.array([np.nanmean(flux_a)])
    freqs_a_b = np.array([np.nanmean(a.freqs[ok_a])])
    err_a_b = np.array([np.nanstd(flux_a) / np.sqrt(ok_a.sum())])

    freqs_b = np.hstack([freqs_m_b, freqs_a_b])
    flux_b = np.hstack([flux_m_b, flux_a_b])
    err_b = np.hstack([err_m_b, err_a_b])
    fit = fit_powerlaw(freqs_b, 1000 * flux_b, 1000 * err_b)
    print(fit.report(f"joint fit to IP{' in Stokes V' if use_stokes == 'V' else ''}"))

    series = [
        SpectrumSeries(m.freqs, 1000 * flux_m, freqs_m_b, 1000 * flux_m_b, 1000 * err_m_b,
                        color="purple", label="MeerKAT", marker="."),
        SpectrumSeries(a.freqs, 1000 * flux_a, freqs_a_b, 1000 * flux_a_b, 1000 * err_a_b,
                        color=STOKES_COLOR["I"], label="ASKAP", marker="s", raw_alpha=0.2),
    ]
    suffix = "" if use_stokes == "I" else "_StokesV"
    plot_powerlaw_spectrum(
        series, fit,
        [cfg.paths.out(f"Joint_spectrum{suffix}.pdf"), cfg.paths.out(f"Joint_spectrum{suffix}.png")],
        cm=CM_PER_INCH, ylim=(3, 50), shade_spans=bin_shading_spans(m.freqs, STANDARD_BIN_EDGES),
    )
