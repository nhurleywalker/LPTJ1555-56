"""Stage: power-law spectral fits to specific pulses in the MeerKAT data.

These three analyses (main-pulse "spiky" microstructure, and the second
interpulse fit two ways -- via Stokes V and via Stokes I) are unified
around the shared `fitting` helpers; the only real differences between them
are which Stokes parameter drives the weighting, the weight threshold, and
whether a background is subtracted first.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..config import AnalysisConfig, STOKES_COLOR, CM_PER_INCH, MAIN_PULSE_ZOOM, SECOND_INTERPULSE_ZOOM
from ..data_io import AnalysisData
from ..plotting import make_dynspec
from ..fitting import (
    normalized_weights, collapse_with_weights, bin_by_edges, fit_powerlaw,
    bin_shading_spans, SpectrumSeries, plot_powerlaw_spectrum,
)

STANDARD_BIN_EDGES = [110, 240, 470, 700, 860]


def _sanity_check_weights(weights, ylabel, out_path):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.plot(weights)
    ax.set_xlabel("index")
    ax.set_ylabel(ylabel)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fit_main_pulse_spectrum(data: AnalysisData, cfg: AnalysisConfig):
    """"Spiky" main-pulse microstructure spectrum: background-subtract,
    weight by the Stokes-I light curve, fit a power law."""
    m = data.meerkat
    lo, hi = MAIN_PULSE_ZOOM
    edge = 17

    bkg_full = np.tile(np.nanmean(np.vstack([m.It[lo:lo + edge], m.It[hi - edge:hi]]), axis=0), (hi - lo, 1))
    bkg = np.tile(np.nanmean(np.vstack([m.It[lo:lo + edge], m.It[hi - edge:hi]]), axis=0), (hi - lo - 2 * edge, 1))
    make_dynspec((m.It[lo:hi] - bkg_full).T, -0.005, 0.03, "plasma",
                 [m.times_z[lo], m.times_z[hi], m.freqs[0], m.freqs[-1]],
                 cfg.paths.out("MeerKAT_StokesI_dynspec_spiky_zoom.png"), imwidth=5)

    It_c = m.It[lo + edge:hi - edge] - bkg
    Qt_c = m.Qt_corr[lo + edge:hi - edge]
    Ut_c = m.Ut_corr[lo + edge:hi - edge]
    Vt_c = m.Vt[lo + edge:hi - edge]

    weights_raw = np.nanmean(It_c, axis=1)
    _sanity_check_weights(weights_raw, "Stokes I light curve just of spiky pulse",
                           cfg.paths.out("test_mp_weights.png"))
    weights = normalized_weights(weights_raw, weight_floor=0.005)

    I_c = collapse_with_weights(It_c, weights)
    Q_c = collapse_with_weights(Qt_c, weights)
    U_c = collapse_with_weights(Ut_c, weights)
    V_c = collapse_with_weights(Vt_c, weights)

    freqs_b, I_b, err_b = bin_by_edges(m.freqs, I_c, STANDARD_BIN_EDGES)
    fit = fit_powerlaw(freqs_b, 1000 * I_b, 1000 * err_b)
    print(fit.report("MeerKAT 'spiky' pulse"))

    series = [SpectrumSeries(m.freqs, 1000 * I_c, freqs_b, 1000 * I_b, 1000 * err_b,
                              color="purple", label="MeerKAT")]
    plot_powerlaw_spectrum(
        series, fit,
        [cfg.paths.out("Spiky_pulse_spectrum.png"), cfg.paths.out("MeerKAT_main_pulse_spectrum.pdf")],
        cm=CM_PER_INCH, ylim=(3, 30), shade_spans=bin_shading_spans(m.freqs, STANDARD_BIN_EDGES),
    )

    # RMS from the first third of the observation, before the main pulse arrives.
    rms_arr = np.nanstd(m.Qt_corr[: int(m.Qt_corr.shape[1] // 3), :], axis=0)
    ok = ~np.isnan(I_c)
    out = np.array([m.freqs[ok] * 1e9, I_c[ok], Q_c[ok], U_c[ok], rms_arr[ok], rms_arr[ok], rms_arr[ok]])
    np.savetxt(cfg.paths.out("MeerKAT_IQU_spectrum.txt"), out.T)

    fig = plt.figure(figsize=(10 * CM_PER_INCH, 4 * CM_PER_INCH))
    ax = fig.add_subplot(111)
    for name, spec in (("Q", Q_c), ("U", U_c)):
        ax.scatter(m.freqs[ok], 1000 * spec[ok], color=STOKES_COLOR[name], alpha=0.8, marker=".",
                   s=4, lw=0.5, zorder=10, label=f"Stokes {name}")
        ax.errorbar(m.freqs[ok], 1000 * spec[ok], yerr=1000 * rms_arr[ok], color=STOKES_COLOR[name],
                    alpha=0.5, elinewidth=0.5, lw=0, zorder=5)
        ax.axhline(np.nanmean(1000 * spec[ok]), color=STOKES_COLOR[name], lw=0.5)
    ax.set_ylabel("Weighted brightness (mJy)")
    ax.set_xlabel("Frequency / GHz")
    ax.set_ylim(-20, 20)
    ax.legend()
    fig.savefig(cfg.paths.out("MeerKAT_MP_weighted_Stokes_spectra.pdf"), bbox_inches="tight")
    fig.savefig(cfg.paths.out("MeerKAT_MP_Stokes_spectra.png"), bbox_inches="tight")
    plt.close(fig)


def _fit_second_interpulse(data: AnalysisData, cfg: AnalysisConfig, weight_stokes: str,
                            weight_floor: float, subtract_background: bool):
    """Shared logic for the second MeerKAT interpulse fit, weighted either
    by -Stokes V (`weight_stokes="V"`) or Stokes I (`weight_stokes="I"`)."""
    m = data.meerkat
    lo, hi = SECOND_INTERPULSE_ZOOM
    edge = 17

    bkg_full = np.tile(np.nanmean(np.vstack([m.It[lo:lo + edge], m.It[hi - edge:hi]]), axis=0), (hi - lo, 1))
    make_dynspec((m.It[lo:hi] - bkg_full).T, -0.005, 0.03, "plasma",
                 [m.times_z[lo], m.times_z[hi], m.freqs[0], m.freqs[-1]],
                 cfg.paths.out("MeerKAT_StokesI_dynspec_IP2_zoom.png"), imwidth=5)

    if subtract_background:
        bkg = np.tile(np.nanmean(np.vstack([m.It[lo:lo + edge], m.It[hi - edge:hi]]), axis=0),
                       (hi - lo - 2 * edge, 1))
        It_c = m.It[lo + edge:hi - edge] - bkg
    else:
        It_c = m.It[lo + edge:hi - edge]
    Qt_c = m.Qt_corr[lo + edge:hi - edge]
    Ut_c = m.Ut_corr[lo + edge:hi - edge]
    Vt_c = m.Vt[lo + edge:hi - edge]

    if weight_stokes == "V":
        make_dynspec(-m.Vt[lo:hi].T, -0.005, 0.03, "bwr_r",
                     [m.times_z[lo], m.times_z[hi], m.freqs[0], m.freqs[-1]],
                     cfg.paths.out("MeerKAT_StokesI_dynspec_IP2_StokesV_zoom.png"), imwidth=5)
        weights_raw = np.nanmean(-Vt_c, axis=1)
        label = "-Stokes V light curve of 2nd interpulse"
        weight_file = "test_weights.png"
    else:
        weights_raw = np.nanmean(It_c, axis=1)
        label = "Stokes I light curve of 2nd interpulse"
        weight_file = "test_weights_I.png"
    _sanity_check_weights(weights_raw, label, cfg.paths.out(weight_file))
    weights = normalized_weights(weights_raw, weight_floor=weight_floor)

    I_c = collapse_with_weights(It_c, weights)
    sign = -1.0 if weight_stokes == "V" else 1.0
    fit_target = sign * collapse_with_weights(Vt_c, weights) if weight_stokes == "V" else I_c

    freqs_b, flux_b, err_b = bin_by_edges(m.freqs, fit_target, STANDARD_BIN_EDGES)
    fit = fit_powerlaw(freqs_b, 1000 * flux_b, 1000 * err_b)
    print(fit.report(f"MeerKAT 2nd interpulse using Stokes {weight_stokes}"))

    series = [SpectrumSeries(m.freqs, 1000 * fit_target, freqs_b, 1000 * flux_b, 1000 * err_b,
                              color="purple", label="MeerKAT")]
    suffix = "" if weight_stokes == "V" else "_I"
    plot_powerlaw_spectrum(
        series, fit,
        [cfg.paths.out(f"IP2_pulse_spectrum{suffix}.png"),
         cfg.paths.out(f"MeerKAT_interpulse_spectrum{'_StokesI' if suffix else ''}.pdf")],
        cm=CM_PER_INCH, ylim=(3, 30), shade_spans=bin_shading_spans(m.freqs, STANDARD_BIN_EDGES),
    )


def fit_second_interpulse_stokes_v(data: AnalysisData, cfg: AnalysisConfig):
    _fit_second_interpulse(data, cfg, weight_stokes="V", weight_floor=0.0025, subtract_background=True)


def fit_second_interpulse_stokes_i(data: AnalysisData, cfg: AnalysisConfig):
    _fit_second_interpulse(data, cfg, weight_stokes="I", weight_floor=0.0025, subtract_background=False)


def compare_band_split_lightcurve(data: AnalysisData, cfg: AnalysisConfig, indstart=460, indend=510):
    """Split the main pulse into a low- and high-frequency half, normalise
    each, and compare -- a check for frequency-dependent sub-structure."""
    m = data.meerkat
    bkg = np.tile(np.nanmean(np.vstack([m.It[indstart:indstart + 20], m.It[indend - 20:indend]]), axis=0),
                  (indend - indstart, 1))
    make_dynspec((m.It[indstart:indend] - bkg).T, -0.03, 0.030, "plasma",
                 [m.times_z[indstart], m.times_z[indend], m.freqs[0], m.freqs[-1]],
                 cfg.paths.out("MeerKAT_StokesI_dynspec_zoom.png"), imwidth=5)

    find1, find2, find3, find4 = 80, 300, 450, 850  # ~0.95-1.15 GHz and ~1.3-1.55 GHz
    ilc_low = np.nanmean(m.It[indstart:indend, find1:find2] - bkg[:, find1:find2], axis=1)
    ilc_high = np.nanmean(m.It[indstart:indend, find3:find4] - bkg[:, find3:find4], axis=1)

    deg, edge = 1, 8
    t = m.times_z[indstart:indend]
    t_fit = np.hstack([t[:edge], t[-edge:]])

    def _normalised_residual(y):
        p = np.polynomial.Polynomial.fit(t_fit, np.hstack([y[:edge], y[-edge:]]), deg=deg)
        resid = y - p(t)
        return resid / np.nanmax(resid)

    low_norm = _normalised_residual(ilc_low)
    high_norm = _normalised_residual(ilc_high)

    fig = plt.figure(figsize=(5, 8))
    ax = fig.add_subplot(211)
    ax.set_ylabel("brightness (mJy/beam)")
    ax.plot(t, low_norm, lw=0.5, color="red", alpha=0.8, label="Lower band")
    ax.plot(t, high_norm, lw=0.5, color="blue", alpha=0.8, label="Upper band")
    ax.legend(loc=1)
    axr = fig.add_subplot(212)
    axr.set_ylabel("brightness (mJy/beam)")
    axr.set_xlabel("time / s")
    axr.plot(t, low_norm - high_norm, lw=0.5, color="black", alpha=0.8, label="Difference")
    fig.savefig(cfg.paths.out("substructure_band_comparison.png"), bbox_inches="tight")
    plt.close(fig)
