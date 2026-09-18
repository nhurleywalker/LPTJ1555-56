"""
Power-law spectral fitting.

The shared recipe used across several pulse/spectrum fits (spiky
main-pulse spectrum, two versions of the second MeerKAT interpulse, the
joint ASKAP+MeerKAT spectrum in Stokes I and V, and the folded EMU
main-pulse/interpulse spectra): weight-and-collapse a dynamic spectrum down
to one spectrum, bin it for S/N, fit a power law with
`scipy.optimize.curve_fit`, draw a Monte Carlo uncertainty band from the
fit covariance, then make a log-log plot. This module factors that recipe
into reusable pieces so each call site is just "here's my data, here's how
I want it binned/plotted".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt

from .config import BOX_PROPERTIES


def pl(nu, norm, alpha):
    """Simple power law: S(nu) = norm * nu**alpha."""
    return norm * nu ** alpha


def normalized_weights(weight_curve, weight_floor):
    """Threshold a 1D weighting curve (e.g. a pulse's Stokes-I light curve)
    at `weight_floor` and normalise its peak to 1."""
    weights = np.array(weight_curve, dtype=float, copy=True)
    weights[weights < weight_floor] = 0.0
    peak = np.nanmax(weights)
    return weights / peak if peak > 0 else weights


def collapse_with_weights(data_2d, weights_1d):
    """Collapse a (time, frequency) array to a 1D spectrum using a
    precomputed 1D time-weighting (see `normalized_weights`). Frequency
    channels with zero total weight (e.g. fully RFI-flagged) come back as
    NaN rather than 0."""
    weights_2d = np.tile(np.asarray(weights_1d)[:, None], (1, data_2d.shape[1]))
    spectrum = np.nansum(np.squeeze(data_2d) * weights_2d, axis=0) / np.nansum(weights_2d, axis=0)
    return np.where(spectrum == 0.0, np.nan, spectrum)


def weighted_collapse(data_2d, weight_curve, freqs, weight_floor):
    """Convenience wrapper: threshold+normalise `weight_curve` then collapse
    `data_2d` with it. `freqs` is only used for its length."""
    weights = normalized_weights(weight_curve, weight_floor)
    return collapse_with_weights(data_2d, weights), weights


def bin_by_edges(freqs, flux, edges):
    """Bin a spectrum into contiguous chunks split at the given channel
    indices `edges` (the last chunk runs to the end of the array). Returns
    per-bin mean frequency (over non-NaN flux channels), mean flux, and the
    standard error on that mean."""
    bounds = [0] + list(edges) + [len(freqs)]
    freqs_b, flux_b, err_b = [], [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        f_chunk, x_chunk = freqs[lo:hi], flux[lo:hi]
        ok = ~np.isnan(x_chunk)
        freqs_b.append(np.nanmean(f_chunk[ok]))
        flux_b.append(np.nanmean(x_chunk))
        err_b.append(np.nanstd(x_chunk) / np.sqrt(ok.sum()))
    return np.array(freqs_b), np.array(flux_b), np.array(err_b)


def bin_by_split(freqs, flux, err, n_splits):
    """Bin a spectrum into `n_splits` equal-size chunks (via np.array_split),
    as used for the folded EMU main-pulse/interpulse spectra."""
    f_chunks = np.array_split(freqs, n_splits)
    x_chunks = np.array_split(flux, n_splits)
    r_chunks = np.array_split(err, n_splits)
    n_b = np.array([len(c) for c in f_chunks])
    freqs_b = np.array([np.mean(c) for c in f_chunks])
    flux_b = np.array([np.mean(c) for c in x_chunks])
    err_b = np.array([np.mean(c) for c in r_chunks]) / np.sqrt(n_b)
    return freqs_b, flux_b, err_b


def bin_shading_spans(freqs, edges):
    """For the standard 5-edge binning pattern (`edges` = [e0..e4], giving 6
    bins), returns the three (freq_lo, freq_hi) spans to highlight yellow:
    bins 2, 4, and 6."""
    e = edges
    return [(freqs[e[0]], freqs[e[1]]), (freqs[e[2]], freqs[e[3]]), (freqs[e[4]], freqs[-1])]


@dataclass
class PowerLawFit:
    best_p: np.ndarray
    err_p: np.ndarray
    nu: np.ndarray
    q16: np.ndarray
    q50: np.ndarray
    q84: np.ndarray

    @property
    def S_1GHz(self):
        return pl(1.0, *self.best_p)

    @property
    def alpha(self):
        return self.best_p[1]

    @property
    def alpha_err(self):
        return self.err_p[1]

    def report(self, label):
        return (
            f"Power-law fit parameters to {label}: "
            f"S at 1 GHz = {self.S_1GHz:3.2f}+/-{self.err_p[0]:3.2f}mJy, "
            f"alpha = {self.alpha:3.2f}+/-{self.alpha_err:3.2f}"
        )


def fit_powerlaw(freqs_b, flux_mjy_b, err_mjy_b, nu_range=(0.890, 1.700),
                  n_samples=1000, p0=None):
    """Fit S(nu) = norm * nu**alpha to binned flux-density data (in mJy),
    and draw `n_samples` posterior-covariance realisations to get a 16/50/84
    percentile uncertainty band over `nu_range`."""
    if p0 is None:
        p0 = (float(np.nanmedian(flux_mjy_b)), -0.7)
    nu = np.geomspace(*nu_range, 100)

    best_p, covar = curve_fit(
        pl, freqs_b, flux_mjy_b, p0, sigma=err_mjy_b, absolute_sigma=True
    )
    err_p = np.sqrt(np.diag(covar))

    samples = np.random.multivariate_normal(best_p, covar, size=n_samples).swapaxes(0, 1)
    models = pl(nu[:, None], *samples)
    q16, q50, q84 = np.percentile(models, [16, 50, 84], axis=1)

    return PowerLawFit(best_p=best_p, err_p=err_p, nu=nu, q16=q16, q50=q50, q84=q84)


@dataclass
class SpectrumSeries:
    """One dataset to overplot on a spectrum figure -- raw per-channel
    points plus the binned points used for the fit."""

    freqs: np.ndarray
    flux_mjy: np.ndarray
    freqs_b: np.ndarray
    flux_b_mjy: np.ndarray
    err_b_mjy: np.ndarray
    color: str
    label: str
    marker: str = "."
    raw_alpha: float = 0.3


def plot_powerlaw_spectrum(series, fit, out_paths, cm, ylim, xscale="log", yscale="log",
                            shade_spans=None, figsize_cm=(5, 5)):
    """Render the standard "raw points + binned points + power-law band"
    plot, for one or more `SpectrumSeries`.

    shade_spans: optional list of (freq_lo, freq_hi) pairs to mark with a
    light yellow band -- purely cosmetic highlighting of alternating
    frequency bins.
    """
    fig = plt.figure(figsize=(figsize_cm[0] * cm, figsize_cm[1] * cm))
    ax = fig.add_subplot(111)

    for s in series:
        ax.scatter(s.freqs, s.flux_mjy, color=s.color, alpha=s.raw_alpha, marker=s.marker,
                   s=2, lw=0.5, zorder=10)
        ax.scatter(s.freqs_b, s.flux_b_mjy, color=s.color, alpha=0.9, marker=s.marker,
                   s=6, lw=0.5, zorder=10, label=s.label)
        ax.errorbar(s.freqs_b, s.flux_b_mjy, yerr=s.err_b_mjy, color=s.color, alpha=0.9,
                    elinewidth=0.5, lw=0, zorder=5)

    for lo, hi in (shade_spans or []):
        ax.axvspan(lo, hi, color="yellow", alpha=0.15)

    ax.plot(fit.nu, fit.q50, lw=0.5, color="red")
    ax.fill_between(fit.nu, fit.q16, fit.q84, alpha=0.3, color="red")

    ax.set_ylabel("Weighted brightness (mJy)")
    ax.set_xlabel("Frequency / GHz")
    ax.set_ylim(*ylim)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xscale == "log" and yscale == "log":
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax.yaxis.set_minor_formatter(StrMethodFormatter("{x:.0f}"))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
        ax.xaxis.set_minor_formatter(StrMethodFormatter("{x:.1f}"))
    ax.legend(loc=1)
    ax.text(0.1, 0.1, r"$\alpha = {0:3.2f}\pm{1:3.2f}$".format(fit.alpha, fit.alpha_err),
            transform=ax.transAxes, bbox=BOX_PROPERTIES)

    for path in out_paths:
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
