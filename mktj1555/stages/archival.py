"""
Stage: build primary-beam-corrected, averaged dynamic spectra for every
ASKAP/MeerKAT observation of the field that did *not* detect the source,
and turn them into a stacked folded-light-curve figure, an upper-limits
figure/table, and a LaTeX observation table.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time, TimeDelta
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from ..config import AnalysisConfig, CM_PER_INCH
from ..detrend import deripple_short, deripple_long
from ..timing import justdate, hhmm
from .. import preprocessing


@dataclass
class ArchivalSource:
    """Source properties needed for the archival phase-folding, beyond
    what `config.Source`/`config.Ephemeris` already track. Position errors
    and RM are kept here for reference, but aren't currently used by
    anything in this stage."""

    coord: SkyCoord
    pos_error_a: u.Quantity
    pos_error_b: u.Quantity
    pos_error_pa: u.Quantity
    RM: u.Quantity
    period: u.Quantity
    period_err: u.Quantity
    pepoch: Time          # barycentric
    pepoch_top: Time      # topocentric
    ltt_bary: TimeDelta


def get_archival_source(cfg: AnalysisConfig) -> ArchivalSource:
    """Build the barycentric ephemeris used for phase-folding, from the
    same T0/period in `cfg.ephemeris` that the rest of the package uses."""
    askap_loc = EarthLocation.of_site("mwa")
    topocentric_pepoch = Time(cfg.ephemeris.T0, format="mjd", scale="utc", location=askap_loc)
    ltt_bary = topocentric_pepoch.light_travel_time(cfg.source.coord, kind="barycentric")
    barycentric_pepoch = topocentric_pepoch.tdb + ltt_bary
    # cfg.ephemeris.P is a *half*-period (see config.py); the archival fold
    # uses the full rotation period.
    period = (2 * cfg.ephemeris.P) * u.day
    period_err = 0.0000136 * u.day

    return ArchivalSource(
        coord=cfg.source.coord,
        pos_error_a=1 * u.arcsec, pos_error_b=1 * u.arcsec, pos_error_pa=0.1 * u.deg,
        RM=6 * u.rad / u.m ** 2,
        period=period, period_err=period_err,
        pepoch=barycentric_pepoch, pepoch_top=topocentric_pepoch, ltt_bary=ltt_bary,
    )


def make_light_curve(dsfile, coords, cfg):
    """Load one pickle, average the frequency axis (after clipping the
    noisiest channels), de-ripple, and return (barycentric times, light
    curve in mJy, centre frequency, topocentric start MJD)."""
    ds = np.load(dsfile, allow_pickle=True)
    stem = Path(dsfile).stem

    telescope = ds["TELESCOPE"]
    if telescope == "ASKAP":
        loc = EarthLocation.of_site("mwa")
    elif telescope == "MeerKAT":
        loc = EarthLocation.of_site("salt")
    else:
        loc = EarthLocation.of_site(telescope)
    times = Time(ds["TIMES"] / (24 * 3600), scale="utc", format="mjd", location=loc)
    tstart = times[0].mjd
    bary_tt = times.light_travel_time(coords, kind="barycentric")

    # Known one-off RFI/interference that wasn't already flagged upstream.
    if stem == "CB1716830412":
        ds["DS"][0:50, :, :] = np.nan       # an airplane flew through the first 50s
        ds["DS"][:, 315:535, :] = np.nan    # RFI that didn't get flagged for some reason
    if stem == "CB1636329974":
        ds["DS"][0:25, :, :] = np.nan       # ditto, another airplane

    It = np.real(ds["DS"][:, :, 0] + ds["DS"][:, :, 3]) / 2
    Qt = np.real(ds["DS"][:, :, 0] - ds["DS"][:, :, 3]) / 2
    # CB1716830412 has no usable Stokes Q, so fall back to Stokes I for the
    # per-channel noise estimate used to pick which channels to average.
    spec_std = np.nanstd(It if stem == "CB1716830412" else Qt, axis=0)

    xrange = np.arange(len(spec_std))
    deg = 3
    ok = ~np.isnan(spec_std)
    p = np.polynomial.Polynomial.fit(xrange[ok], spec_std[ok], deg=deg)
    # Two rounds of sigma-clipping against the polynomial baseline.
    for _ in range(2):
        std_spec_std = np.nanstd(spec_std - p(xrange))
        ok = np.logical_and(~np.isnan(spec_std), np.abs(spec_std - p(xrange)) < std_spec_std)
        p = np.polynomial.Polynomial.fit(xrange[ok], spec_std[ok], deg=deg)
    med_spec_std = np.nanmedian(spec_std - p(xrange))
    std_spec_std = np.nanstd(spec_std - p(xrange))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(spec_std - p(xrange))
    ax.set_ylabel("standard deviation - poly offset")
    ax.set_xlabel("channel index")
    ax.axhline(med_spec_std + 1.5 * std_spec_std, color="red")
    fig.savefig(cfg.paths.out(f"sanity_check_{stem}.png"), bbox_inches="tight")
    plt.close(fig)

    good_channels = (spec_std - p(xrange)) < (med_spec_std + 1.5 * std_spec_std)
    lc = np.nanmean(It[:, good_channels], axis=1)
    lc[lc == 0.0] = np.nan

    # The discovery observation needs per-segment baseline removal; every
    # other (much shorter) observation is fine with a single global fit.
    if telescope == "MeerKAT" and stem.startswith("CB1652551867"):
        print(f"derippling (long) {dsfile}")
        lc, _bounds = deripple_long(times.mjd * 24 * 3600, lc)
    else:
        print(f"derippling {dsfile}")
        lc = deripple_short(times.mjd * 24 * 3600, lc)

    return times.tdb + bary_tt, 1000 * lc, ds["FREQS"][len(ds["FREQS"]) // 2], tstart


def compute_phase(t_abs: Time, t0: Time, period: u.Quantity, phi0: float) -> Tuple[np.ndarray, np.ndarray]:
    cycles = ((t_abs - t0) / period).decompose() + phi0
    return cycles - np.floor(cycles), cycles


def plot_rows(ax, ts, source, phase_errors, phase, flux, color, rows) -> dict:
    """Plot an observation that may span multiple rows: for each contiguous
    run of equal row index, draw one line segment offset vertically by
    `row_step * row_index`."""
    scale = 0.75
    breaks = list(np.flatnonzero(rows[1:] != rows[:-1]) + 1)
    step_size = 10
    starts, stops = [0, *breaks], [*breaks, len(rows)]

    for start, stop in zip(starts, stops):
        row = rows[start]
        ytrace = scale * flux[start:stop] - step_size * row
        ax.plot(phase[start:stop], ytrace, color=color, lw=0.75, alpha=0.8)

        time_err = source.period_err / source.period * (ts[start] - source.pepoch)
        phase_err = (time_err / source.period).to(1)
        phase_errors["rows"].append(-row * step_size)
        phase_errors["mins"].append(0.5 - phase_err)
        phase_errors["maxs"].append(0.5 + phase_err)

    return phase_errors


def plot_folded_lightcurves(ax, ts, lcs, colors, source):
    """Folded light-curve plot obeying: phase(t) = ((t - t0)/P + phi0) % 1;
    within an observation, phase wrapping to 1 moves to the next row; each
    new observation starts on the next row at its own intrinsic phase."""
    next_base_row = 0
    ytick_pos, ytick_lab = [], []
    phase_errors = {"rows": [], "mins": [], "maxs": []}

    for t, lc, color in zip(ts, lcs, colors):
        row_step = 10
        phase, cycles = compute_phase(t, t0=source.pepoch, period=source.period, phi0=0.5)

        base_row = next_base_row
        row_indices = base_row + (np.floor(cycles) - np.floor(cycles[0])).astype(int)

        ytick_pos.append(-row_step * base_row)
        ytick_lab.append(t[0].isot.split("T")[0])

        phase_errors = plot_rows(ax, t, source, phase_errors, phase, lc, color, row_indices)
        next_base_row = row_indices.max() + 1

    try:
        ax.fill_betweenx(phase_errors["rows"], phase_errors["mins"], phase_errors["maxs"], color="g", alpha=0.1)
    except ValueError:
        pass

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Phase")
    # phi0=0.5 above shifts phase 0.5 to the plot's left edge; these labels
    # need to change too if phi0 changes.
    ax.set_xticklabels(["0.5", "0.75", "0.0", "0.25", "0.5"])
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lab)
    return ax


def run_archival_pipeline(cfg: AnalysisConfig):
    """The full archival pipeline: preprocess (if requested), build every
    light curve, and write the stacked-lightcurve figure, upper-limits
    figure/table, and LaTeX observation table."""
    source = get_archival_source(cfg)
    dynspec_dir, averaged_dir = cfg.paths.dynspec_dir, cfg.paths.averaged_dir
    spec_index, cal_freq = cfg.archival.spec_index, cfg.archival.cal_freq_hz
    outpath = cfg.paths.out

    # Preprocessing is always run -- but average_askap_beams/correct_meerkat_beams
    # skip any output file that already exists (unless force=True), so this
    # is cheap on repeat runs and doesn't need its own on/off switch.
    preprocessing.average_askap_beams(cfg, force=cfg.stages.askap_pb_corr)
    preprocessing.correct_meerkat_beams(cfg, force=cfg.stages.mkt_pb_corr)

    dynspecs = sorted(glob.glob(f"{averaged_dir}/SB*.pkl"))
    sbids = np.array([int(Path(f).stem.split("SB")[1]) for f in dynspecs])
    dynspecs_m = sorted(glob.glob(f"{averaged_dir}/CB*pkl"))
    cbids = np.array([int(Path(f).stem.split("CB")[1]) for f in dynspecs_m])

    tstarts, obslengths, ts, lcs, maxs, rmss, colors, freqs = [], [], [], [], [], [], [], []
    for pkl in dynspecs:
        print(f"Making light curve from {pkl}")
        t, l, fc, tstart = make_light_curve(pkl, source.coord, cfg)
        tstarts.append(tstart)
        obslengths.append(24 * 60 * (t[-1].mjd - t[0].mjd))
        ts.append(t)
        lcs.append(l)
        maxs.append(np.nanmax(l))
        rmss.append(np.nanstd(l))
        freqs.append(fc)
        colors.append("black")

    for pkl in dynspecs_m:
        print(f"Making light curve from {pkl}")
        t, l, fc, tstart = make_light_curve(pkl, source.coord, cfg)
        tstarts.append(tstart)
        obslengths.append(24 * 60 * (t[-1].mjd - t[0].mjd))
        ts.append(t)
        lcs.append(l)
        maxs.append(np.nanmax(l))
        rmss.append(np.nanstd(l))
        freqs.append(fc)
        colors.append("purple")

    ids = np.concatenate([sbids, cbids])
    order = np.argsort(tstarts)
    ts, lcs, colors, maxs, rmss, freqs, ids, obslengths, tstarts = (
        [seq[i] for i in order] for seq in (ts, lcs, colors, maxs, rmss, freqs, list(ids), obslengths, tstarts)
    )
    tstarts = sorted(tstarts)

    _plot_stacked_lightcurves(ts, lcs, colors, source, outpath)
    _plot_upper_limits(tstarts, rmss, maxs, colors, freqs, ids, spec_index, cal_freq, outpath)
    _write_observation_table(ids, tstarts, freqs, obslengths, dynspec_dir, outpath)


def _plot_stacked_lightcurves(ts, lcs, colors, source, outpath):
    cm = CM_PER_INCH
    fig = plt.figure(figsize=(17 * cm, 12.5 * cm))
    gs = GridSpec(1, 3, figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # The first ~18 observations are very long; the rest are short -- split
    # across panels purely so each panel's rows are a readable height.
    slices = [slice(0, 18), slice(18, 35), slice(35, None)]
    for ax, sl in zip(axes, slices):
        plot_folded_lightcurves(ax, ts[sl], lcs[sl], colors[sl], source=source)
    fig.tight_layout()

    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin + 20, ymax - 20)  # original padding was oddly large; trim it
    fig.savefig(outpath("observations_stacked.png"), format="png")
    fig.savefig(outpath("observations_stacked.pdf"), format="pdf")
    plt.close(fig)


def _plot_upper_limits(tstarts, rmss, maxs, colors, freqs, ids, spec_index, cal_freq, outpath):
    cm = CM_PER_INCH
    fig = plt.figure(figsize=(17 * cm, 5 * cm))
    ax = fig.add_subplot(111)
    for t, rms, mx, color, f in zip(tstarts, rmss, maxs, colors, freqs):
        if rms >= 50:  # one useless outlier point in this dataset
            continue
        scale = (cal_freq / f) ** spec_index
        if mx / rms > 9:
            ax.scatter(t, mx * scale, color=color, marker="*", s=8)
            ax.errorbar(t, mx * scale, yerr=rms, color=color, elinewidth=0.5)
        else:
            ax.scatter(t, 3 * rms * scale, marker="v", color=color, s=8)
    ax.set_xlabel("MJD")
    ax.set_ylabel("Flux density (mJy)")
    legend_elements = [
        Line2D([0], [0], lw=0, markersize=4, markerfacecolor="none", markeredgecolor="k", marker="v",
               label=r"3-$\sigma$ upper limits"),
        Line2D([0], [0], lw=0, markersize=4, markerfacecolor="none", markeredgecolor="k", marker="*",
               label="Detections\n(brightest pulse)"),
    ]
    ax.legend(loc=1, handles=legend_elements)
    fig.savefig(outpath("Archival_upper_limits.pdf"), bbox_inches="tight")
    plt.close(fig)

    freqs_ghz = np.array(freqs) / 1e9
    scale = (cal_freq / np.array(freqs)) ** spec_index
    out = np.array([ids, tstarts, rmss, maxs, freqs_ghz, np.array(rmss) * scale, np.array(maxs) * scale])
    np.savetxt(outpath("Archival_measurements.csv"), out.T,
               fmt=["%s", "%5.8f", "%0.6f", "%0.6f", "%1.3f", "%0.6f", "%0.6f"],
               delimiter=",", header="#ID,MJD,RMS,MAX,F_GHz,RMS_scaled,MAX_scaled")


def _write_observation_table(ids, tstarts, freqs, obslengths, dynspec_dir, outpath):
    times = Time(tstarts, format="mjd", scale="utc")
    datestrs = [justdate(t) for t in times]
    hhmmstrs = [hhmm(t) for t in times]
    freqs_mhz = np.array(freqs) / 1e6

    with open(outpath("obs_table.tex"), "w") as f:
        for date, time, iid, freq, obslength in zip(datestrs, hhmmstrs, ids, freqs_mhz, obslengths):
            if iid < 999999:  # ASKAP: SBID
                beams = [p.split("beam")[1][:2] for p in sorted(glob.glob(f"{dynspec_dir}/*SB{iid}*pkl"))]
                f.write(f"{date} & {time} & ASKAP & SB{iid} beams:{','.join(beams)}"
                        f"& {freq:4.0f} & {obslength:2.0f} \\\\\n")
            else:  # MeerKAT: coherent-beam ID
                f.write(f"{date} & {time} & MeerKAT & CB{iid} & {freq:4.0f} & {obslength:2.0f} \\\\\n")
