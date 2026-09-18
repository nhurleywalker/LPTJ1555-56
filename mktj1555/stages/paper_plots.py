"""Stage: the polished multi-panel Stokes I/Q/U/V figures made for the paper
(as opposed to the plain per-Stokes PNGs in `dynspec.py`/`lightcurves.py`)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ..config import AnalysisConfig, STOKES_COLOR, STOKES_CMAP, CM_PER_INCH
from ..data_io import AnalysisData
from ..timing import ephem, nicedate
from ..plotting import slow_fade_diverging_cmap
from astropy.time import Time


def plot_askap_paper_figure(data: AnalysisData, cfg: AnalysisConfig):
    """Single-segment (no time-gap breaks) 4-panel Stokes I/Q/U/V figure for
    the continuous ASKAP observation."""
    a = data.askap
    eph = cfg.ephemeris
    cm = CM_PER_INCH
    freq0 = 16  # first `freq0` channels are unreliable/unused for ASKAP

    fig = plt.figure(figsize=(17.9 * cm, 8 * cm))
    extent = [0, a.times_z[-1] / 3600, a.freqs[freq0], a.freqs[-1]]
    lw, alpha, vmin, vmax = 0.5, 0.8, -3, 20

    ax_I_ds = fig.add_axes([0.1, 0.53, 0.3, 0.3])
    cax_I = fig.add_axes([0.41, 0.53, 0.015, 0.3])
    Ids = ax_I_ds.imshow(1000 * a.It[:, freq0:].T, interpolation="none", origin="lower", vmin=vmin,
                          vmax=vmax, aspect="auto", cmap=STOKES_CMAP["I"], extent=extent)
    fig.colorbar(Ids, cax=cax_I)
    ax_I_lc = fig.add_axes([0.1, 0.83, 0.3, 0.1])
    ax_I_lc.plot(a.times_z / 3600, 1000 * a.ilc, lw=lw, color=STOKES_COLOR["I"], alpha=alpha, label="I")

    slow_fade_cmap = slow_fade_diverging_cmap()
    from matplotlib import colors
    max_abs = 20
    my_norm = colors.Normalize(vmin=-max_abs, vmax=max_abs)

    ax_Q_ds = fig.add_axes([0.5, 0.53, 0.3, 0.3])
    cax_Q = fig.add_axes([0.81, 0.53, 0.015, 0.3])
    Qds = ax_Q_ds.imshow(1000 * a.Qt[:, freq0:].T, interpolation="none", origin="lower", aspect="auto",
                          extent=extent, cmap=slow_fade_cmap, norm=my_norm)
    fig.colorbar(Qds, cax=cax_Q, label="Flux density / mJy")
    ax_Q_lc = fig.add_axes([0.5, 0.83, 0.3, 0.1])
    ax_Q_lc.plot(a.times_z / 3600, 1000 * a.qlc, lw=lw, color=STOKES_COLOR["Q"], alpha=alpha, label="Q")

    ax_U_ds = fig.add_axes([0.1, 0.1, 0.3, 0.3])
    cax_U = fig.add_axes([0.41, 0.1, 0.015, 0.3])
    Uds = ax_U_ds.imshow(1000 * a.Ut[:, freq0:].T, interpolation="none", origin="lower", aspect="auto",
                          extent=extent, cmap=slow_fade_cmap, norm=my_norm)
    fig.colorbar(Uds, cax=cax_U)
    ax_U_lc = fig.add_axes([0.1, 0.4, 0.3, 0.1])
    ax_U_lc.plot(a.times_z / 3600, 1000 * a.ulc, lw=lw, color=STOKES_COLOR["U"], alpha=alpha, label="U")

    ax_V_ds = fig.add_axes([0.5, 0.1, 0.3, 0.3])
    cax_V = fig.add_axes([0.81, 0.1, 0.015, 0.3])
    Vds = ax_V_ds.imshow(1000 * a.Vt[:, freq0:].T, interpolation="none", origin="lower", aspect="auto",
                          extent=extent, cmap=slow_fade_cmap, norm=my_norm)
    fig.colorbar(Vds, cax=cax_V, label="Flux density / mJy")
    ax_V_lc = fig.add_axes([0.5, 0.4, 0.3, 0.1])
    ax_V_lc.plot(a.times_z / 3600, 1000 * a.vlc, lw=lw, color=STOKES_COLOR["V"], alpha=alpha, label="V")

    offset = a.times[0]
    for ax in [ax_I_lc, ax_Q_lc, ax_U_lc, ax_V_lc]:
        ax.set_xlim(a.times_z[0] / 3600, a.times_z[-1] / 3600)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        for i in range(-3, 35):
            ax.axvline((ephem(i, eph) * 24 * 3600 - offset) / 3600, alpha=0.1, color="grey")
    for ax in [ax_I_ds, ax_U_ds]:
        ax.set_ylabel("Frequency / GHz")
    for ax in [ax_I_lc, ax_U_lc]:
        ax.set_ylabel("$S$ / mJy")

    tstart = nicedate(Time(a.times[0] / (24 * 3600), format="mjd", scale="utc"))
    for ax in [ax_U_ds, ax_V_ds]:
        ax.set_xlabel(f"Time / hours since {tstart}")
    for ax in [ax_Q_ds, ax_Q_lc, ax_I_ds, ax_I_lc, ax_U_lc, ax_V_lc]:
        ax.set_xticklabels([])
    for ax in [ax_Q_ds, ax_V_ds]:
        ax.set_yticklabels([])

    fig.savefig(cfg.paths.out("ASKAP_dynamic_spectra_lcs.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# --- MeerKAT version: same idea, but the observation has real time gaps, so
# each of the four Stokes panels is drawn as several "broken axis" segments
# that stay aligned with each other. -----------------------------------------

def _broken_axes_row(fig, rect, seg_starts, seg_ends, times_mz, gap=0.006, min_frac=0.03):
    left, bottom, width, height = rect
    durations = np.array([max(times_mz[e - 1] - times_mz[s], 1.0) for s, e in zip(seg_starts, seg_ends)],
                          dtype=float)
    durations = np.maximum(durations, min_frac * durations.sum())
    n = len(durations)
    avail = width - gap * (n - 1)
    seg_widths = durations / durations.sum() * avail

    axes, x0 = [], left
    for w in seg_widths:
        axes.append(fig.add_axes([x0, bottom, w, height]))
        x0 += w + gap
    return axes


def _draw_break_marks(ax_left, ax_right, d=1, size=3):
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=size, linestyle="none", color="k", mec="k",
                  mew=1, clip_on=False)
    ax_left.plot([1], [0], transform=ax_left.transAxes, **kwargs)
    ax_left.plot([1], [1], transform=ax_left.transAxes, **kwargs)
    ax_right.plot([0], [0], transform=ax_right.transAxes, **kwargs)
    ax_right.plot([0], [1], transform=ax_right.transAxes, **kwargs)


def _style_segment_row(axes, seg_starts, seg_ends, times_mz, show_x_ticklabels, show_y_ticklabels):
    """Shared cosmetics for one row of broken sub-axes. Only the first
    (leftmost) segment ever shows y tick labels -- the rest share that axis.
    Uses tick_params(labelleft=False) rather than set_yticklabels([]),
    because sharey() links the tick formatter across axes and
    set_yticklabels([]) on one would wipe labels on all of them."""
    for ax, s, e in zip(axes, seg_starts, seg_ends):
        ax.set_xlim(times_mz[s] / 3600, times_mz[e - 1] / 3600)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    for ax in axes[1:]:
        ax.sharey(axes[0])
        ax.tick_params(labelleft=False)
    if not show_y_ticklabels:
        for ax in axes:
            ax.tick_params(labelleft=False)
    for ax_l, ax_r in zip(axes[:-1], axes[1:]):
        _draw_break_marks(ax_l, ax_r)
    if not show_x_ticklabels:
        for ax in axes:
            ax.tick_params(labelbottom=False)


def _plot_ephem_lines(ax, offset, eph):
    for i in range(-3, 35):
        ax.axvline((ephem(i, eph) * 24 * 3600 - offset) / 3600, alpha=0.1, color="grey")


def _set_matching_ylim(axes_groups, data_arrays, pad=0.08):
    """Force the same y-limits across two (or more) light-curve rows,
    computed directly from the data rather than chaining sharey() calls
    (which can leave stale references when chained across quadrants)."""
    vals = np.concatenate([1000 * np.asarray(d) for d in data_arrays])
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    for axes in axes_groups:
        for ax in axes:
            ax.set_ylim(lo, hi)


def _plot_broken_stokes(fig, ds_rect, lc_rect, cax_rect, data2d, data1d, seg_starts, seg_ends,
                         times_mz, freqs_m, offset, eph, imshow_kwargs, line_color, label,
                         freq0, cbar_label=None, show_x_ticklabels=False, show_y_ticklabels=True):
    ds_axes = _broken_axes_row(fig, ds_rect, seg_starts, seg_ends, times_mz)
    lc_axes = _broken_axes_row(fig, lc_rect, seg_starts, seg_ends, times_mz)

    im = None
    for ax, s, e in zip(ds_axes, seg_starts, seg_ends):
        extent = [times_mz[s] / 3600, times_mz[e - 1] / 3600, freqs_m[freq0], freqs_m[-1]]
        im = ax.imshow(1000 * data2d[s:e, freq0:].T, interpolation="none", origin="lower", aspect="auto",
                        extent=extent, **imshow_kwargs)
    for ax, s, e in zip(lc_axes, seg_starts, seg_ends):
        ax.plot(times_mz[s:e] / 3600, 1000 * data1d[s:e], lw=0.5, color=line_color, alpha=0.8, label=label)
        _plot_ephem_lines(ax, offset, eph)

    _style_segment_row(ds_axes, seg_starts, seg_ends, times_mz, show_x_ticklabels, show_y_ticklabels)
    _style_segment_row(lc_axes, seg_starts, seg_ends, times_mz, False, show_y_ticklabels)
    lc_axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    cax = fig.add_axes(cax_rect)
    fig.colorbar(im, cax=cax, label=cbar_label)
    return ds_axes, lc_axes, cax


def plot_meerkat_paper_figure(data: AnalysisData, cfg: AnalysisConfig):
    m = data.meerkat
    eph = cfg.ephemeris
    cm = CM_PER_INCH
    freq0 = 16

    tdiff = m.times_z[1:] - m.times_z[:-1]
    tbreak = np.where(np.abs(tdiff) > 50)[0]
    seg_starts = np.concatenate(([0], tbreak + 1))
    seg_ends = np.concatenate((tbreak + 1, [len(m.times_z)]))

    slow_fade_cmap = slow_fade_diverging_cmap()
    from matplotlib import colors
    my_norm = colors.Normalize(vmin=-20, vmax=20)

    fig = plt.figure(figsize=(17.9 * cm, 8 * cm))
    offset = m.times[0]
    I_kwargs = dict(vmin=-3, vmax=20, cmap=STOKES_CMAP["I"])
    QUV_kwargs = dict(cmap=slow_fade_cmap, norm=my_norm)

    I_ds, I_lc, _ = _plot_broken_stokes(
        fig, [0.1, 0.53, 0.3, 0.3], [0.1, 0.83, 0.3, 0.1], [0.41, 0.53, 0.015, 0.3],
        m.It, m.ilc, seg_starts, seg_ends, m.times_z, m.freqs, offset, eph, I_kwargs,
        STOKES_COLOR["I"], "I", freq0, show_x_ticklabels=False, show_y_ticklabels=True,
    )
    Q_ds, Q_lc, _ = _plot_broken_stokes(
        fig, [0.5, 0.53, 0.3, 0.3], [0.5, 0.83, 0.3, 0.1], [0.81, 0.53, 0.015, 0.3],
        m.Qt_corr, m.qlc_corr, seg_starts, seg_ends, m.times_z, m.freqs, offset, eph, QUV_kwargs,
        STOKES_COLOR["Q"], "Q", freq0, cbar_label="Flux density / mJy",
        show_x_ticklabels=False, show_y_ticklabels=False,
    )
    U_ds, U_lc, _ = _plot_broken_stokes(
        fig, [0.1, 0.1, 0.3, 0.3], [0.1, 0.4, 0.3, 0.1], [0.41, 0.1, 0.015, 0.3],
        m.Ut_corr, m.ulc_corr, seg_starts, seg_ends, m.times_z, m.freqs, offset, eph, QUV_kwargs,
        STOKES_COLOR["U"], "U", freq0, show_x_ticklabels=True, show_y_ticklabels=True,
    )
    V_ds, V_lc, _ = _plot_broken_stokes(
        fig, [0.5, 0.1, 0.3, 0.3], [0.5, 0.4, 0.3, 0.1], [0.81, 0.1, 0.015, 0.3],
        m.Vt, m.vlc, seg_starts, seg_ends, m.times_z, m.freqs, offset, eph, QUV_kwargs,
        STOKES_COLOR["V"], "V", freq0, cbar_label="Flux density / mJy",
        show_x_ticklabels=True, show_y_ticklabels=False,
    )

    _set_matching_ylim([I_lc, Q_lc], [m.ilc, m.qlc_corr])
    _set_matching_ylim([U_lc, V_lc], [m.ulc_corr, m.vlc])

    I_ds[0].set_ylabel("Frequency / GHz")
    U_ds[0].set_ylabel("Frequency / GHz")
    I_lc[0].set_ylabel("$S$ / mJy")
    U_lc[0].set_ylabel("$S$ / mJy")

    tstart = nicedate(Time(m.times[0] / (24 * 3600), format="mjd", scale="utc"))
    for rect in [[0.1, 0.1, 0.3, 0.3], [0.5, 0.1, 0.3, 0.3]]:
        left, bottom, width, height = rect
        fig.text(left + width / 2, bottom - 0.08, f"Time / hours since {tstart}", ha="center", va="top")

    fig.savefig(cfg.paths.out("MeerKAT_dynamic_spectra_lcs.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)
