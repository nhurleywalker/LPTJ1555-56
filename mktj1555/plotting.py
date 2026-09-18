"""Plotting and small array-wrangling helpers shared across analysis stages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors


def make_dynspec(data, vmin, vmax, cmap, extent, outname, imwidth=13):
    """Save a single dynamic-spectrum image (time vs frequency)."""
    fig = plt.figure(figsize=(imwidth, 5))
    ax = fig.add_subplot(111)
    ax.imshow(data, origin="lower", vmin=vmin, vmax=vmax, aspect="auto", cmap=cmap, extent=extent)
    ax.set_xlabel("time / s")
    ax.set_ylabel("frequency / GHz")
    fig.savefig(outname, bbox_inches="tight")
    plt.close(fig)


def make_lightcurve(times, lc, vmin, vmax, lw, color, alpha, label, outname, ephemeris,
                     offset=0.0, imwidth=13):
    """Save a single light-curve plot, with vertical lines marking each
    predicted pulse epoch. `times`/`lc`/`lw`/`color`/`alpha`/`label` can each
    either be a single array/value, or (matching lists) to overplot several
    curves on the same axes."""
    from .timing import ephem  # local import to avoid a circular import at module load

    fig = plt.figure(figsize=(imwidth, 5))
    ax = fig.add_subplot(111)
    ax.set_ylabel("brightness (mJy/beam)")
    ax.set_xlabel("time / s")
    if isinstance(alpha, list):
        for t, l, w, c, a, lb in zip(times, lc, lw, color, alpha, label):
            ax.plot(t, l, lw=w, color=c, alpha=a, label=lb)
    else:
        ax.plot(times, lc, lw=lw, color=color, alpha=alpha, label=label)
        t = times
    for i in range(-3, 35):
        ax.axvline((ephem(i, ephemeris) * 24 * 3600 - offset), alpha=0.4, color="orange")
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc=1)
    fig.savefig(outname, bbox_inches="tight")
    plt.close(fig)


def slow_fade_diverging_cmap(base="bwr", exponent=2.0, num_points=256):
    """A diverging colormap that fades to white more slowly near zero than
    the default, so noise-level Q/U/V values don't dominate the image."""
    base_cmap = plt.get_cmap(base)
    x = np.linspace(-1, 1, num_points)
    warped_x = np.sign(x) * (np.abs(x) ** exponent)
    colors_sampled = base_cmap((warped_x + 1) / 2)
    return colors.ListedColormap(colors_sampled)


def unwrap(x):
    """Map a negative angle (degrees) into [0, 360)."""
    return x + 360 if x < 0 else x


vunwrap = np.vectorize(unwrap, otypes=[float])


def pwrap(x, num_bins, add=False):
    """Re-centre a phase-folded array so it runs from phase 0.5 to 1.5
    instead of 0 to 1, avoiding a split across the phase-0 wrap point."""
    if add:
        return np.hstack([x[int(num_bins / 2):], x[: int(num_bins / 2)] + 1])
    return np.hstack([x[int(num_bins / 2):], x[: int(num_bins / 2)]])


def get_weighted_sum(bin_indices, weights):
    """np.bincount doesn't handle NaNs; this does, via a pandas groupby."""
    df = pd.DataFrame({"values": bin_indices, "weights": weights})
    return df.groupby("values")["weights"].sum()
