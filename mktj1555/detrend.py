"""
MeerKAT Stokes I has a slow instrumental baseline ripple that needs
subtracting before the light curve is usable. Handles any number of
contiguous segments (split wherever there's a gap in the timestamps), each
with its own sigma-clipped polynomial baseline fit.
"""
from __future__ import annotations

import numpy as np


def deripple_short(t, lc, deg=3):
    """Remove a slow instrumental baseline ripple by subtracting a single
    degree-`deg` polynomial fit over the whole light curve. Fine for
    observations short enough that the ripple doesn't change shape."""
    ok = ~np.isnan(lc)
    p = np.polynomial.Polynomial.fit(t[ok], lc[ok], deg=deg)
    return lc - p(t)


def segment_bounds(times, gap_threshold=50):
    """Split `times` into contiguous segments wherever consecutive samples
    are more than `gap_threshold` (seconds) apart. Returns boundary indices
    [0, ..., len(times)] such that segment i runs from bounds[i]:bounds[i+1]."""
    tdiff = times[1:] - times[:-1]
    tbreak = np.where(np.abs(tdiff) > gap_threshold)[0]
    return [0] + [b + 1 for b in tbreak] + [len(times)]


def deripple_long(times, lc, cutoff_jy=0.003, deg=3, gap_threshold=50):
    """As `deripple_short`, but for observations long enough that the
    ripple's shape changes across the observation -- split into contiguous
    segments (wherever there's a `gap_threshold`-second gap in the
    timestamps) and sigma-clip + fit a separate polynomial baseline per
    segment. Returns the detrended light curve and the segment boundaries
    (so callers can e.g. pick a clean segment for a noise estimate)."""
    bounds = segment_bounds(times, gap_threshold=gap_threshold)
    detrended = np.array(lc, dtype=float, copy=True)

    for lo, hi in zip(bounds[:-1], bounds[1:]):
        t, y = times[lo:hi], detrended[lo:hi]
        p = np.polynomial.Polynomial.fit(t, y, deg=deg)
        keep = np.abs(y - p(t)) < cutoff_jy
        if keep.sum() > deg:  # enough points left to fit; else keep the unclipped fit
            p = np.polynomial.Polynomial.fit(t[keep], y[keep], deg=deg)
        detrended[lo:hi] = y - p(t)

    return detrended, bounds
