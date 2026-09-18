"""Pulse-phase / date-formatting helpers, parametrised by an Ephemeris."""
from __future__ import annotations

import numpy as np


def ephem(n, ephemeris):
    """MJD of the n-th half-period after T0."""
    return ephemeris.T0 + n * ephemeris.P


def pulsenum(mjd, ephemeris):
    """Rotation-period pulse number at the given MJD (period = 2*P)."""
    return (mjd - ephemeris.T0) / (2 * ephemeris.P)


def ipulsenum(mjd, ephemeris):
    """Half-period pulse number at the given MJD."""
    return (mjd - ephemeris.T0) / ephemeris.P


def phase_of(mjd_days, ephemeris, period_offset=0.0):
    """Fold MJD (in days) onto [0, 1) phase, period = 2*P, with an optional
    extra offset (in units of P) before folding -- used to re-centre the
    main pulse for different plots."""
    trange = mjd_days
    P = ephemeris.P
    return np.mod(trange - ephemeris.T0 + period_offset * P, 2 * P) / (2 * P)


def phase_window_mask(phase, start, end):
    """Boolean mask selecting `phase` values inside the window [start, end).
    Handles windows that wrap around phase 0 (e.g. the main pulse here is
    centred on phase 0, so its window is given as start=0.94, end=0.05):
    when `start <= end` this is the usual `start < phase < end`; when
    `start > end` (a wrapping window) it's `phase > start OR phase < end`.
    """
    if start <= end:
        return np.logical_and(phase > start, phase < end)
    return np.logical_or(phase > start, phase < end)


def nicedate(t):
    """Format an astropy Time as 'YYYY-MM-DD HH:MM:SS' without excess precision."""
    return "{0}-{1:02.0f}-{2:02.0f} {3:02.0f}:{4:02.0f}:{5:02.0f}".format(
        t.ymdhms[0], t.ymdhms[1], t.ymdhms[2], t.ymdhms[3], t.ymdhms[4], t.ymdhms[5]
    )


def justdate(t):
    return "{0}-{1:02.0f}-{2:02.0f}".format(t.ymdhms[0], t.ymdhms[1], t.ymdhms[2])


def hhmm(t):
    return "{0:02.0f}:{1:02.0f}".format(t.ymdhms[3], t.ymdhms[4])
