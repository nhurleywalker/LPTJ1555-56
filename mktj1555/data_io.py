"""
Load the ASKAP (EMU) and MeerKAT dynamic-spectrum pickles and turn them into
calibrated Stokes I/Q/U/V light curves and dynamic spectra.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.time import Time
from astroplan import Observer

from .config import AnalysisConfig
from .primary_beams import GaussianPB
from . import detrend, preprocessing

# Indices into the pickle's polarisation axis.
XX, XY, YX, YY = 0, 1, 2, 3


@dataclass
class ASKAPData:
    freqs: np.ndarray        # GHz
    times: np.ndarray        # MJD seconds
    times_z: np.ndarray      # MJD seconds, zeroed to the first sample

    # Primary-beam-weighted combination of beam09 + beam15 (the science data)
    It: np.ndarray
    Qt: np.ndarray
    Ut: np.ndarray
    Vt: np.ndarray
    ilc: np.ndarray
    qlc: np.ndarray
    ulc: np.ndarray
    vlc: np.ndarray

    # Individual-beam Stokes planes, kept around for sanity-check plots.
    It_beam09: np.ndarray
    Qt_beam09: np.ndarray
    Ut_beam09: np.ndarray
    Vt_beam09: np.ndarray
    ilc_beam09: np.ndarray
    qlc_beam09: np.ndarray
    ulc_beam09: np.ndarray
    vlc_beam09: np.ndarray

    It_beam15: np.ndarray
    Qt_beam15: np.ndarray
    Ut_beam15: np.ndarray
    Vt_beam15: np.ndarray
    ilc_beam15: np.ndarray
    qlc_beam15: np.ndarray
    ulc_beam15: np.ndarray
    vlc_beam15: np.ndarray

    noise_mjy: float          # representative RMS, from Stokes Q (little signal there)


@dataclass
class MeerKATData:
    freqs: np.ndarray
    times: np.ndarray
    times_z: np.ndarray
    pa: np.ndarray            # parallactic angle at each time sample (astropy Angle)

    It: np.ndarray
    Qt: np.ndarray             # uncorrected for parallactic angle
    Ut: np.ndarray
    Vt: np.ndarray
    Qt_corr: np.ndarray        # parallactic-angle corrected
    Ut_corr: np.ndarray

    ilc: np.ndarray            # detrended (see _detrend_meerkat_ripple)
    qlc: np.ndarray
    ulc: np.ndarray
    vlc: np.ndarray
    qlc_corr: np.ndarray
    ulc_corr: np.ndarray

    noise_mjy: float           # representative RMS, from a clean detrended segment
    segment_bounds: tuple       # (seg1_end, seg2_end, seg3_end) time-gap boundaries


@dataclass
class AnalysisData:
    askap: ASKAPData
    meerkat: MeerKATData


def _askap_stokes(ds):
    """FixMS-convention Stokes conversion (Alec's convention, matches WSClean)."""
    It = np.real(ds[:, :, XX] + ds[:, :, YY]) / 2
    Qt = np.real(ds[:, :, XX] - ds[:, :, YY]) / 2
    Ut = np.real(ds[:, :, XY] + ds[:, :, YX]) / 2
    Vt = np.real(-1j * ds[:, :, XY] + 1j * ds[:, :, YX]) / 2
    return It, Qt, Ut, Vt


def _apply_askap_rfi_flags(planes, rfi):
    """Flag one bad channel and the low-frequency edge, in place, on each of
    the given (time, freq) Stokes planes."""
    for plane in planes:
        plane[:, rfi.askap_bad_channel] = np.nan
        plane[:, rfi.askap_edge_channels] = np.nan


def _primary_beam_weight(freqs_ghz, separation_rad, n_time):
    """Inverse Gaussian-primary-beam correction, tiled to (time, freq),
    for a source offset `separation_rad` from the beam centre."""
    pb_vals = [
        GaussianPB(frequency=f * 1e9).evaluate(separation_rad, freq=f * 1e9)
        for f in freqs_ghz
    ]
    return 1.0 / np.tile(np.array(pb_vals), (n_time, 1))


def load_askap_data(cfg: AnalysisConfig) -> ASKAPData:
    arr09 = np.load(cfg.paths.askap_beam09, allow_pickle=True)
    arr15 = np.load(cfg.paths.askap_beam15, allow_pickle=True)

    freqs = arr09["FREQS"] / 1e9
    times = arr09["TIMES"]
    times_z = times - times[0]

    It09, Qt09, Ut09, Vt09 = _askap_stokes(arr09["DS"])
    It15, Qt15, Ut15, Vt15 = _askap_stokes(arr15["DS"])
    _apply_askap_rfi_flags([It09, Qt09, Ut09, Vt09, It15, Qt15, Ut15, Vt15], cfg.rfi)

    # Primary-beam correction: weight each beam by the inverse of its primary
    # beam response at the source's true offset, then combine (the two
    # normalisation factors cancel, leaving 1/sum-of-inverse-weights).
    seps = cfg.source.beam_coords.separation(cfg.source.coord)
    w09 = _primary_beam_weight(freqs, seps[0].rad, It09.shape[0])
    w15 = _primary_beam_weight(freqs, seps[1].rad, It15.shape[0])
    # Convert from correction factors (1/pb) to weights (pb) for the combine below.
    w09, w15 = 1.0 / w09, 1.0 / w15

    It = (It09 + It15) / (w09 + w15)
    Qt = (Qt09 + Qt15) / (w09 + w15)
    Ut = (Ut09 + Ut15) / (w09 + w15)
    Vt = (Vt09 + Vt15) / (w09 + w15)

    ilc, qlc, ulc, vlc = (np.nanmean(x, axis=1) for x in (It, Qt, Ut, Vt))
    ilc09, qlc09, ulc09, vlc09 = (np.nanmean(x, axis=1) for x in (It09, Qt09, Ut09, Vt09))
    ilc15, qlc15, ulc15, vlc15 = (np.nanmean(x, axis=1) for x in (It15, Qt15, Ut15, Vt15))

    noise_mjy = 1000 * np.nanstd(qlc)

    return ASKAPData(
        freqs=freqs, times=times, times_z=times_z,
        It=It, Qt=Qt, Ut=Ut, Vt=Vt, ilc=ilc, qlc=qlc, ulc=ulc, vlc=vlc,
        It_beam09=It09, Qt_beam09=Qt09, Ut_beam09=Ut09, Vt_beam09=Vt09,
        ilc_beam09=ilc09, qlc_beam09=qlc09, ulc_beam09=ulc09, vlc_beam09=vlc09,
        It_beam15=It15, Qt_beam15=Qt15, Ut_beam15=Ut15, Vt_beam15=Vt15,
        ilc_beam15=ilc15, qlc_beam15=qlc15, ulc_beam15=ulc15, vlc_beam15=vlc15,
        noise_mjy=noise_mjy,
    )


def _apply_meerkat_rfi_flags(ds, rfi):
    for t_slice, f_slice, p_slice in rfi.meerkat_time_freq_pol:
        ds[t_slice, f_slice, p_slice] = np.nan


def _detrend_meerkat_ripple(times, ilc):
    """MeerKAT Stokes I has a slow instrumental ripple (see `detrend.py`).
    Returns the detrended light curve, a representative RMS (measured on
    the second segment, which for the discovery observation is clean of any
    known feature), and the segment boundaries.
    """
    detrended, bounds = detrend.deripple_long(times, ilc)

    # Segment 2 (index 1) is clean of any known feature, so it's used for
    # the representative noise estimate, trimming 5 samples off each edge.
    # (Falls back to the first segment if there's only one -- shouldn't
    # happen for the discovery observation, but keeps this robust for
    # other MeerKAT pickles that might get routed through here.)
    seg = 1 if len(bounds) > 2 else 0
    seg_lo, seg_hi = bounds[seg], bounds[seg + 1]
    noise_mjy = 1000 * np.nanstd(detrended[seg_lo + 5:seg_hi - 5])

    return detrended, noise_mjy, tuple(bounds[1:4])


def load_meerkat_data(cfg: AnalysisConfig) -> MeerKATData:
    preprocessing.ensure_meerkat_file(cfg)
    mkt = np.load(cfg.paths.meerkat, allow_pickle=True)
    _apply_meerkat_rfi_flags(mkt["DS"], cfg.rfi)

    freqs = mkt["FREQS"] / 1e9
    times = mkt["TIMES"]
    times_z = times - times[0]

    meerkat_site = Observer.at_site("salt")
    pa = meerkat_site.parallactic_angle(
        Time(times / (24 * 3600), format="mjd", scale="utc"), cfg.source.coord
    )

    # "Alec's convention" -- effectively swaps Q and U relative to the naive
    # XX/YY/XY/YX combination.
    ds = mkt["DS"]
    It = np.real(ds[:, :, XX] + ds[:, :, YY]) / 2
    Qt = np.real(-ds[:, :, XY] - ds[:, :, YX]) / 2
    Ut = np.real(ds[:, :, XX] - ds[:, :, YY]) / 2
    Vt = np.real(-1j * ds[:, :, XY] + 1j * ds[:, :, YX]) / 2

    pa_tile = np.tile(pa[:, np.newaxis], (1, Qt.shape[1]))
    Qt_corr = Qt * np.cos(2 * pa_tile) + Ut * np.sin(2 * pa_tile)
    Ut_corr = Ut * np.cos(2 * pa_tile) - Qt * np.sin(2 * pa_tile)

    ilc_raw, qlc, ulc, vlc = (np.nanmean(x, axis=1) for x in (It, Qt, Ut, Vt))
    qlc_corr = qlc * np.cos(2 * pa) + ulc * np.sin(2 * pa)
    ulc_corr = ulc * np.cos(2 * pa) - qlc * np.sin(2 * pa)

    ilc, noise_mjy, segment_bounds = _detrend_meerkat_ripple(times, ilc_raw)

    return MeerKATData(
        freqs=freqs, times=times, times_z=times_z, pa=pa,
        It=It, Qt=Qt, Ut=Ut, Vt=Vt, Qt_corr=Qt_corr, Ut_corr=Ut_corr,
        ilc=ilc, qlc=qlc, ulc=ulc, vlc=vlc, qlc_corr=qlc_corr, ulc_corr=ulc_corr,
        noise_mjy=noise_mjy, segment_bounds=segment_bounds,
    )


def load_all(cfg: AnalysisConfig) -> AnalysisData:
    return AnalysisData(askap=load_askap_data(cfg), meerkat=load_meerkat_data(cfg))
