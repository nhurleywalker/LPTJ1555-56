"""Stage: Stokes I/Q/U/V light-curve plots and the plain-text exports used to
hand data to collaborators."""
from __future__ import annotations

import numpy as np

from ..config import AnalysisConfig, STOKES_COLOR
from ..data_io import AnalysisData
from ..plotting import make_lightcurve


def plot_askap_lightcurves(data: AnalysisData, cfg: AnalysisConfig):
    a = data.askap
    eph = cfg.ephemeris
    offset = a.times[0]

    vmin, vmax = -5, 15
    make_lightcurve(a.times_z, 1000 * a.ilc, vmin, vmax, 0.5, STOKES_COLOR["I"], 1.0, "Stokes I",
                     cfg.paths.out("EMU_StokesI_light_curve.png"), eph, offset=offset)
    make_lightcurve(a.times_z, 1000 * a.ilc_beam09, vmin, vmax, 0.5, STOKES_COLOR["I"], 1.0, "Stokes I",
                     cfg.paths.out("EMU_beam09_StokesI_fixms_light_curve.png"), eph, offset=offset)
    make_lightcurve(a.times_z, 1000 * a.ilc_beam15, vmin, vmax, 0.5, STOKES_COLOR["I"], 1.0, "Stokes I",
                     cfg.paths.out("EMU_beam15_StokesI_fixms_light_curve.png"), eph, offset=offset)

    vmin, vmax = -15, 15
    for name, lc, lc09, lc15 in [
        ("Q", a.qlc, a.qlc_beam09, a.qlc_beam15),
        ("U", a.ulc, a.ulc_beam09, a.ulc_beam15),
        ("V", a.vlc, a.vlc_beam09, a.vlc_beam15),
    ]:
        make_lightcurve(a.times_z, 1000 * lc, vmin, vmax, 0.5, STOKES_COLOR[name], 1.0, f"Stokes {name}",
                         cfg.paths.out(f"EMU_Stokes{name}_light_curve.png"), eph, offset=offset)
        make_lightcurve(a.times_z, 1000 * lc09, vmin, vmax, 0.5, STOKES_COLOR[name], 1.0, f"Stokes {name}",
                         cfg.paths.out(f"EMU_beam09_Stokes{name}_fixms_light_curve.png"), eph, offset=offset)
        make_lightcurve(a.times_z, 1000 * lc15, vmin, vmax, 0.5, STOKES_COLOR[name], 1.0, f"Stokes {name}",
                         cfg.paths.out(f"EMU_beam15_Stokes{name}_fixms_light_curve.png"), eph, offset=offset)


def save_askap_lightcurve_table(data: AnalysisData, cfg: AnalysisConfig):
    """Save the Stokes I light curve as a plain-text table (time, flux, a
    representative noise estimate from Stokes Q). Historically handed off to
    collaborators doing independent checks."""
    a = data.askap
    ok = ~np.isnan(a.ilc)
    out = np.array([a.times[ok] / (24 * 3600), a.ilc[ok], a.noise_mjy / 1000 * np.ones(ok.sum())])
    np.savetxt(cfg.paths.out("ASKAP_StokesI_light_curve.txt"), out.T, fmt=["%5.8f", "%0.6f", "%0.5f"])
    print(f"Typical noise of EMU light curves is {a.noise_mjy:2.1f}mJy/beam")


def plot_meerkat_lightcurves(data: AnalysisData, cfg: AnalysisConfig):
    m = data.meerkat
    eph = cfg.ephemeris
    offset = m.times[0]

    vmin, vmax = -3, 20
    make_lightcurve(m.times_z, 1000 * m.ilc, vmin, vmax, 0.5, STOKES_COLOR["I"], 1.0, "Stokes I",
                     cfg.paths.out("MeerKAT_StokesI_light_curve.png"), eph, offset=offset)

    vmin, vmax = -20, 20
    make_lightcurve(m.times_z, 1000 * m.qlc, vmin, vmax, 0.5, STOKES_COLOR["Q"], 1.0, "Stokes Q",
                     cfg.paths.out("MeerKAT_StokesQ_light_curve.png"), eph, offset=offset)
    make_lightcurve(m.times_z, 1000 * m.ulc, vmin, vmax, 0.5, STOKES_COLOR["U"], 1.0, "Stokes U",
                     cfg.paths.out("MeerKAT_StokesU_light_curve.png"), eph, offset=offset)
    make_lightcurve(m.times_z, 1000 * m.qlc_corr, vmin, vmax, 0.5, STOKES_COLOR["Q"], 1.0, "Stokes Q",
                     cfg.paths.out("MeerKAT_StokesQ_pacorr_light_curve.png"), eph, offset=offset)
    make_lightcurve(m.times_z, 1000 * m.ulc_corr, vmin, vmax, 0.5, STOKES_COLOR["U"], 1.0, "Stokes U",
                     cfg.paths.out("MeerKAT_StokesU_pacorr_light_curve.png"), eph, offset=offset)
    make_lightcurve(m.times_z, 1000 * m.vlc, vmin, vmax, 0.5, STOKES_COLOR["V"], 1.0, "Stokes V",
                     cfg.paths.out("MeerKAT_StokesV_light_curve.png"), eph, offset=offset)

    make_lightcurve(
        [m.times_z] * 3, [1000 * m.qlc, 1000 * m.ulc, 1000 * m.vlc], vmin, vmax,
        [0.5, 0.5, 0.5], [STOKES_COLOR["Q"], STOKES_COLOR["U"], STOKES_COLOR["V"]], [1.0, 1.0, 1.0],
        ["Stokes Q", "Stokes U", "Stokes V"], cfg.paths.out("MeerKAT_StokesQUV_light_curve.png"),
        eph, offset=offset,
    )
    make_lightcurve(
        [m.times_z] * 3, [1000 * m.qlc_corr, 1000 * m.ulc_corr, 1000 * m.vlc], vmin, vmax,
        [0.5, 0.5, 0.5], [STOKES_COLOR["Q"], STOKES_COLOR["U"], STOKES_COLOR["V"]], [1.0, 1.0, 1.0],
        ["Stokes Q", "Stokes U", "Stokes V"], cfg.paths.out("MeerKAT_StokesQUV_pacorr_light_curve.png"),
        eph, offset=offset,
    )

    # Zoom on the main-pulse microstructure window (same one used for the
    # zoomed dynamic spectra).
    from ..config import MAIN_PULSE_ZOOM
    lo, hi = MAIN_PULSE_ZOOM
    vmin, vmax = -3, 12
    zoom_slice = slice(lo, hi)
    make_lightcurve(
        [m.times_z[zoom_slice]] * 4,
        [1000 * m.ilc[zoom_slice], 1000 * m.qlc[zoom_slice], 1000 * m.ulc[zoom_slice], 1000 * m.vlc[zoom_slice]],
        vmin, vmax, [0.5] * 4,
        [STOKES_COLOR["I"], STOKES_COLOR["Q"], STOKES_COLOR["U"], STOKES_COLOR["V"]], [0.8] * 4,
        ["Stokes I", "Stokes Q", "Stokes U", "Stokes V"],
        cfg.paths.out("MeerKAT_StokesIQUV_light_curve_zoom.png"), eph, offset=m.times[lo], imwidth=5,
    )
    # This one resembles the folded EMU data -- i.e. parallactic-angle
    # correction IS necessary to make Q/U comparable to ASKAP's.
    make_lightcurve(
        [m.times_z[zoom_slice]] * 4,
        [1000 * m.ilc[zoom_slice], 1000 * m.qlc_corr[zoom_slice], 1000 * m.ulc_corr[zoom_slice], 1000 * m.vlc[zoom_slice]],
        vmin, vmax, [0.5] * 4,
        [STOKES_COLOR["I"], STOKES_COLOR["Q"], STOKES_COLOR["U"], STOKES_COLOR["V"]], [0.8] * 4,
        ["Stokes I", "Stokes Q", "Stokes U", "Stokes V"],
        cfg.paths.out("MeerKAT_StokesIQUV_pacorr_light_curve_zoom.png"), eph, offset=m.times[lo], imwidth=5,
    )


def save_meerkat_lightcurve_table(data: AnalysisData, cfg: AnalysisConfig):
    m = data.meerkat
    ok = ~np.isnan(m.ilc)
    out = np.array([m.times[ok] / (24 * 3600), m.ilc[ok], (m.noise_mjy / 1000) * np.ones(ok.sum())])
    np.savetxt(cfg.paths.out("MeerKAT_StokesI_light_curve.txt"), out.T, fmt=["%5.8f", "%0.6f", "%0.5f"])
    print(f"Typical noise of MeerKAT light curves is {m.noise_mjy:2.2f}mJy/beam")
