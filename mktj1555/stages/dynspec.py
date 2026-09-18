"""Stage: plain Stokes I/Q/U/V dynamic-spectrum images (no folding/fitting)."""
from __future__ import annotations

from ..config import AnalysisConfig, STOKES_CMAP, MAIN_PULSE_ZOOM
from ..data_io import AnalysisData
from ..plotting import make_dynspec


def plot_askap_dynspec(data: AnalysisData, cfg: AnalysisConfig):
    a = data.askap
    extent = [0, a.times_z[-1], a.freqs[0], a.freqs[-1]]

    vmin, vmax = -0.005, 0.03
    make_dynspec(a.It.T, vmin, vmax, STOKES_CMAP["I"], extent, cfg.paths.out("EMU_StokesI_dynspec.png"))
    make_dynspec(a.It_beam09.T, vmin, vmax, STOKES_CMAP["I"], extent, cfg.paths.out("EMU_beam09_StokesI_fixms_dynspec.png"))
    make_dynspec(a.It_beam15.T, vmin, vmax, STOKES_CMAP["I"], extent, cfg.paths.out("EMU_beam15_StokesI_fixms_dynspec.png"))

    vmin, vmax = -0.02, 0.02
    make_dynspec(a.Qt.T, vmin, vmax, STOKES_CMAP["Q"], extent, cfg.paths.out("EMU_StokesQ_dynspec.png"))
    make_dynspec(a.Qt_beam09.T, vmin, vmax, STOKES_CMAP["Q"], extent, cfg.paths.out("EMU_beam09_StokesQ_fixms_dynspec.png"))
    make_dynspec(a.Qt_beam15.T, vmin, vmax, STOKES_CMAP["Q"], extent, cfg.paths.out("EMU_beam15_StokesQ_fixms_dynspec.png"))

    make_dynspec(a.Ut.T, vmin, vmax, STOKES_CMAP["U"], extent, cfg.paths.out("EMU_StokesU_dynspec.png"))
    make_dynspec(a.Ut_beam09.T, vmin, vmax, STOKES_CMAP["U"], extent, cfg.paths.out("EMU_beam09_StokesU_fixms_dynspec.png"))
    make_dynspec(a.Ut_beam15.T, vmin, vmax, STOKES_CMAP["U"], extent, cfg.paths.out("EMU_beam15_StokesU_fixms_dynspec.png"))

    make_dynspec(a.Vt.T, vmin, vmax, STOKES_CMAP["V"], extent, cfg.paths.out("EMU_StokesV_dynspec.png"))
    make_dynspec(a.Vt_beam09.T, vmin, vmax, STOKES_CMAP["V"], extent, cfg.paths.out("EMU_beam09_StokesV_fixms_dynspec.png"))
    make_dynspec(a.Vt_beam15.T, vmin, vmax, STOKES_CMAP["V"], extent, cfg.paths.out("EMU_beam15_StokesV_fixms_dynspec.png"))


def plot_meerkat_dynspec(data: AnalysisData, cfg: AnalysisConfig):
    m = data.meerkat
    extent = [0, m.times_z[-1], m.freqs[0], m.freqs[-1]]

    vmin, vmax = -0.005, 0.03
    make_dynspec(m.It.T, vmin, vmax, STOKES_CMAP["I"], extent, cfg.paths.out("MeerKAT_StokesI_dynspec.png"))

    vmin, vmax = -0.005, 0.005
    make_dynspec(m.Qt.T, vmin, vmax, STOKES_CMAP["Q"], extent, cfg.paths.out("MeerKAT_StokesQ_dynspec.png"))
    make_dynspec(m.Ut.T, vmin, vmax, STOKES_CMAP["U"], extent, cfg.paths.out("MeerKAT_StokesU_dynspec.png"))
    make_dynspec(m.Vt.T, vmin, vmax, STOKES_CMAP["V"], extent, cfg.paths.out("MeerKAT_StokesV_dynspec.png"))

    # Zoom on the main-pulse microstructure window.
    lo, hi = MAIN_PULSE_ZOOM
    zoom_extent = [m.times_z[lo], m.times_z[hi], m.freqs[0], m.freqs[-1]]
    make_dynspec(m.Qt[lo:hi].T, vmin, vmax, STOKES_CMAP["Q"], zoom_extent,
                 cfg.paths.out("MeerKAT_StokesQ_dynspec_zoom.png"), imwidth=5)
    make_dynspec(m.Ut[lo:hi].T, vmin, vmax, STOKES_CMAP["U"], zoom_extent,
                 cfg.paths.out("MeerKAT_StokesU_dynspec_zoom.png"), imwidth=5)
    make_dynspec(m.Vt[lo:hi].T, vmin, vmax, STOKES_CMAP["V"], zoom_extent,
                 cfg.paths.out("MeerKAT_StokesV_dynspec_zoom.png"), imwidth=5)
