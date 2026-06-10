import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
 
import pickle
import astropy.units as u
import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time, TimeDelta
#from dstools.dynamic_spectrum import DynamicSpectrum, LightCurve
#from dstools.logger import setupLogger
#from dstools.utils import LOCATIONS
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

#logger = logging.getLogger(__name__)
cm = 1/2.54

FULL_WIDTH = 508 / 72.27

PARAMS = {
    "text.latex.preamble": "\\usepackage{gensymb}",
    "image.origin": "lower",
    "image.interpolation": "nearest",
    "image.cmap": "gray",
    "savefig.dpi": 300,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "font.size": 7,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 6,
}

matplotlib.rcParams.update(PARAMS)


@dataclass
class Source:
    coord: SkyCoord
    pos_error_a: u.Quantity
    pos_error_b: u.Quantity
    pos_error_pa: u.Quantity
    RM: u.Quantity
    period: u.Quantity
    period_err: u.Quantity
    pepoch: Time
    pepoch_top: Time
    ltt_bary: TimeDelta

def make_light_curve(dsfile, coords):
    ''' Import a pkl file, average the frequency axis, return the times and the light curve '''
    arr = np.load(dsfile, allow_pickle=True)
# Barycentre times
    telescope = arr["TELESCOPE"]
    if telescope == "ASKAP":
        loc = EarthLocation.of_site('mwa')
    else:
        loc = EarthLocation.of_site(telescope)
    times = Time(arr["TIMES"]/(24*3600), scale='utc', format='mjd', location=loc)
    bary_tt = times.light_travel_time(coords, kind="barycentric")

# Do some RFI flagging
    It = np.real(arr["DS"][:,:,0]+arr["DS"][:,:,3])/2
    Qt = np.real((arr["DS"][:,:,0]-arr["DS"][:,:,3]))/2
    spec_std = np.nanstd(Qt, axis=0)
    med_spec_std = np.nanmedian(spec_std)
    std_spec_std = np.nanstd(spec_std)
# TODO also fit a curve to this to improve flagging
# Sanity plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(spec_std)
    ax.set_ylabel("standard deviation")
    ax.set_xlabel("channel index")
    ax.axhline(med_spec_std+3*std_spec_std, color='red')
    stem = Path(dsfile).stem
    fig.savefig(f'sanity_check_{stem}.png', bbox_inches="tight")
    lc = 1000*np.nanmean(It[:,spec_std<(med_spec_std+3*std_spec_std)], axis=1)

    return times.tdb + bary_tt, lc

#@dataclass
#class Observation:
#    path: str
#    source: Source
#
#    def __post_init__(self):
#        self.ds = DynamicSpectrum(
#            self.path,
#            barycentre=True,
#        )
#        self.t_start = Time(
#            self.ds.header["time_start"],
#            scale=self.ds.header["time_scale"],
#        )
#
#        self.lc = LightCurve(self.ds)
#
#    @property
#    def t_abs(self):
#        return self.t_start + (self.lc.x * self.ds.tunit)
#
#    @property
#    def freq(self):
#        return self.ds.freq[0] + (self.ds.freq[-1] - self.ds.freq[0]) / 2


def get_source() -> Source:
    # Derived from MeerKAT observations
    j1555_coord = SkyCoord(
        ra="15h55m43.69s",
        dec="-56d31m02.4s",
        unit="hourangle,deg",
        frame="fk5",
    )
    # These are still preliminary
    j1555_pos_error_a = 1 * u.arcsec
    j1555_pos_error_b = 1 * u.arcsec
    j1555_pos_error_pa = 0.1 * u.deg

    # Rotation Measure -- probably won't be needed
    RM = 6 * u.rad / u.m**2

    # Pulse ephemeris
    askap_loc = EarthLocation.of_site('mwa')
    topocentric_pepoch = Time(59713.512505, format='mjd', scale="utc", location=askap_loc)
    ltt_bary = topocentric_pepoch.light_travel_time(j1555_coord, kind="barycentric")
    barycentric_pepoch = topocentric_pepoch.tdb + ltt_bary
    period =  0.04319854*u.day
    period_err = 0.0000136*u.day

    J1555 = Source(
        coord=j1555_coord,
        pos_error_a=j1555_pos_error_a,
        pos_error_b=j1555_pos_error_b,
        pos_error_pa=j1555_pos_error_pa,
        RM=RM,
        period=period,
        period_err=period_err,
        pepoch=barycentric_pepoch,
        pepoch_top=topocentric_pepoch,
        ltt_bary=ltt_bary,
    )

    return J1555


def compute_phase(
    t_abs: Time,
    t0: Time,
    period: u.Quantity,
    phi0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    cycles = ((t_abs - t0) / period).decompose() + phi0
    phase = cycles - np.floor(cycles)

    return phase, cycles

def plot_rows(
    ax,
    ts,
    source,
    phase_errors,
    phase,
    flux,
    rows,
) -> dict:
    """
    Plot an observation that may span multiple rows.
    For each contiguous run of equal row index,
    draw a single line segment at vertical position (flux - row_step * row_index).
    """

    # Sample colourmap
    cmap = plt.get_cmap("inferno")
    norm = Normalize(vmin=500, vmax=3000)
    color = cmap(norm(1000))

    # Walk through contiguous chunks of constant row index
    breaks = list(np.flatnonzero(rows[1:] != rows[:-1]) + 1)

    step_size = 10
    starts = [0, *breaks]
    stops = [*breaks, len(rows)]
    for start, stop in zip(starts, stops):
        # Slice this cycle of flux and offset by row count
        row = rows[start]
        ytrace = flux[start:stop] - step_size * row

        ax.plot(
            phase[start:stop],
            ytrace,
            color=color,
            alpha=0.5,
        )

        # Calculate phase error for this row
        time_err = source.period_err / source.period * (ts[start] - source.pepoch)
        phase_err = (time_err / source.period).to(1)
# Was originally
#         time_err = source.period_err / source.period * (obs.t_abs[start] - source.pepoch)
#        phase_err = (time_err / source.period).to(1)

            # Accumulate dictionary of pulse phase uncertainty ranges
        phase_errors["rows"].append(-row * step_size)
        phase_errors["mins"].append(0.5 - phase_err)
        phase_errors["maxs"].append(0.5 + phase_err)
        #except(IndexError):
        #    pass

    return phase_errors

def plot_folded_lightcurves(
    ax,
    ts,
    lcs,
    source
):
    """
    Make a folded lightcurve plot obeying the rules:
      - phase(ts) = ((ts - t0)/P + phi0) % 1
      - within an observation: when phase reaches 1, wrap to next row
      - each new observation begins on the next row at its intrinsic starting phase
    """

    next_base_row = 0

    # Collect one y-tick per observation at its first (top) row
    ytick_pos = []
    ytick_lab = []

    phase_errors = {
        "rows": [],
        "mins": [],
        "maxs": [],
    }

    for t, lc in zip(ts, lcs):
        row_step = 10

        flux = lc

        # Convert times to phase and cycle count
        phase, cycles = compute_phase(
            t,
            t0=source.pepoch,
            period=source.period,
            phi0=0.5,
        )

        # Row indices for this observation, starting at 'next_base_row'
        base_row = next_base_row
        row_indices = base_row + (np.floor(cycles) - np.floor(cycles[0])).astype(int)

        # Add a y-tick at the top row of this observation, labeled by the obs start date
        ytick_pos.append(-row_step * base_row)
        # TODO FIX
        ytick_lab.append(t[0].isot.split("T")[0])

        # Draw and accumulate phase errors
        phase_errors = plot_rows(
            ax,
            t,
            source,
            phase_errors=phase_errors,
            phase=phase,
            flux=flux,
            rows=row_indices,
        )

        # Update base row for the next observation: last row used + 1
        next_base_row = row_indices.max() + 1

    # Plot phase uncertainty band
    try:
        ax.fill_betweenx(
            phase_errors["rows"],
            phase_errors["mins"],
            phase_errors["maxs"],
            color="g",
            alpha=0.1,
        )
    except(ValueError):
        pass

    # Axis limits
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Phase")

    # Apply y-ticks at the top row of each observation
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lab)

    return ax


# TODO not sure about this click thing
@click.command()
def main():
    #setupLogger(verbose=True)

    J1555 = get_source()

    # Set DS filepaths
    dynspecs = sorted(glob.glob("./dynspec/science*.pkl"))
#    sbids = []
#    for dsfile in dynspecs:
#        sbid = int(dsfile.split("SB")[1][0:5].replace("_",""))
    sbids = np.unique(np.array([int(dsfile.split("SB")[1][0:5].replace("_","")) for dsfile in dynspecs]))
    for sbid in sbids:
        dynspecs = glob.glob(f"./dynspec/*SB{sbid}*pkl")
        ds_example = np.load(dynspecs[0], allow_pickle=True)
        final_array = np.empty((ds_example["DS"].shape[0],ds_example["DS"].shape[1],ds_example["DS"].shape[2],len(dynspecs)))
        for i in range(0, len(dynspecs)):
            dsfile = dynspecs[i]
            arr = np.load(dsfile, allow_pickle=True)
            final_array[:,:,:,i] = arr["DS"]
# TODO: RFI flagging, primary beam correction
        ds_example["DS"] = np.nanmean(final_array, axis=3)
        with open(f'./averaged_dynspec/SB{sbid:05d}.pkl', 'wb') as file:
            pickle.dump(ds_example, file)
    dynspecs = sorted(glob.glob("./averaged_dynspec/*.pkl"))
#    data_root = Path("/Users/pri239/astro/papers/j1424-6126/analysis/data")
#    vast_ds1 = glob.glob(f"{data_root}/vast/set1/*ds")
#    vast_ds2 = glob.glob(f"{data_root}/vast/set2/*ds")
#    atca_ds = glob.glob(f"{data_root}/atca/*.ds")
#    emu_ds = glob.glob(f"{data_root}/emu/*ds")
#    mkt_ds = glob.glob(f"{data_root}/mkt/*ds")

# We will have to work this out later -- for now just do one massive panel
    # Partition archival VAST / RACS in first two panels
#    panel1_obs = [Observation(path=p, source=J1424) for p in vast_ds1]
#    panel2_obs = [Observation(path=p, source=J1424) for p in vast_ds2]

    # EMU and followup observations in final panel
#    emu_obs = [Observation(path=p, source=J1424) for p in emu_ds]
#    atca_obs = [Observation(path=p, source=J1424) for p in atca_ds]
#    mkt_obs = [Observation(path=p, source=J1424) for p in mkt_ds]
#    panel3_obs = [obs for obs in (atca_obs + emu_obs + mkt_obs)]

    ts = []
    lcs = []
    for pkl in dynspecs:
        t, l = make_light_curve(pkl, J1555.coord)
        ts.append(t)
        lcs.append(l)

# One column of A4
    fig = plt.figure(figsize=(8.5*cm, 25*cm))
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
#    ax_vast2 = fig.add_subplot(gs[0, 1])
#    ax_others = fig.add_subplot(gs[0, 2])

    # Plot panels
# Commenting for now to try to get things working?!
    ax = plot_folded_lightcurves(
        ax,
        ts,
        lcs,
        source=J1555,
    )

#    ax_vast2 = plot_folded_lightcurves(
#        ax_vast2,
#        sorted(panel2_obs, key=lambda x: x.t_start),
#        source=J1424,
#    )
#    ax_others = plot_folded_lightcurves(
#        ax_others,
#        sorted(panel3_obs, key=lambda x: x.t_start),
#        source=J1424,
#    )

    fig.tight_layout()

    ymin, ymax = ax.get_ylim()
# Decrease padding
    ax.set_ylim(ymin+60, ymax-60)
    print( ax.get_ylim())
    fig.savefig("observations_stacked.png", format="png")
    fig.savefig("observations_stacked.pdf", format="pdf")

#    plt.show()


if __name__ == "__main__":
    main()
