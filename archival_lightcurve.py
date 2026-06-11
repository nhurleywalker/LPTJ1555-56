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
    "ytick.labelsize": 7,
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
    elif telescope == "MeerKAT":
        loc = EarthLocation.of_site('salt')
    else:
        loc = EarthLocation.of_site(telescope)
    times = Time(arr["TIMES"]/(24*3600), scale='utc', format='mjd', location=loc)
    bary_tt = times.light_travel_time(coords, kind="barycentric")

# Do some RFI flagging
    It = np.real(arr["DS"][:,:,0]+arr["DS"][:,:,3])/2
    Qt = np.real((arr["DS"][:,:,0]-arr["DS"][:,:,3]))/2
    spec_std = np.nanstd(Qt, axis=0)
    xrange = np.arange(0,len(spec_std))
    deg = 3
    p = np.polynomial.Polynomial.fit(xrange[~np.isnan(spec_std)], spec_std[~np.isnan(spec_std)], deg=deg)
# One round of sigma-clipping
    std_spec_std = np.nanstd(spec_std-p(xrange))
    ind = np.logical_and(~np.isnan(spec_std),np.abs(spec_std - p(xrange))<2*std_spec_std)
    p = np.polynomial.Polynomial.fit(xrange[ind], spec_std[ind], deg=deg)
    med_spec_std = np.nanmedian(spec_std-p(xrange))
    std_spec_std = np.nanstd(spec_std-p(xrange))

# Sanity plot for RFI flagging
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.plot(spec_std-p(xrange))
#    ax.set_ylabel("standard deviation - poly offset")
#    ax.set_xlabel("channel index")
#    ax.axhline(med_spec_std+3*std_spec_std, color='red')
#    stem = Path(dsfile).stem
#    fig.savefig(f'sanity_check_{stem}.png', bbox_inches="tight")

    lc = np.nanmean(It[:,(spec_std-p(xrange))<(med_spec_std+3*std_spec_std)], axis=1)

# Remove any time-dependent slow variation (esp. MeerKAT data)
    if telescope == "MeerKAT" and len(times)>100:
# It's the discovery observation and we need to do a lot of baseline removal
        print(f"derippling {dsfile}")
        lc = deripple_long(times.mjd*24*3600, lc)
    else:
        print(f"derippling {dsfile}")
        lc = deripple_short(times.mjd*24*3600, lc)

# Convert times to barycentred and light curve to mJy
    return times.tdb + bary_tt, 1000*lc

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
    color,
    rows,
) -> dict:
    """
    Plot an observation that may span multiple rows.
    For each contiguous run of equal row index,
    draw a single line segment at vertical position (flux - row_step * row_index).
    """

    # Sample colourmap
#    cmap = plt.get_cmap("inferno")
#    norm = Normalize(vmin=500, vmax=3000)
#    color = cmap(norm(1000))

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
            lw=0.75,
            alpha=0.8,
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
    colors,
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

    for t, lc, color in zip(ts, lcs, colors):
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
            color=color,
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

def deripple_short(t, lc):
    deg = 3
    p = np.polynomial.Polynomial.fit(t[~np.isnan(lc)], lc[~np.isnan(lc)], deg=deg)
    return lc - p(t)

def deripple_long(times_m, ilc_m):
    # First break the data into four segments
    tdiff = times_m[1:] - times_m[0:-1]
    tbreak = np.where(np.abs(tdiff) > 50)[0]
    #(array([223, 447, 672]),)
    seg1_end = tbreak[0]+1
    seg2_end = tbreak[1]+1
    seg3_end = tbreak[2]+1

    # I noticed the first and last samples are bad in each scan, so we will flag those

    # Segment 1
    deg = 3
    vmin, vmax = -10, 30
    b = 3 # b for buffer

    t = times_m[:seg1_end]
    t_fit = times_m[b:seg1_end-b]
    y = ilc_m[:seg1_end]
    y_fit = ilc_m[b:seg1_end-b]
    p = np.polynomial.Polynomial.fit(t_fit, y_fit, deg=deg)
    y_smooth = p(t)

    ilc_m[:seg1_end] = y - y_smooth
    # And now flag the buffer
    ilc_m[0:b] = np.nan
    ilc_m[seg1_end-b:seg1_end] = np.nan

    # Segment 2
    deg = 3
    vmin, vmax = -30, 30
    b = 10 # b for buffer
    t = times_m[seg1_end:seg2_end]
    t_fit = times_m[seg1_end+b:seg2_end-b]
    y = ilc_m[seg1_end:seg2_end]
    y_fit = ilc_m[seg1_end+b:seg2_end-b]
    p = np.polynomial.Polynomial.fit(t_fit, y_fit, deg=deg)
    y_smooth = p(t)

    ilc_m[seg1_end:seg2_end] = y - y_smooth
    # And now flag the buffer
    ilc_m[seg1_end:seg1_end+b] = np.nan
    ilc_m[seg2_end-b:seg2_end] = np.nan

    # Segment 3
    deg = 6
    vmin, vmax = -30, 25
    b = 10 
    t = times_m[seg2_end:seg3_end]
    t_fit = times_m[seg2_end+b:seg3_end-b]
    y = ilc_m[seg2_end:seg3_end]
    y_fit = ilc_m[seg2_end+b:seg3_end-b]
    p = np.polynomial.Polynomial.fit(t_fit, y_fit, deg=deg)
    y_smooth = p(t)

    # In the case of this segment, there is a lot of gnarly RFI, and the pulse itself is quite bright, so do some sigma-clipping
    new_y_fit = y_fit[np.abs(y_fit - p(t_fit))<0.003]
    new_t_fit = t_fit[np.abs(y_fit - p(t_fit))<0.003]

    p = np.polynomial.Polynomial.fit(new_t_fit, new_y_fit, deg=deg)
    y_smooth = p(t)

    ilc_m[seg2_end:seg3_end] = y - y_smooth
    # And now flag the buffer
    ilc_m[seg2_end:seg2_end+b] = np.nan
    ilc_m[seg3_end-b:seg3_end] = np.nan

    # Segment 4
    vmin, vmax = -10, 60
    deg = 3
    b = 10
    t = times_m[seg3_end:]
    t_fit = times_m[seg3_end+b:-b]
    y = ilc_m[seg3_end:]
    y_fit = ilc_m[seg3_end+b:-b]
    p = np.polynomial.Polynomial.fit(t_fit, y_fit, deg=deg)
    y_smooth = p(t)

    ilc_m[seg3_end:] = y - y_smooth
    # And now flag the buffer
    ilc_m[seg3_end:seg3_end+b] = np.nan
    ilc_m[-b:] = np.nan

    return ilc_m

@click.command()
def main():
    #setupLogger(verbose=True)

    J1555 = get_source()

    # ASKAP data
    # Pre-processing includes averaging the two beams together 
    dynspecs = sorted(glob.glob("./dynspec/science*.pkl"))
    sbids = np.unique(np.array([int(dsfile.split("SB")[1][0:5].replace("_","")) for dsfile in dynspecs]))
    for sbid in sbids:
        dynspecs = glob.glob(f"./dynspec/*SB{sbid}*pkl")
        ds_example = np.load(dynspecs[0], allow_pickle=True)
        final_array = np.empty((ds_example["DS"].shape[0],ds_example["DS"].shape[1],ds_example["DS"].shape[2],len(dynspecs)))
        for i in range(0, len(dynspecs)):
            dsfile = dynspecs[i]
            arr = np.load(dsfile, allow_pickle=True)
            final_array[:,:,:,i] = arr["DS"]
        ds_example["DS"] = np.nanmean(final_array, axis=3)
        with open(f'./averaged_dynspec/SB{sbid:05d}.pkl', 'wb') as file:
            pickle.dump(ds_example, file)
    dynspecs = sorted(glob.glob("./averaged_dynspec/*.pkl"))


    # MeerKAT data -- currently just two pkls
    # Preprocessing includes removing the Stokes I ripple
    dynspecs_m = ["./dynspec/1652551867-sdp-l0_2026-05-22T14-51-05_zBI_nominbl.pkl", "./dynspec/1656147142-sdp-l0_2026-06-08T16-46-06_bOB.pkl"]

    tstarts = []
    ts = []
    lcs = []
    colors = []
    for pkl in dynspecs:
        t, l = make_light_curve(pkl, J1555.coord)
        tstarts.append(t[0])
        ts.append(t)
        lcs.append(l)
        colors.append('black')

    for pkl in dynspecs_m:
        t, l = make_light_curve(pkl, J1555.coord)
        tstarts.append(t[0])
        ts.append(t)
        lcs.append(l)
        colors.append('purple')

# Sort all data by the start time
    ts = [t for _, t in sorted(zip(tstarts, ts))]
    lcs = [lc for _, lc in sorted(zip(tstarts, lcs))]
    colors = [lc for _, lc in sorted(zip(tstarts, colors))]

# Full page width
    fig = plt.figure(figsize=(17*cm, 12.5*cm))
    gs = GridSpec(1, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Plot panels
# The first few observations are very long
    ax1 = plot_folded_lightcurves(
        ax1,
        ts[0:9],
        lcs[0:9],
        colors[0:9],
        source=J1555
    )
# The rest are short
    ax2 = plot_folded_lightcurves(
        ax2,
        ts[9:25],
        lcs[9:25],
        colors[9:25],
        source=J1555
    )
    ax3 = plot_folded_lightcurves(
        ax3,
        ts[25:],
        lcs[25:],
        colors[25:],
        source=J1555
    )
    fig.tight_layout()

    for ax in [ax1, ax2]:
        ymin, ymax = ax.get_ylim()
# Decrease padding (no idea why it is so large)
        ax.set_ylim(ymin+10, ymax-10)
    fig.savefig("observations_stacked.png", format="png")
    fig.savefig("observations_stacked.pdf", format="pdf")

#    plt.show()


if __name__ == "__main__":
    main()


