import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import os
 
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
from matplotlib.lines import Line2D

from primary_beams import GaussianPB, get_beam_pos, MKCosBeam, get_beam_pos_mkt

#logger = logging.getLogger(__name__)
cm = 1/2.54

# Spectral index to use for correcting to same flux density scale
SPECIND = -1.5
# Which frequency to correct to
CALFREQ = 1.e9 # (1 GHz)

# If you've run this already, you can make these False and save some time
ASKAPPBCorr = False
MKTPBCorr = False

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
    ds = np.load(dsfile, allow_pickle=True)
    stem = Path(dsfile).stem
# Barycentre times
    telescope = ds["TELESCOPE"]
    if telescope == "ASKAP":
        loc = EarthLocation.of_site('mwa')
    elif telescope == "MeerKAT":
        loc = EarthLocation.of_site('salt')
    else:
        loc = EarthLocation.of_site(telescope)
    times = Time(ds["TIMES"]/(24*3600), scale='utc', format='mjd', location=loc)
    tstart = times[0].mjd
    bary_tt = times.light_travel_time(coords, kind="barycentric")
# One of the MeerKAT observations has an airplane fly through the first 50 seconds
    if stem == "CB1716830412":
        ds["DS"][0:50,:,:] = np.nan
# Also Scott didn't flag the RFI for some reason
        ds["DS"][:,315:535,:] = np.nan
# This one too!!
    if stem == "CB1636329974":
        ds["DS"][0:25,:,:] = np.nan
# and in this one, it flies right through the middle!! -- we had to remove it completely
#    if stem == "CB1628607085":
#        ds["DS"][int(len(times)/3):2*int(len(times)/3),:,:] = np.nan

# Do some RFI flagging
    It = np.real(ds["DS"][:,:,0]+ds["DS"][:,:,3])/2
    Qt = np.real((ds["DS"][:,:,0]-ds["DS"][:,:,3]))/2
# This one has no Stokes Q
    if stem == "CB1716830412":
        spec_std = np.nanstd(It, axis=0)
    else:
        spec_std = np.nanstd(Qt, axis=0)
    xrange = np.arange(0,len(spec_std))
    deg = 3
    p = np.polynomial.Polynomial.fit(xrange[~np.isnan(spec_std)], spec_std[~np.isnan(spec_std)], deg=deg)
# Two rounds of sigma-clipping
    std_spec_std = np.nanstd(spec_std-p(xrange))
    ind = np.logical_and(~np.isnan(spec_std),np.abs(spec_std - p(xrange))<std_spec_std)
    p = np.polynomial.Polynomial.fit(xrange[ind], spec_std[ind], deg=deg)
    std_spec_std = np.nanstd(spec_std-p(xrange))
    ind = np.logical_and(~np.isnan(spec_std),np.abs(spec_std - p(xrange))<std_spec_std)
    p = np.polynomial.Polynomial.fit(xrange[ind], spec_std[ind], deg=deg)
    med_spec_std = np.nanmedian(spec_std-p(xrange))
    std_spec_std = np.nanstd(spec_std-p(xrange))

# Sanity plot for RFI flagging
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(spec_std-p(xrange))
    ax.set_ylabel("standard deviation - poly offset")
    ax.set_xlabel("channel index")
    ax.axhline(med_spec_std+1.5*std_spec_std, color='red')
    fig.savefig(f'sanity_check_{stem}.png', bbox_inches="tight")

    lc = np.nanmean(It[:,(spec_std-p(xrange))<(med_spec_std+1.5*std_spec_std)], axis=1)
# Set to NaN any exactly zero values
    lc[lc==0.0] = np.nan

# Remove any time-dependent slow variation (esp. MeerKAT data)
    if telescope == "MeerKAT" and stem[0:12]=="CB1652551867":
# It's the discovery observation and we need to do a lot of baseline removal
        print(f"derippling (long) {dsfile}")
        lc = deripple_long(times.mjd*24*3600, lc)
    else:
        print(f"derippling {dsfile}")
        lc = deripple_short(times.mjd*24*3600, lc)

# Convert times to barycentred and light curve to mJy
    return times.tdb + bary_tt, 1000*lc, ds["FREQS"][int(len(ds["FREQS"])/2)], tstart

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
    topocentric_pepoch = Time(59713.42655646942, format='mjd', scale="utc", location=askap_loc)
    ltt_bary = topocentric_pepoch.light_travel_time(j1555_coord, kind="barycentric")
    barycentric_pepoch = topocentric_pepoch.tdb + ltt_bary
    period = (62.205893 / (24*60))*u.day
    period_err = 0.0000136*u.day

# Most up to date, from the paper
#T0 = 59713.42655646942
#P = 62.205893 / (2*24*60)



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

    # Setting a scale factor so that the plots look nice
    scale = 0.75

    # Walk through contiguous chunks of constant row index
    breaks = list(np.flatnonzero(rows[1:] != rows[:-1]) + 1)

    step_size = 10
    starts = [0, *breaks]
    stops = [*breaks, len(rows)]
    for start, stop in zip(starts, stops):
        # Slice this cycle of flux and offset by row count
        row = rows[start]
        ytrace = scale*flux[start:stop] - step_size * row

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
            #phi0=0,

        # Row indices for this observation, starting at 'next_base_row'
        base_row = next_base_row
        row_indices = base_row + (np.floor(cycles) - np.floor(cycles[0])).astype(int)

        # Add a y-tick at the top row of this observation, labeled by the obs start date
        ytick_pos.append(-row_step * base_row)
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
# This replaces the phase labels -- USE WITH CAUTION -- MAKE SURE YOU CHANGE phi0
#    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = ['0.5', '0.75', '0.0', '0.25', '0.5']
    ax.set_xticklabels(labels)

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

    cutoff = 0.003 # Jy

    # Segment 1
    deg = 3

    t = times_m[:seg1_end]
    y = ilc_m[:seg1_end]
    p = np.polynomial.Polynomial.fit(t, y, deg=deg)
    y_smooth = p(t)
    # Sigma-clip
    y_fit = y[np.abs(y - p(t))<cutoff]
    t_fit = t[np.abs(y - p(t))<cutoff]
    p = np.polynomial.Polynomial.fit(t_fit,y_fit, deg=deg)
    y_smooth = p(t)

    ilc_m[:seg1_end] = y - y_smooth

    # Segment 2
    t = times_m[seg1_end:seg2_end]
    y = ilc_m[seg1_end:seg2_end]
    p = np.polynomial.Polynomial.fit(t, y, deg=deg)
    y_smooth = p(t)
    # Sigma-clip
    y_fit = y[np.abs(y - p(t))<cutoff]
    t_fit = t[np.abs(y - p(t))<cutoff]
    p = np.polynomial.Polynomial.fit(t_fit, y_fit, deg=deg)
    y_smooth = p(t)

    ilc_m[seg1_end:seg2_end] = y - y_smooth


    # Segment 3
    t = times_m[seg2_end:seg3_end]
    y = ilc_m[seg2_end:seg3_end]
    p = np.polynomial.Polynomial.fit(t, y, deg=deg)
    y_smooth = p(t)

    # Sigma-clip
    y_fit = y[np.abs(y - p(t))<cutoff]
    t_fit = t[np.abs(y - p(t))<cutoff]
    p = np.polynomial.Polynomial.fit(t_fit, y_fit, deg=deg)
    y_smooth = p(t)

    ilc_m[seg2_end:seg3_end] = y - y_smooth

    # Segment 4
    t = times_m[seg3_end:]
    y = ilc_m[seg3_end:]
    p = np.polynomial.Polynomial.fit(t, y, deg=deg)
    y_smooth = p(t)
    # Sigma-clip
    y_fit = y[np.abs(y - p(t))<cutoff]
    t_fit = t[np.abs(y - p(t))<cutoff]
    p = np.polynomial.Polynomial.fit(t_fit, y_fit, deg=deg)
    y_smooth = p(t)
    ilc_m[seg3_end:] = y - y_smooth

    return ilc_m

def justdate(t = None):
    ''' take a Time object and return just the date '''
    return "{0}-{1:02.0f}-{2:02.0f}".format(t.ymdhms[0], t.ymdhms[1], t.ymdhms[2])

def hhmm(t = None):
    ''' take a Time object and return just the time with no seconds '''
    return "{0:02.0f}:{1:02.0f}".format(t.ymdhms[3], t.ymdhms[4])


@click.command()
def main():
    #setupLogger(verbose=True)

    J1555 = get_source()

    # ASKAP data
    # Pre-processing includes averaging the two beams together, and applying the primary beam correction
    if ASKAPPBCorr is True:
        dynspecs = sorted(glob.glob("./dynspec/science*.pkl"))
        sbids = np.unique(np.array([int(dsfile.split("SB")[1][0:5].replace("_","")) for dsfile in dynspecs]))
        for sbid in sbids:
            outname = f'./averaged_dynspec/SB{sbid:05d}.pkl'
            if not os.path.exists(outname):
                print(f"Correcting {sbid}")
                dynspecs = glob.glob(f"./dynspec/*SB{sbid}*pkl")
                ds_example = np.load(dynspecs[0], allow_pickle=True)
                final_array = np.empty((ds_example["DS"].shape[0],ds_example["DS"].shape[1],ds_example["DS"].shape[2],len(dynspecs)))
                weights_array = np.empty((ds_example["DS"].shape[1],len(dynspecs)))
                for i in range(0, len(dynspecs)):
                    dsfile = dynspecs[i]
                    ds = np.load(dsfile, allow_pickle=True)
                    beam = dsfile.split("beam")[1][0:2]
        # Survey is always after the SBID like:
        #scienceData.VAST_1552-56.SB81799.VAST_1552-56.beam01_averaged_cal.leakage.pkl
        #scienceData_SB8676_RACS_1552-56A.beam01_averaged_cal.pkl
        # But sometimes it's an underscore and sometimes a full stop -- so replace at the critical point
                    survey = dsfile.split("SB")[1].replace("_", ".").split(".")[1]
        # Apply frequency-dependent primary beam 
                    pb_vals = []
    # TODO improve efficiency
                    for freq in ds["FREQS"]:
                        pb = GaussianPB(frequency = freq*1.e9)
                        sep = get_beam_pos(survey, beam).separation(J1555.coord)
                        pb_vals.append(pb.evaluate(sep.rad, freq=freq))
            # This is just frequencies for each dynamic spectrum
                    weights_array[:,i] = np.array(pb_vals) 
                    final_array[:,:,:,i] = ds["DS"]
        # Correcting for the primary beam means multiplying by the primary beam correction (which is inversely proportional to distance to the phase centre)
        # Weighting by the primary beam means dividing by the primary beam correction (as big numbers are bad!)
        # So, effectively, those terms cancel out, and at the end we want to simply divide by the sum of the weights

                ds_example["DS"][:,:,:] = np.nansum(final_array, axis=3) / np.nansum(np.tile(weights_array[None, :, None, :], (ds_example["DS"].shape[0], 1, 4, 1)), axis=3)
# SBID 40625 was run through FixMS but all of the other SBIDs were not, so they need to be multiplied by a factor of 2 to match other telescope Stokes conventions (note all the PAs will still be wrong, but there is only noise in these data, so it's fine)
                if sbid != 40625:
                    ds_example["DS"] *= 2
                with open(f'./averaged_dynspec/SB{sbid:05d}.pkl', 'wb') as file:
                    pickle.dump(ds_example, file)

    dynspecs = sorted(glob.glob("./averaged_dynspec/SB*.pkl"))
    sbids = np.array([int(Path(dsfile).stem.split("SB")[1]) for dsfile in dynspecs])

    # Preprocessing means applying the primary beam
    dynspecs_m = glob.glob("./dynspec/1*pkl")
    
    if MKTPBCorr is True:
        for dsfile in dynspecs_m:
            ds = np.load(dsfile, allow_pickle=True)
            cbid = int(Path(dsfile).stem[0:10])
            print(cbid)
             
            pb_vals = []
    # TODO improve efficiency
            for freq in ds["FREQS"]:
                pb_vals.append(MKCosBeam(get_beam_pos_mkt(cbid).separation(J1555.coord).deg, freq))
            pb_vals = np.array(pb_vals) 
# This is the one that Scott calibrated, which only has Stokes I already
#            if cbid == 1716830412:
#                ds["DS"] /= np.tile(pb_vals[None, :, None], (ds["DS"].shape[0], 1, 1))
#                print(ds["DS"].shape)
# Just duplicate the Stokes I (which is currently occupying 'XX') to XY, YX, YY
#                ds["DS"] = np.tile(ds["DS"], (1, 1, 4))
#                print(ds["DS"].shape)
#
#            else:
            ds["DS"] /= np.tile(pb_vals[None, :, None], (ds["DS"].shape[0], 1, 4))
            with open(f'./averaged_dynspec/CB{cbid:010d}.pkl', 'wb') as file:
                pickle.dump(ds, file)

    dynspecs_m = sorted(glob.glob("./averaged_dynspec/CB*pkl"))
    cbids = np.array([int(Path(dsfile).stem.split("CB")[1]) for dsfile in dynspecs_m])

    tstarts = []
    obslengths = []
    ts = []
    lcs = []
    maxs = []
    rmss = []
    colors = []
    freqs = []
    for pkl in dynspecs:
        print(f"Making light curve from {pkl}")
        t, l, fc, tstart = make_light_curve(pkl, J1555.coord)
        tstarts.append(tstart)
        obslengths.append(24*60*(t[-1].mjd - t[0].mjd))
        ts.append(t)
        lcs.append(l)
        maxs.append(np.nanmax(l))
        rmss.append(np.nanstd(l))
        freqs.append(fc)
        colors.append('black')

    for pkl in dynspecs_m:
        stem = Path(pkl).stem
        print(f"Making light curve from {pkl}")
        t, l, fc, tstart = make_light_curve(pkl, J1555.coord)
        tstarts.append(tstart)
        obslengths.append(24*60*(t[-1].mjd - t[0].mjd))
        ts.append(t)
        maxs.append(np.nanmax(l))
        rmss.append(np.nanstd(l))
        freqs.append(fc)
#        if stem == "CB1716830412":
# This one has such high noise that it blows up the light curve plot, so bring it down just for plotting
#            lcs.append(l/100)
#        else:
        lcs.append(l)
        colors.append('purple')

    ids = np.concatenate([sbids, cbids])
# Sort all data by the start time
    ts = [t for _, t in sorted(zip(tstarts, ts))]
    lcs = [lc for _, lc in sorted(zip(tstarts, lcs))]
    colors = [color for _, color in sorted(zip(tstarts, colors))]
    maxs = [m for _, m in sorted(zip(tstarts, maxs))]
    rmss = [rms for _, rms in sorted(zip(tstarts, rmss))]
    freqs = [f for _, f in sorted(zip(tstarts, freqs))]
    ids = [i for _, i in sorted(zip(tstarts, ids))]
    obslengths = [o for _, o in sorted(zip(tstarts, obslengths))]
# And LAST sort the start times
    tstarts = sorted(tstarts)

# Stacked light curve plot
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
        ts[0:18],
        lcs[0:18],
        colors[0:18],
        source=J1555
    )
# The rest are short
    ax2 = plot_folded_lightcurves(
        ax2,
        ts[18:35],
        lcs[18:35],
        colors[18:35],
        source=J1555
    )
    ax3 = plot_folded_lightcurves(
        ax3,
        ts[35:],
        lcs[35:],
        colors[35:],
        source=J1555
    )
    fig.tight_layout()

    for ax in [ax1, ax2, ax3]:
        ymin, ymax = ax.get_ylim()
# Decrease padding (no idea why it is so large)
        ax.set_ylim(ymin+20, ymax-20)
    fig.savefig("observations_stacked.png", format="png")
    fig.savefig("observations_stacked.pdf", format="pdf")

# Upper limits as a function of time
    fig = plt.figure(figsize=(17*cm,5*cm))
    ax = fig.add_subplot(111)
    for ts, rms, m, c, f, i in zip(tstarts, rmss, maxs, colors, freqs, ids):
# Skip that one point that is useless
        if rms < 50:
            if m / rms > 8:
                y = m*(CALFREQ/f)**(SPECIND)
                yr = rms*(CALFREQ/f)**(SPECIND)
                ax.scatter(ts, y, color=c, marker='*', s=8)
                ax.errorbar(ts, y, yerr=rms, color=c, elinewidth=0.5)
            else:
                yval = (3*rms)*(CALFREQ/f)**(SPECIND)
                ax.scatter(ts, yval, marker='v', color=c, s=8)
    ax.set_xlabel("MJD")
    ax.set_ylabel("Flux density (mJy)")
    legend_elements = [Line2D([0], [0], lw=0, markersize=4, markerfacecolor='none', markeredgecolor='k', marker='v', label='3-$\sigma$ upper limits'),
                       Line2D([0], [0], lw=0, markersize=4, markerfacecolor='none', markeredgecolor='k', marker='*', label='Detections\n(brightest pulse)')]
    ax.legend(loc=1, handles=legend_elements)
    fig.savefig("Archival_upper_limits.pdf", bbox_inches="tight")

# Save as a text file
    out = np.array([ids, tstarts, rmss, maxs, np.array(freqs)/1.e9, np.array(rmss)*(CALFREQ/np.array(freqs))**(SPECIND), np.array(maxs)*(CALFREQ/np.array(freqs))**SPECIND])
    np.savetxt("Archival_measurements.csv", out.T, fmt=['%s', '%5.8f', '%0.6f', '%0.6f', '%1.3f', '%0.6f', '%0.6f'], delimiter=',', header='#ID,MJD,RMS,MAX,F_GHz,RMS_scaled,MAX_scaled')

# Make a LaTeX table output
    times = Time(tstarts, format='mjd', scale='utc')
    datestrs = np.array([justdate(t) for t in times])
    hhmmstrs = np.array([hhmm(t) for t in times])
    freqs = np.array(freqs)/1.e6
    obslengths = np.array(obslengths)

    with open("obs_table.tex", "w") as f:
        for i in range(0, len(ids)):
            date, time, iid, freq, obslength = datestrs[i], hhmmstrs[i], ids[i], freqs[i], obslengths[i]
            if iid < 999999:
    # It's an ASKAP observation
                valid_pkls = sorted(glob.glob(f"./dynspec/*SB{iid}*pkl"))
                beams = []
                for dsfile in valid_pkls:
                    beams.append(dsfile.split("beam")[1][0:2])
                f.write(f"{date} & {time} & ASKAP & SB{iid} beams:")
                for j in range(0,len(beams)):
                    f.write(f"{beams[j]}")
                    if j < len(beams) - 1:
                        f.write(f",")
                f.write(f"& {freq:4.0f} & {obslength:2.0f} \\\\\n")
    # It's a MeerKAT observation
            else:
                f.write(f"{date} & {time} & MeerKAT & CB{iid} & {freq:4.0f} & {obslength:2.0f} \\\\\n")

if __name__ == "__main__":
    main()



