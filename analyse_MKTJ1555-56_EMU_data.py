#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

import glob
import sys

T0 = 59713.512505
P = 0.02168 # days

def ephem(n):
    return 59713.512505 + n*P

def pulsenum(mjd):
    return (mjd - 59712.512505) / (n*P)

XX = 0
XY = 1
YX = 2
YY = 3

# TODO find positions in primary beams and take weighted average, correct for beam attenuation
arr = np.load("dynspec/scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_averaged_cal.leakage.pkl", allow_pickle=True)
arr2 = np.load("dynspec/scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam09_averaged_cal.leakage.pkl", allow_pickle=True)

# Sadly these didn't show the source at all (SNR much more problematic in Stokes I as well)
#arr = np.load("dynspec/scienceData.EMU_1554-55_band1.SB43773.EMU_1554-55_band1.beam09_averaged_cal.leakage.pkl", allow_pickle=True)
#arr2 = np.load("dynspec/scienceData.EMU_1554-55_band1.SB43773.EMU_1554-55_band1.beam15_averaged_cal.leakage.pkl", allow_pickle=True)
#arr = np.load("dynspec/scienceData.EMU_1554-55.SB33284.EMU_1554-55.beam09_averaged_cal.leakage.pkl", allow_pickle=True)
#arr2 = np.load("dynspec/scienceData.EMU_1554-55.SB33284.EMU_1554-55.beam15_averaged_cal.leakage.pkl", allow_pickle=True)

# But I think Q and V are swapped
It = (np.real((arr["DS"][:,:,XX]+arr["DS"][:,:,YY])) + np.real((arr2["DS"][:,:,XX]+arr2["DS"][:,:,YY]))) / 2
Qt = (np.real((arr["DS"][:,:,XX]-arr["DS"][:,:,YY])) + np.real((arr2["DS"][:,:,XX]-arr2["DS"][:,:,YY]))) / 2
Ut = (np.real((arr["DS"][:,:,XY]+arr["DS"][:,:,YX])) + np.real((arr2["DS"][:,:,XY]+arr2["DS"][:,:,YX]))) / 2
Vt = (np.imag((arr["DS"][:,:,XY]-arr["DS"][:,:,YX])) + np.imag((arr2["DS"][:,:,XY]-arr2["DS"][:,:,YX]))) / 2

# RFI flagging
It[:,124] = np.nan
Qt[:,124] = np.nan
Ut[:,124] = np.nan
Vt[:,124] = np.nan

It[:,0:16] = np.nan
Qt[:,0:16] = np.nan
Ut[:,0:16] = np.nan
Vt[:,0:16] = np.nan

# Form individual Stokes dynamic spectra
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.imshow(It.T, origin='lower',vmin = -0.005, vmax=0.03, aspect='auto', extent=[0, (arr["TIMES"][-1]-arr["TIMES"][0]), arr["FREQS"][0]/1.e9, arr["FREQS"][-1]/1.e9] )
ax.set_xlabel("time / s")
ax.set_ylabel("frequency / GHz")
fig.savefig("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_StokesI_dynspec.png", bbox_inches="tight")

fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.imshow(Qt.T, origin='lower',vmin = -0.005, vmax=0.03, aspect='auto', extent=[0, (arr["TIMES"][-1]-arr["TIMES"][0]), arr["FREQS"][0]/1.e9, arr["FREQS"][-1]/1.e9] )
ax.set_xlabel("time / s")
ax.set_ylabel("frequency / GHz")
fig.savefig("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_StokesQ_dynspec.png", bbox_inches="tight")

fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.imshow(Ut.T, origin='lower',vmin = -0.005, vmax=0.03, aspect='auto', extent=[0, (arr["TIMES"][-1]-arr["TIMES"][0]), arr["FREQS"][0]/1.e9, arr["FREQS"][-1]/1.e9] )
ax.set_xlabel("time / s")
ax.set_ylabel("frequency / GHz")
fig.savefig("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_StokesU_dynspec.png", bbox_inches="tight")

fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.imshow(Vt.T, origin='lower',vmin = -0.005, vmax=0.03, aspect='auto', extent=[0, (arr["TIMES"][-1]-arr["TIMES"][0]), arr["FREQS"][0]/1.e9, arr["FREQS"][-1]/1.e9] )
ax.set_xlabel("time / s")
ax.set_ylabel("frequency / GHz")
fig.savefig("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_StokesV_dynspec.png", bbox_inches="tight")

# Form light curves
lc = np.nanmean(It, axis=1)
qlc = np.nanmean(Vt, axis=1)
ulc = np.nanmean(Ut, axis=1)
vlc = np.nanmean(Qt, axis=1)

fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.set_ylabel("brightness (mJy/beam)")
ax.set_xlabel("time / s")
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*lc, lw=0.5, color='black', alpha=0.8, label="Stokes I")
for i in range(-3, 20):
    ax.axvline(ephem(i)*24*3600 - arr["TIMES"][0], alpha=0.4, color='orange')
ax.set_xlim(0, arr["TIMES"][-1] - arr["TIMES"][0])
ax.legend(loc=1)
fig.savefig("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_lightcurve.png", bbox_inches="tight")

fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.set_ylabel("brightness (mJy/beam)")
ax.set_xlabel("time / s")
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*qlc, lw=0.5, color='red', alpha=0.8, label="Stokes Q") 
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*ulc, lw=0.5, color='blue', alpha=0.8, label="Stokes U")
for i in range(-3, 20):
    ax.axvline(ephem(i)*24*3600 - arr["TIMES"][0], alpha=0.4, color='orange')
ax.set_xlim(0, arr["TIMES"][-1] - arr["TIMES"][0])
ax.legend(loc=1)
fig.savefig("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_QU_lightcurve.png", bbox_inches="tight")

fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.set_ylabel("brightness (mJy/beam)")
ax.set_xlabel("time / s")
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*vlc, lw=0.5, color='green', alpha=0.8, label="Stokes V")
for i in range(-3, 20):
    ax.axvline(ephem(i)*24*3600 - arr["TIMES"][0], alpha=0.4, color='orange')
ax.set_xlim(0, arr["TIMES"][-1] - arr["TIMES"][0])
ax.set_ylim(-20, 20)
ax.legend(loc=1)
fig.savefig("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_V_lightcurve.png", bbox_inches="tight")

# Fold the data
# Put these into MJD (instead of MJD seconds)
trange = arr["TIMES"] / (24*3600)

phase = np.mod(trange, 2*P)/(2*P)
idx = np.argsort(phase)

num_bins = 150

bin_edges = np.linspace(0, 1, num_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_indices = np.digitize(phase[idx], bin_edges) - 1

# Add 'phase' and 'bin_idx' to DataFrame
#df['phase'] = phase
#df['bin_idx'] = bin_indices

I_sums = np.bincount(bin_indices, weights=1000*lc[idx], minlength=num_bins)
Q_sums = np.bincount(bin_indices, weights=1000*qlc[idx], minlength=num_bins)
U_sums = np.bincount(bin_indices, weights=1000*ulc[idx], minlength=num_bins)
V_sums = np.bincount(bin_indices, weights=1000*vlc[idx], minlength=num_bins)

counts = np.bincount(bin_indices, minlength=num_bins)

# Prevent division by zero if a bin is empty
I_binned_average = I_sums / np.where(counts == 0, 1, counts)
Q_binned_average = Q_sums / np.where(counts == 0, 1, counts)
U_binned_average = U_sums / np.where(counts == 0, 1, counts)
V_binned_average = V_sums / np.where(counts == 0, 1, counts)

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.plot(bin_centers, I_binned_average, color='black', alpha=0.8, lw=0.5, label="Stokes I")
ax.plot(bin_centers, Q_binned_average, color='red', alpha=0.8, lw=0.5, label="Stokes Q")
ax.plot(bin_centers, U_binned_average, color='blue', alpha=0.8, lw=0.5, label="Stokes U")
ax.plot(bin_centers, V_binned_average, color='green', alpha=0.8, lw=0.5, label="Stokes V")
ax.set_xlabel("Phase")
ax.set_ylabel("Mean brightness (mJy)")
ax.legend(loc=1)
fig.savefig("Folded_light_curve.png", bbox_inches="tight")

# Load the rest of the .pkl files and calculate RMS and time for non-detections plot

pkls = sorted(glob.glob("dynspec/*RACS*pkl") + glob.glob("dynspec/*VAST*pkl") + glob.glob("dynspec/*SB43773*pkl") + glob.glob("dynspec/*SB33284*pkl"))

mjds = []
rmss = []
for pkl in pkls:
    arr = np.load(pkl, allow_pickle=True)
    mjds.append(arr["TIMES"][int(len(arr["TIMES"])/2)]/(24*3600))
    rmss.append(np.nanstd(np.nanmean(np.real((arr["DS"][:,:,XX]+arr["DS"][:,:,YY])),axis=1)))

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.scatter(mjds, 1000*np.array(rmss), marker='v', color='black', label='1-sigma RMS\n(10s time resolution)')
ax.set_xlabel("MJD")
ax.set_ylabel("Flux density (mJy)")
ax.scatter(T0, 17., color='red', marker='*', label='EMU Pilot detection\n(brightest pulse)')
ax.errorbar(T0, 17., yerr=1, color='red')
ax.legend(loc=1)
fig.savefig("Non-detections_ASKAP.png", bbox_inches="tight")

