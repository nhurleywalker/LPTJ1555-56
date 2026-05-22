#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

import sys

XX = 0
XY = 1
YX = 2
YY = 3

# TODO, load both, take weighted average
arr = np.load("dynspec/scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_averaged_cal.leakage.pkl", allow_pickle=True)
arr2 = np.load("dynspec/scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam09_averaged_cal.leakage.pkl", allow_pickle=True)

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
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*lc, lw=0.5, color='darkblue', alpha=0.8)
ax.set_xlim(0, arr["TIMES"][-1] - arr["TIMES"][0])
fig.savefig("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_lightcurve.png", bbox_inches="tight")

fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.set_ylabel("brightness (mJy/beam)")
ax.set_xlabel("time / s")
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*qlc, lw=0.5, color='red', alpha=0.8, label="Stokes Q") 
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*ulc, lw=0.5, color='blue', alpha=0.8, label="Stokes U")
ax.set_xlim(0, arr["TIMES"][-1] - arr["TIMES"][0])
fig.savefig("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_QU_lightcurve.png", bbox_inches="tight")

fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.set_ylabel("brightness (mJy/beam)")
ax.set_xlabel("time / s")
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*vlc, lw=0.5, color='green', alpha=0.8, label="Stokes V")
ax.set_xlim(0, arr["TIMES"][-1] - arr["TIMES"][0])
ax.set_ylim(-20, 20)
ax.legend()
fig.savefig("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_V_lightcurve.png", bbox_inches="tight")

