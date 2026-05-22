# coding: utf-8
import numpy as np
arr = np.load("scienceData.EMU_1554-55.SB74457.EMU_1554-55.beam09_averaged_cal.leakage.pkl", allow_pickle=True)
It = np.real((arr["DS"][:,:,XX]+arr["DS"][:,:,YY]))
Qt = np.real((arr["DS"][:,:,XX]-arr["DS"][:,:,YY]))
Ut = np.real((arr["DS"][:,:,XY]+arr["DS"][:,:,YX]))
Vt = np.imag((arr["DS"][:,:,XY]-arr["DS"][:,:,YX]))
XX = 0
XY = 1
YX = 2
YY = 3
It = np.real((arr["DS"][:,:,XX]+arr["DS"][:,:,YY]))
Qt = np.real((arr["DS"][:,:,XX]-arr["DS"][:,:,YY]))
Ut = np.real((arr["DS"][:,:,XY]+arr["DS"][:,:,YX]))
Vt = np.imag((arr["DS"][:,:,XY]-arr["DS"][:,:,YX]))
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 8))
lc = np.nanmean(It, axis=1)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)

ax.set_ylabel("brightness (Jy/beam)")

ax.set_xlabel("time / s")
ax.plot(arr["TIMES"] - arr["TIMES"][0], lc)
plt.show()
arr = np.load("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_averaged_cal.leakage.pkl", allow_pickle=True)
It = np.real((arr["DS"][:,:,XX]+arr["DS"][:,:,YY]))
Qt = np.real((arr["DS"][:,:,XX]-arr["DS"][:,:,YY]))
Ut = np.real((arr["DS"][:,:,XY]+arr["DS"][:,:,YX]))
Vt = np.imag((arr["DS"][:,:,XY]-arr["DS"][:,:,YX]))
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)

ax.set_ylabel("brightness (Jy/beam)")

ax.set_xlabel("time / s")
ax.plot(arr["TIMES"] - arr["TIMES"][0], lc)
lc = np.nanmean(It, axis=1)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)

ax.set_ylabel("brightness (Jy/beam)")

ax.set_xlabel("time / s")
ax.plot(arr["TIMES"] - arr["TIMES"][0], lc)
plt.show()
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.set_ylabel("brightness (mJy/beam)")
ax.set_xlabel("time / s")
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*lc, lw=0.5, color='k')
plt.show()
from astropy.time import Time
t0 = Time(arr["TIMES"][0], format='mjd', scale='utc')
t0.isot
t0
t0 = Time(arr["TIMES"][0]/(24*3600), format='mjd', scale='utc')
t0.isot
def ephem(n):
    return 59713.512505 + n*0.02168
ephem(100)
t0 = Time(ephem(1), format='mjd', scale='utc')
t0.isot
31.2/(60*24)
Time(ephem(20/0.021666666), format='mjd', scale='utc').isot
Time(ephem(19/0.021666666), format='mjd', scale='utc').isot
19/0.021666666
Time(ephem(900), format='mjd', scale='utc').isot
Time(ephem(880), format='mjd', scale='utc').isot
Time(ephem(890), format='mjd', scale='utc').isot
Time(ephem(885), format='mjd', scale='utc').isot
Time(ephem(884), format='mjd', scale='utc').isot
Time(ephem(883), format='mjd', scale='utc').isot
Time(ephem(882), format='mjd', scale='utc').isot
Time(ephem(882), format='mjd', scale='utc').gps
Time(ephem(-30), format='mjd', scale='utc').isot
Time(ephem(-60), format='mjd', scale='utc').isot
Time(ephem(-300), format='mjd', scale='utc').isot
Time(ephem(-800), format='mjd', scale='utc').isot
Time(ephem(-1200), format='mjd', scale='utc').isot
Time(ephem(-2000), format='mjd', scale='utc').isot
Time(ephem(-1800), format='mjd', scale='utc').isot
Time(ephem(-1500), format='mjd', scale='utc').isot
Time(ephem(-1600), format='mjd', scale='utc').isot
Time(ephem(-1550), format='mjd', scale='utc').isot
Time(ephem(-1565), format='mjd', scale='utc').isot
Time(ephem(-1551), format='mjd', scale='utc').isot
Time(ephem(-1555), format='mjd', scale='utc').isot
Time(ephem(-1554), format='mjd', scale='utc').isot
t0.isot
t0 = Time(arr["TIMES"][0]/(24*3600), format='mjd', scale='utc')
t0.isot
t0 = Time(arr["TIMES"][-1]/(24*3600), format='mjd', scale='utc')
t0.isot
Time(ephem(-4000), format='mjd', scale='utc').isot
Time(ephem(-5000), format='mjd', scale='utc').isot
Time(ephem(-5050), format='mjd', scale='utc').isot
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.set_ylabel("brightness (mJy/beam)")
ax.set_xlabel("time / s")
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*lc, lw=0.5, color='k')
plt.show()
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.set_ylabel("brightness (mJy/beam)")
ax.set_xlabel("time / s")
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*lc, lw=0.5, color='darkblue', alpha=0.8)
fig.savefig("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_lightcurve.png", bbox_inches="tight")
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.imshow(It.T, origin='lower',vmin = -0.01, vmax=0.05)
plt.show()
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.imshow(It.T, origin='lower',vmin = -0.01, vmax=0.03, aspect='auto')
plt.show()
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.imshow(It.T, origin='lower',vmin = -0.01, vmax=0.03, aspect='auto')
ax.axhline(124)
ax.axhline(16)
plt.show()
It.shape
It[:,0:16] = np.nan
It[:,124] = np.nan
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.imshow(It.T, origin='lower',vmin = -0.01, vmax=0.03, aspect='auto')
ax.axhline(124)
ax.axhline(16)
plt.show()
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.imshow(It.T, origin='lower',vmin = -0.01, vmax=0.03, aspect='auto')
plt.show()
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.imshow(It.T, origin='lower',vmin = -0.01, vmax=0.03, aspect='auto')
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.imshow(It.T, origin='lower',vmin = -0.01, vmax=0.03, aspect='auto', extent=[arr["TIMES"][0], arr["TIMES"][-1], arr["FREQS"][0], arr["FREQS"][-1]] )
plt.show()
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.imshow(It.T, origin='lower',vmin = -0.005, vmax=0.03, aspect='auto', extent=[0, (arr["TIMES"][-1]-arr["TIMES"][0])/(24*3600), arr["FREQS"][0], arr["FREQS"][-1]] )
ax.set_xlabel("time / s")
ax.set_ylabel("frequency / Hz")
plt.show()
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.imshow(It.T, origin='lower',vmin = -0.005, vmax=0.03, aspect='auto', extent=[0, (arr["TIMES"][-1]-arr["TIMES"][0])*(24*3600), arr["FREQS"][0], arr["FREQS"][-1]] )
ax.set_xlabel("time / s")
ax.set_ylabel("frequency / Hz")
plt.show()
arr["TIMES"][-1]
arr["TIMES"][-1]-arr["TIMES"][0]
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.imshow(It.T, origin='lower',vmin = -0.005, vmax=0.03, aspect='auto', extent=[0, (arr["TIMES"][-1]-arr["TIMES"][0])/3600, arr["FREQS"][0], arr["FREQS"][-1]] )
ax.set_xlabel("time / s")
ax.set_ylabel("frequency / Hz")
arr["TIMES"][-1]-arr["TIMES"][0]
plt.show()
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.imshow(It.T, origin='lower',vmin = -0.005, vmax=0.03, aspect='auto', extent=[0, (arr["TIMES"][-1]-arr["TIMES"][0]), arr["FREQS"][0], arr["FREQS"][-1]] )
ax.set_xlabel("time / s")
ax.set_ylabel("frequency / Hz")
plt.show()
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.imshow(It.T, origin='lower',vmin = -0.005, vmax=0.03, aspect='auto', extent=[0, (arr["TIMES"][-1]-arr["TIMES"][0]), arr["FREQS"][0]/1.e9, arr["FREQS"][-1]/1.e9] )
ax.set_xlabel("time / s")
ax.set_ylabel("frequency / GHz")
fig.savefig("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_StokesI_dynspec.png", bbox_inches="tight")
lc = np.nanmean(It, axis=1)
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.set_ylabel("brightness (mJy/beam)")
ax.set_xlabel("time / s")
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*lc, lw=0.5, color='darkblue', alpha=0.8)
fig.savefig("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_lightcurve.png", bbox_inches="tight")
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.set_ylabel("brightness (mJy/beam)")
ax.set_xlabel("time / s")
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*lc, lw=0.5, color='darkblue', alpha=0.8)
ax.set_xlim(0, arr["TIMES"][-1] - arr["TIMES"][0])
fig.savefig("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_lightcurve.png", bbox_inches="tight")
Qt[:,124] = np.nan
Ut[:,124] = np.nan
Vt[:,124] = np.nan
Qt[:,124] = np.nan
Ut[:,124] = np.nan
Vt[:,124] = np.nan
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
Qt[:,0:16] = np.nan
Ut[:,0:16] = np.nan
Vt[:,0:16] = np.nan
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
qlc = np.nanmean(Vt, axis=1)
ulc = np.nanmean(Ut, axis=1)
vlc = np.nanmean(Qt, axis=1)
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.set_ylabel("brightness (mJy/beam)")
ax.set_xlabel("time / s")
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*qlc, lw=0.5, color='red', alpha=0.8, label="Q") 
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*ulc, lw=0.5, color='blue', alpha=0.8, label="U")
ax.set_xlim(0, arr["TIMES"][-1] - arr["TIMES"][0])
fig.savefig("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_QU_lightcurve.png", bbox_inches="tight")
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.set_ylabel("brightness (mJy/beam)")
ax.set_xlabel("time / s")
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*vlc, lw=0.5, color='green', alpha=0.8)
ax.set_xlim(0, arr["TIMES"][-1] - arr["TIMES"][0])
fig.savefig("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_V_lightcurve.png", bbox_inches="tight")
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.set_ylabel("brightness (mJy/beam)")
ax.set_xlabel("time / s")
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*qlc, lw=0.5, color='red', alpha=0.8, label="Q") 
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*ulc, lw=0.5, color='blue', alpha=0.8, label="U")
ax.set_xlim(0, arr["TIMES"][-1] - arr["TIMES"][0])
ax.legend()
fig.savefig("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_QU_lightcurve.png", bbox_inches="tight")
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.set_ylabel("brightness (mJy/beam)")
ax.set_xlabel("time / s")
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*vlc, lw=0.5, color='green', alpha=0.8)
ax.set_xlim(0, arr["TIMES"][-1] - arr["TIMES"][0])
ax.set_ylim(-20, 20)
fig.savefig("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_V_lightcurve.png", bbox_inches="tight")
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(111)
ax.set_ylabel("brightness (mJy/beam)")
ax.set_xlabel("time / s")
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*qlc, lw=0.5, color='red', alpha=0.8, label="Q") 
ax.plot(arr["TIMES"] - arr["TIMES"][0], 1000*ulc, lw=0.5, color='blue', alpha=0.8, label="U")
ax.set_xlim(0, arr["TIMES"][-1] - arr["TIMES"][0])
ax.legend()
ax.set_ylim(-20, 20)
fig.savefig("scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_QU_lightcurve.png", bbox_inches="tight")
