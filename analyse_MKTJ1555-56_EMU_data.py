#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

import glob
import sys

makeDynspec = False
makeLightcurves = False
makeFold = False
debugPoly = False
makeJointIQUV = False
makeUpperlimits = False

T0 = 59713.512505
P = 0.02168 # days

def ephem(n):
    return T0 + n*P

def pulsenum(mjd):
    return (mjd - T0) / P

def make_dynspec(data, vmin, vmax, cmap, extent, outname, imwidth=13):
    fig = plt.figure(figsize=(imwidth,5))
    ax = fig.add_subplot(111)
    ax.imshow(data, origin='lower',vmin = vmin, vmax=vmax, aspect='auto', cmap=cmap, extent=extent)
    ax.set_xlabel("time / s")
    ax.set_ylabel("frequency / GHz")
    fig.savefig(outname, bbox_inches="tight")
    plt.close()

def make_lightcurve(times, lc, vmin, vmax, lw, color, alpha, label, outname, offset=0.0, imwidth=13):
    fig = plt.figure(figsize=(imwidth,5))
    ax = fig.add_subplot(111)
    ax.set_ylabel("brightness (mJy/beam)")
    ax.set_xlabel("time / s")
    if isinstance(alpha, list):
        for t, l, w, c, a, lb in zip(times, lc, lw, color, alpha, label):
            ax.plot(t, l, lw=w, color=c, alpha=a, label=lb)
    else:
        ax.plot(times, lc, lw=lw, color=color, alpha=alpha, label=label)
        t = times
    for i in range(-3, 35):
        ax.axvline(ephem(i)*24*3600 - offset, alpha=0.4, color='orange')
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(vmin, vmax)
    ax.legend(loc=1)
    fig.savefig(outname, bbox_inches="tight")

XX = 0
XY = 1
YX = 2
YY = 3

cmap = { "I" : "viridis",
         "Q" : "RdBu_r",
         "U" : "RdBu",
         "V" : "PRGn" }

color = { "I" : "black",
         "Q" : "red",
         "U" : "blue",
         "V" : "green" }
# TODO find positions in primary beams and take weighted average, correct for beam attenuation
arr1 = np.load("dynspec/scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_averaged_cal.leakage.pkl", allow_pickle=True)
arr2 = np.load("dynspec/scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam09_averaged_cal.leakage.pkl", allow_pickle=True)
# This version has been run through a different set of software to try to get saner polarisation results
arr3 = np.load("dynspec/scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam09_averaged_cal.leakage_fixms.pkl", allow_pickle=True)
arr4 = np.load("dynspec/scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_averaged_cal.leakage_fixms.pkl", allow_pickle=True)

freqs_a = arr1["FREQS"]/1.e9
times_a = arr1["TIMES"]
times_az = arr1["TIMES"] - arr1["TIMES"][0]

mkt = np.load("dynspec/1652551867-sdp-l0_2026-05-22T14-51-05_zBI.pkl", allow_pickle=True)

freqs_m = mkt["FREQS"]/1.e9
times_m = mkt["TIMES"]
times_mz = mkt["TIMES"] - mkt["TIMES"][0]

# ASKAP data correct transforms -- if data has not been modified by FixMS
polaxis = -45.0
theta = 2.0 * np.radians(polaxis)
It1 = np.real((arr1["DS"][:,:,YY]+arr1["DS"][:,:,XX]))
It2 = np.real((arr2["DS"][:,:,YY]+arr2["DS"][:,:,XX]))
Qt1 = np.real(np.cos(theta)*(arr1["DS"][:,:,YX]+arr1["DS"][:,:,XY]) - np.sin(theta)*(arr1["DS"][:,:,YY]-arr1["DS"][:,:,XX]))
Qt2 = np.real(np.cos(theta)*(arr2["DS"][:,:,YX]+arr2["DS"][:,:,XY]) - np.sin(theta)*(arr2["DS"][:,:,YY]-arr2["DS"][:,:,XX]))
Ut1 = np.real(np.sin(theta)*(arr1["DS"][:,:,YX]+arr1["DS"][:,:,XY]) + np.cos(theta)*(arr1["DS"][:,:,YY]-arr1["DS"][:,:,XX]))
Ut2 = np.real(np.sin(theta)*(arr2["DS"][:,:,YX]+arr2["DS"][:,:,XY]) + np.cos(theta)*(arr2["DS"][:,:,YY]-arr2["DS"][:,:,XX]))
#Vt2 = np.real(-1j * (arr["DS"][:,:,XY]-arr["DS"][:,:,YX]))
Vt1 = np.imag((arr1["DS"][:,:,YX]-arr1["DS"][:,:,XY]))
Vt2 = np.imag((arr2["DS"][:,:,YX]-arr2["DS"][:,:,XY]))

# ASKAP transforms *after* running through FixMS, using Alec's conventions (which he says are the same as WScClean)
It3 = np.real((arr3["DS"][:,:,XX]+arr3["DS"][:,:,YY]))/2
Qt3 = np.real((arr3["DS"][:,:,XX]-arr3["DS"][:,:,YY]))/2
Ut3 = np.real((arr3["DS"][:,:,XY]+arr3["DS"][:,:,YX]))/2
Vt3 = np.real((-1j*arr3["DS"][:,:,XY]+1j*arr3["DS"][:,:,YX]))/2
It4 = np.real((arr4["DS"][:,:,XX]+arr4["DS"][:,:,YY]))/2
Qt4 = np.real((arr4["DS"][:,:,XX]-arr4["DS"][:,:,YY]))/2
Ut4 = np.real((arr4["DS"][:,:,XY]+arr4["DS"][:,:,YX]))/2
Vt4 = np.real((-1j*arr4["DS"][:,:,XY]+1j*arr4["DS"][:,:,YX]))/2

# RFI flagging
It1[:,124] = np.nan
Qt1[:,124] = np.nan
Ut1[:,124] = np.nan
Vt1[:,124] = np.nan
It2[:,124] = np.nan
Qt2[:,124] = np.nan
Ut2[:,124] = np.nan
Vt2[:,124] = np.nan
It3[:,124] = np.nan
Qt3[:,124] = np.nan
Ut3[:,124] = np.nan
Vt3[:,124] = np.nan
It4[:,124] = np.nan
Qt4[:,124] = np.nan
Ut4[:,124] = np.nan
Vt4[:,124] = np.nan

It1[:,0:16] = np.nan
Qt1[:,0:16] = np.nan
Ut1[:,0:16] = np.nan
Vt1[:,0:16] = np.nan
It2[:,0:16] = np.nan
Qt2[:,0:16] = np.nan
Ut2[:,0:16] = np.nan
Vt2[:,0:16] = np.nan
It3[:,0:16] = np.nan
Qt3[:,0:16] = np.nan
Ut3[:,0:16] = np.nan
Vt3[:,0:16] = np.nan
It4[:,0:16] = np.nan
Qt4[:,0:16] = np.nan
Ut4[:,0:16] = np.nan
Vt4[:,0:16] = np.nan

# Let's use the FixMS version since Alec is confident about that
It = (It3 + It4) / 2
Qt = (Qt3 + Qt4) / 2
Ut = (Ut3 + Ut4) / 2
Vt = (Vt3 + Vt4) / 2

# Form individual Stokes dynamic spectra -- of each beam, so we can check they agree, and then the combined data
if makeDynspec is True:
    vmin, vmax = -0.005, 0.03
    make_dynspec(It.T, vmin, vmax, cmap["I"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_StokesI_dynspec.png")
    make_dynspec(It1.T, vmin, vmax, cmap["I"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_beam15_StokesI_dynspec.png")
    make_dynspec(It2.T, vmin, vmax, cmap["I"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_beam09_StokesI_dynspec.png")
    make_dynspec(It3.T, vmin, vmax, cmap["I"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_beam09_StokesI_fixms_dynspec.png")
    make_dynspec(It4.T, vmin, vmax, cmap["I"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_beam15_StokesI_fixms_dynspec.png")

    vmin, vmax = -0.02, 0.02
    make_dynspec(Qt.T, vmin, vmax, cmap["Q"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_StokesQ_dynspec.png")
    make_dynspec(Qt1.T, vmin, vmax, cmap["Q"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_beam15_StokesQ_dynspec.png")
    make_dynspec(Qt2.T, vmin, vmax, cmap["Q"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_beam09_StokesQ_dynspec.png")
    make_dynspec(Qt3.T, vmin, vmax, cmap["Q"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_beam09_StokesQ_fixms_dynspec.png")
    make_dynspec(Qt4.T, vmin, vmax, cmap["Q"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_beam15_StokesQ_fixms_dynspec.png")

    make_dynspec(Ut.T, vmin, vmax, cmap["U"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_StokesU_dynspec.png")
    make_dynspec(Ut1.T, vmin, vmax, cmap["U"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_beam15_StokesU_dynspec.png")
    make_dynspec(Ut2.T, vmin, vmax, cmap["U"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_beam09_StokesU_dynspec.png")
    make_dynspec(Ut3.T, vmin, vmax, cmap["U"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_beam09_StokesU_fixms_dynspec.png")
    make_dynspec(Ut4.T, vmin, vmax, cmap["U"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_beam15_StokesU_fixms_dynspec.png")

    make_dynspec(Vt.T, vmin, vmax, cmap["V"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_StokesV_dynspec.png")
    make_dynspec(Vt1.T, vmin, vmax, cmap["V"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_beam15_StokesV_dynspec.png")
    make_dynspec(Vt2.T, vmin, vmax, cmap["V"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_beam09_StokesV_dynspec.png")
    make_dynspec(Vt3.T, vmin, vmax, cmap["V"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_beam09_StokesV_fixms_dynspec.png")
    make_dynspec(Vt4.T, vmin, vmax, cmap["V"], [0, times_az[-1], freqs_a[0], freqs_a[-1]], "EMU_beam15_StokesV_fixms_dynspec.png")


# Form light curves
ilc_a = np.nanmean(It, axis=1)
qlc_a = np.nanmean(Qt, axis=1)
ulc_a = np.nanmean(Ut, axis=1)
vlc_a = np.nanmean(Vt, axis=1)

ilc1_a = np.nanmean(It1, axis=1)
qlc1_a = np.nanmean(Qt1, axis=1)
ulc1_a = np.nanmean(Ut1, axis=1)
vlc1_a = np.nanmean(Vt1, axis=1)

ilc2_a = np.nanmean(It2, axis=1)
qlc2_a = np.nanmean(Qt2, axis=1)
ulc2_a = np.nanmean(Ut2, axis=1)
vlc2_a = np.nanmean(Vt2, axis=1)

ilc3_a = np.nanmean(It3, axis=1)
qlc3_a = np.nanmean(Qt3, axis=1)
ulc3_a = np.nanmean(Ut3, axis=1)
vlc3_a = np.nanmean(Vt3, axis=1)

ilc4_a = np.nanmean(It4, axis=1)
qlc4_a = np.nanmean(Qt4, axis=1)
ulc4_a = np.nanmean(Ut4, axis=1)
vlc4_a = np.nanmean(Vt4, axis=1)

if makeLightcurves is True:
    vmin, vmax =  -5, 15
    make_lightcurve(times_az, 1000*ilc_a, vmin, vmax, 0.5, color["I"], 1.0, 'Stokes I', 'EMU_StokesI_light_curve.png', offset=times_a[0])
    make_lightcurve(times_az, 1000*ilc1_a, vmin, vmax, 0.5, color["I"], 1.0, 'Stokes I', 'EMU_beam15_StokesI_light_curve.png', offset=times_a[0])
    make_lightcurve(times_az, 1000*ilc2_a, vmin, vmax, 0.5, color["I"], 1.0, 'Stokes I', 'EMU_beam09_StokesI_light_curve.png', offset=times_a[0])
    make_lightcurve(times_az, 1000*ilc3_a, vmin, vmax, 0.5, color["I"], 1.0, 'Stokes I', 'EMU_beam09_StokesI_fixms_light_curve.png', offset=times_a[0])
    make_lightcurve(times_az, 1000*ilc4_a, vmin, vmax, 0.5, color["I"], 1.0, 'Stokes I', 'EMU_beam15_StokesI_fixms_light_curve.png', offset=times_a[0])

    vmin, vmax =  -15, 15
    make_lightcurve(times_az, 1000*qlc_a, vmin, vmax, 0.5, color["Q"], 1.0, 'Stokes Q', 'EMU_StokesQ_light_curve.png', offset=times_a[0])
    make_lightcurve(times_az, 1000*qlc1_a, vmin, vmax, 0.5, color["Q"], 1.0, 'Stokes Q', 'EMU_beam15_StokesQ_light_curve.png', offset=times_a[0])
    make_lightcurve(times_az, 1000*qlc2_a, vmin, vmax, 0.5, color["Q"], 1.0, 'Stokes Q', 'EMU_beam09_StokesQ_light_curve.png', offset=times_a[0])
    make_lightcurve(times_az, 1000*qlc3_a, vmin, vmax, 0.5, color["Q"], 1.0, 'Stokes Q', 'EMU_beam09_StokesQ_fixms_light_curve.png', offset=times_a[0])
    make_lightcurve(times_az, 1000*qlc4_a, vmin, vmax, 0.5, color["Q"], 1.0, 'Stokes Q', 'EMU_beam15_StokesQ_fixms_light_curve.png', offset=times_a[0])

    make_lightcurve(times_az, 1000*ulc_a, vmin, vmax, 0.5, color["U"], 1.0, 'Stokes U', 'EMU_StokesU_light_curve.png', offset=times_a[0])
    make_lightcurve(times_az, 1000*ulc1_a, vmin, vmax, 0.5, color["U"], 1.0, 'Stokes U', 'EMU_beam15_StokesU_light_curve.png', offset=times_a[0])
    make_lightcurve(times_az, 1000*ulc2_a, vmin, vmax, 0.5, color["U"], 1.0, 'Stokes U', 'EMU_beam09_StokesU_light_curve.png', offset=times_a[0])
    make_lightcurve(times_az, 1000*ulc3_a, vmin, vmax, 0.5, color["U"], 1.0, 'Stokes U', 'EMU_beam09_StokesU_fixms_light_curve.png', offset=times_a[0])
    make_lightcurve(times_az, 1000*ulc4_a, vmin, vmax, 0.5, color["U"], 1.0, 'Stokes U', 'EMU_beam15_StokesU_fixms_light_curve.png', offset=times_a[0])

    make_lightcurve(times_az, 1000*vlc_a, vmin, vmax, 0.5, color["V"], 1.0, 'Stokes V', 'EMU_StokesV_light_curve.png', offset=times_a[0])
    make_lightcurve(times_az, 1000*vlc1_a, vmin, vmax, 0.5, color["V"], 1.0, 'Stokes V', 'EMU_beam15_StokesV_light_curve.png', offset=times_a[0])
    make_lightcurve(times_az, 1000*vlc2_a, vmin, vmax, 0.5, color["V"], 1.0, 'Stokes V', 'EMU_beam09_StokesV_light_curve.png', offset=times_a[0])
    make_lightcurve(times_az, 1000*vlc3_a, vmin, vmax, 0.5, color["V"], 1.0, 'Stokes V', 'EMU_beam09_StokesV_fixms_light_curve.png', offset=times_a[0])
    make_lightcurve(times_az, 1000*vlc4_a, vmin, vmax, 0.5, color["V"], 1.0, 'Stokes V', 'EMU_beam15_StokesV_fixms_light_curve.png', offset=times_a[0])

# MeerKAT data - basic transforms
#It_m = np.real((mkt["DS"][:,:,XX]+mkt["DS"][:,:,YY]))/2
#Qt_m = np.real((mkt["DS"][:,:,XX]-mkt["DS"][:,:,YY]))/2
#Ut_m = np.real((mkt["DS"][:,:,XY]+mkt["DS"][:,:,YX]))/2
#Vt_m = np.imag((mkt["DS"][:,:,XY]-mkt["DS"][:,:,YX]))/2

# MeerKAT data - Alec's suggestion (effectively this swaps Q and U)
It_m = np.real((mkt["DS"][:,:,XX]+mkt["DS"][:,:,YY]))/2
Qt_m = np.real((-mkt["DS"][:,:,XY]-mkt["DS"][:,:,YX]))/2
Ut_m = np.real((mkt["DS"][:,:,XX]-mkt["DS"][:,:,YY]))/2
Vt_m = np.real((-1j*mkt["DS"][:,:,XY]+1j*mkt["DS"][:,:,YX]))/2

indstart = 460
indend = 505
i = 22
if makeDynspec is True:
    # Form individual Stokes dynamic spectra
    vmin, vmax = -0.005, 0.03
    make_dynspec(It_m.T, vmin, vmax, cmap["I"], [0, times_mz[-1], freqs_m[0], freqs_m[-1]], "MeerKAT_StokesI_dynspec.png")

    vmin, vmax = -0.005, 0.005
    make_dynspec(Qt_m.T, vmin, vmax, cmap["Q"], [0, times_mz[-1], freqs_m[0], freqs_m[-1]], "MeerKAT_StokesQ_dynspec.png")
    make_dynspec(Ut_m.T, vmin, vmax, cmap["U"], [0, times_mz[-1], freqs_m[0], freqs_m[-1]], "MeerKAT_StokesU_dynspec.png")
    make_dynspec(Vt_m.T, vmin, vmax, cmap["V"], [0, times_mz[-1], freqs_m[0], freqs_m[-1]], "MeerKAT_StokesV_dynspec.png")

    # Zoom in on that really interesting bit
    vmin, vmax = -0.005, 0.005
    make_dynspec(Qt_m[indstart:indend].T, vmin, vmax, cmap["Q"], [times_mz[indstart], times_mz[indend], freqs_m[0], freqs_m[-1]], "MeerKAT_StokesQ_dynspec_zoom.png", imwidth=5)
    make_dynspec(Ut_m[indstart:indend].T, vmin, vmax, cmap["U"], [times_mz[indstart], times_mz[indend], freqs_m[0], freqs_m[-1]], "MeerKAT_StokesU_dynspec_zoom.png", imwidth=5)
    make_dynspec(Vt_m[indstart:indend].T, vmin, vmax, cmap["V"], [times_mz[indstart], times_mz[indend], freqs_m[0], freqs_m[-1]], "MeerKAT_StokesV_dynspec_zoom.png", imwidth=5)

# Form light curves
ilc_m = np.nanmean(It_m, axis=1)
qlc_m = np.nanmean(Qt_m, axis=1)
ulc_m = np.nanmean(Ut_m, axis=1)
vlc_m = np.nanmean(Vt_m, axis=1)

# MeerKAT Stokes I has a slow ripple to it that needs fixing
# First break the data into four segments
tdiff = times_m[1:] - times_m[0:-1]
tbreak = np.where(np.abs(tdiff) > 50)[0]
#print(tbreaks)
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

if debugPoly is True:
    make_lightcurve([t, t, t],
                    [1000*y, 1000*y_smooth, 1000*(y - y_smooth)],
                    vmin, vmax,
                    [2, 0.5, 1],
                    [color["I"], color["I"], color["I"]],
                    [0.4, 0.8, 1.0],
                    ['Original data', 'Fitted curve', 'Residual'],
                    'MeerKAT_StokesI_polyfit_segment1.png',
                    imwidth=8)

ilc_m[:seg1_end] = y - y_smooth
# And now flag the buffer
ilc_m[0:b] = np.nan
ilc_m[seg1_end-b:seg1_end] = np.nan
qlc_m[0:b] = np.nan
qlc_m[seg1_end-b:seg1_end] = np.nan
ulc_m[0:b] = np.nan
ulc_m[seg1_end-b:seg1_end] = np.nan
vlc_m[0:b] = np.nan
vlc_m[seg1_end-b:seg1_end] = np.nan

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

if debugPoly is True:
    make_lightcurve([t, t, t],
                    [1000*y, 1000*y_smooth, 1000*(y - y_smooth)],
                    vmin, vmax,
                    [2, 0.5, 1],
                    [color["I"], color["I"], color["I"]],
                    [0.4, 0.8, 1.0],
                    ['Original data', 'Fitted curve', 'Residual'],
                    'MeerKAT_StokesI_polyfit_segment2.png',
                    imwidth=8)

ilc_m[seg1_end:seg2_end] = y - y_smooth
# And now flag the buffer
ilc_m[seg1_end:seg1_end+b] = np.nan
ilc_m[seg2_end-b:seg2_end] = np.nan
qlc_m[seg1_end:seg1_end+b] = np.nan
qlc_m[seg2_end-b:seg2_end] = np.nan
ulc_m[seg1_end:seg1_end+b] = np.nan
ulc_m[seg2_end-b:seg2_end] = np.nan
vlc_m[seg1_end:seg1_end+b] = np.nan
vlc_m[seg2_end-b:seg2_end] = np.nan

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

if debugPoly is True:
    make_lightcurve([t, t, t],
                    [1000*y, 1000*y_smooth, 1000*(y - y_smooth)],
                    vmin, vmax,
                    [2, 0.5, 1],
                    [color["I"], color["I"], color["I"]],
                    [0.4, 0.8, 1.0],
                    ['Original data', 'Fitted curve', 'Residual'],
                    'MeerKAT_StokesI_polyfit_segment3.png',
                    imwidth=8)

ilc_m[seg2_end:seg3_end] = y - y_smooth
# And now flag the buffer
ilc_m[seg2_end:seg2_end+b] = np.nan
ilc_m[seg3_end-b:seg3_end] = np.nan
qlc_m[seg2_end:seg2_end+b] = np.nan
qlc_m[seg3_end-b:seg3_end] = np.nan
ulc_m[seg2_end:seg2_end+b] = np.nan
ulc_m[seg3_end-b:seg3_end] = np.nan
vlc_m[seg2_end:seg2_end+b] = np.nan
vlc_m[seg3_end-b:seg3_end] = np.nan

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

if debugPoly is True:
    make_lightcurve([t, t, t],
                    [1000*y, 1000*y_smooth, 1000*(y - y_smooth)],
                    vmin, vmax,
                    [2, 0.5, 1],
                    [color["I"], color["I"], color["I"]],
                    [0.4, 0.8, 1.0],
                    ['Original data', 'Fitted curve', 'Residual'],
                    'MeerKAT_StokesI_polyfit_segment4.png',
                    imwidth=8)

ilc_m[seg3_end:] = y - y_smooth
# And now flag the buffer
ilc_m[seg3_end:seg3_end+b] = np.nan
ilc_m[-b:] = np.nan
qlc_m[seg3_end:seg3_end+b] = np.nan
qlc_m[-b:] = np.nan
ulc_m[seg3_end:seg3_end+b] = np.nan
ulc_m[-b:] = np.nan
vlc_m[seg3_end:seg3_end+b] = np.nan
vlc_m[-b:] = np.nan

# Plot light curves
vmin, vmax = -3, 20
make_lightcurve(times_mz, 1000*ilc_m, vmin, vmax, 0.5, color["I"], 1.0, 'Stokes I', 'MeerKAT_StokesI_light_curve.png', offset=times_m[0])
vmin, vmax = -20, 20
make_lightcurve(times_mz, 1000*qlc_m, vmin, vmax, 0.5, color["Q"], 1.0, 'Stokes Q', 'MeerKAT_StokesQ_light_curve.png', offset=times_m[0])
make_lightcurve(times_mz, 1000*ulc_m, vmin, vmax, 0.5, color["U"], 1.0, 'Stokes U', 'MeerKAT_StokesU_light_curve.png', offset=times_m[0])
make_lightcurve(times_mz, 1000*vlc_m, vmin, vmax, 0.5, color["V"], 1.0, 'Stokes V', 'MeerKAT_StokesV_light_curve.png', offset=times_m[0])
make_lightcurve([times_mz, times_mz, times_mz],
                [1000*qlc_m, 1000*ulc_m, 1000*vlc_m],
                vmin, vmax,
                [0.5, 0.5, 0.5],
                [color["Q"], color["U"], color["V"]],
                [1.0, 1.0, 1.0],
                ['Stokes Q', 'Stokes U', 'Stokes V'],
                'MeerKAT_StokesQUV_light_curve.png', offset=times_m[0])

# That interesting section of microstructure
vmin, vmax = -3, 12
make_lightcurve([times_mz[indstart:indend], times_mz[indstart:indend], times_mz[indstart:indend], times_mz[indstart:indend]],
                [1000*ilc_m[indstart:indend], 1000*qlc_m[indstart:indend], 1000*ulc_m[indstart:indend], 1000*vlc_m[indstart:indend]],
                vmin, vmax,
                [0.5, 0.5, 0.5, 0.5],
                [color["I"], color["Q"], color["U"], color["V"]],
                [0.8, 0.8, 0.8, 0.8],
                ['Stokes I', 'Stokes Q', 'Stokes U', 'Stokes V'],
                'MeerKAT_StokesIQUV_light_curve_zoom.png',
                offset=times_m[indstart],
                imwidth=5)

# Can we use this to constrain the RM and/or test whether there is Faraday rotation?
# I noticed there is a nasty RFI spike in channel index 871
# And values up to index 16 are not trustworthy
Qt_m[:, 871] = np.nan
Ut_m[:, 871] = np.nan
Vt_m[:, 871] = np.nan

Qt_m[:, 0:17] = np.nan
Ut_m[:, 0:17] = np.nan
Vt_m[:, 0:17] = np.nan


# These are really only informative for finding RFI
for i in range(indstart, indend):
    if ilc_m[i] > 0.001:
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)
#        ax.plot(freqs_m, 1000*It_m[i], color = color["I"], lw=0.5,  label="Stokes I")
        ax.plot(freqs_m, 1000*Qt_m[i], color = color["Q"], lw=0.5,  label="Stokes Q")
        ax.plot(freqs_m, 1000*Ut_m[i], color = color["U"], lw=0.5,  label="Stokes U")
        ax.plot(freqs_m, 1000*Vt_m[i], color = color["V"], lw=0.5,  label="Stokes V")
        ax.set_xlabel("Frequency / GHz")
        ax.set_ylabel("Flux density / mJy")
        t = times_m[i]
        fig.savefig(f"MeerKAT_IQUV_spectrum_{t}.png", bbox_inches="tight")

# find the common time range

tstart = np.nanmin([arr1["TIMES"][0], mkt["TIMES"][0]])
tend = np.nanmax([arr1["TIMES"][-1], mkt["TIMES"][-1]])

# Zoom in on the interesting section, make joint plots
tstart, tend = 5159271500.0, 5159272500.0

if makeJointIQUV is True:
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.set_ylabel("brightness (mJy/beam)")
    ax.set_xlabel("time / s")
    ax.plot(times_a, 1000*ilc_a, lw=2, color='darkgrey', alpha=0.5, label="ASKAP Stokes I")
    ax.plot(times_m, 1000*ilc_m, lw=0.5, color='black', alpha=0.8, label="MeerKAT Stokes I")
    for i in range(-3, 30):
        ax.axvline(ephem(i)*24*3600, alpha=0.4, color='orange')
    ax.set_xlim(tstart, tend)
    ax.set_ylim(-10, 30)
    ax.legend(loc=1)
    fig.savefig("Joint_StokesI_lightcurve.png", bbox_inches="tight")

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.set_ylabel("brightness (mJy/beam)")
    ax.set_xlabel("time / s")
    ax.plot(times_a, 1000*qlc_a, lw=2, color='red', alpha=0.5, label="ASKAP Stokes Q") 
    ax.plot(times_m, 1000*qlc_m, lw=0.5, color='darkred', alpha=0.8, label="MeerKAT Stokes Q") 
    ax.plot(times_a, 1000*ulc_a, lw=2, color='blue', alpha=0.5, label="ASKAP Stokes U")
    ax.plot(times_m, 1000*ulc_m, lw=0.5, color='darkblue', alpha=0.8, label="MeerKAT Stokes U")
    ax.plot(times_a, 1000*vlc_a, lw=2, color='green', alpha=0.5, label="ASKAP Stokes V")
    ax.plot(times_m, 1000*vlc_m, lw=0.5, color='darkgreen', alpha=0.8, label="MeerKAT Stokes V")
    for i in range(-3, 30):
        ax.axvline(ephem(i)*24*3600, alpha=0.4, color='orange')
    ax.set_xlim(tstart, tend)
    ax.legend(loc=1)
    fig.savefig("Joint_StokesQUV_lightcurve.png", bbox_inches="tight")

if makeFold is True:
    # Fold the ASKAP data
    # Put these into MJD (instead of MJD seconds)
    trange = times_a / (24*3600)

    phase = np.mod(trange, 2*P)/(2*P)
    idx = np.argsort(phase)

    num_bins = 150

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(phase[idx], bin_edges) - 1

    I_sums = np.bincount(bin_indices, weights=1000*ilc_a[idx], minlength=num_bins)
    Q_sums = np.bincount(bin_indices, weights=1000*qlc_a[idx], minlength=num_bins)
    U_sums = np.bincount(bin_indices, weights=1000*ulc_a[idx], minlength=num_bins)
    V_sums = np.bincount(bin_indices, weights=1000*vlc_a[idx], minlength=num_bins)

    counts = np.bincount(bin_indices, minlength=num_bins)

    # Prevent division by zero if a bin is empty
    I_binned_average = I_sums / np.where(counts == 0, 1, counts)
    Q_binned_average = Q_sums / np.where(counts == 0, 1, counts)
    U_binned_average = U_sums / np.where(counts == 0, 1, counts)
    V_binned_average = V_sums / np.where(counts == 0, 1, counts)

    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    ax.plot(bin_centers, I_binned_average, color=color['I'], alpha=0.8, lw=0.5, label="Stokes I")
    ax.plot(bin_centers, Q_binned_average, color=color['Q'], alpha=0.8, lw=0.5, label="Stokes Q")
    ax.plot(bin_centers, U_binned_average, color=color['U'], alpha=0.8, lw=0.5, label="Stokes U")
    ax.plot(bin_centers, V_binned_average, color=color['V'], alpha=0.8, lw=0.5, label="Stokes V")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Mean brightness (mJy)")
    ax.legend(loc=1)
    fig.savefig("Folded_EMU_light_curve.png", bbox_inches="tight")

    # Fold the MeerKAT data
    # TODO: Need to solve for the polarisation calibration AND apply a parallactic angle correction before this makese sense
    trange = times_m / (24*3600)

    phase = np.mod(trange, 2*P)/(2*P)
    idx = np.argsort(phase)

    num_bins = 150

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(phase[idx], bin_edges) - 1

    I_sums = np.bincount(bin_indices, weights=1000*ilc_m[idx], minlength=num_bins)
    Q_sums = np.bincount(bin_indices, weights=1000*qlc_m[idx], minlength=num_bins)
    U_sums = np.bincount(bin_indices, weights=1000*ulc_m[idx], minlength=num_bins)
    V_sums = np.bincount(bin_indices, weights=1000*vlc_m[idx], minlength=num_bins)

    counts = np.bincount(bin_indices, minlength=num_bins)

    # Prevent division by zero if a bin is empty
    I_binned_average = I_sums / np.where(counts == 0, 1, counts)
    Q_binned_average = Q_sums / np.where(counts == 0, 1, counts)
    U_binned_average = U_sums / np.where(counts == 0, 1, counts)
    V_binned_average = V_sums / np.where(counts == 0, 1, counts)

    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    ax.plot(bin_centers, I_binned_average, color=color['I'], alpha=0.8, lw=0.5, label="Stokes I")
    ax.plot(bin_centers, Q_binned_average, color=color['Q'], alpha=0.8, lw=0.5, label="Stokes Q")
    ax.plot(bin_centers, U_binned_average, color=color['U'], alpha=0.8, lw=0.5, label="Stokes U")
    ax.plot(bin_centers, V_binned_average, color=color['V'], alpha=0.8, lw=0.5, label="Stokes V")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Mean brightness (mJy)")
    ax.legend(loc=1)
    fig.savefig("Folded_MeerKAT_light_curve.png", bbox_inches="tight")

# Load the rest of the .pkl files and calculate RMS and time for non-detections plot

if makeUpperlimits is True:
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

