#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u

import glob
import sys

makeDynspec = False
makeLightcurves = False
makePaperDS = False
makeFold = False
# NB: if you want to makeRM or makePhaseBin, you must also makeFold
makeRM = False
makePhaseBin = False
debugPoly = False
makeJointIQUV = False
makeUpperlimits = True

cm = 1/2.54  # centimeters in inches
# Figure font size
plt.rcParams.update({
    "font.size": 6})

T0 = 59713.512505
P = 62.2028 / (2*24*60) # minutes into days

source_sc = SkyCoord("15h55m43.69s -56d31m02.4s", frame='fk5')
beam_names = ["09", "15"]
beams_sc = SkyCoord(["15h58m56.563 -56d53m16.761", "15h55m36.850 -56d06m49.968"], frame='fk5')

# ASKAP Aperture radius (m)
a = 6

# Phase of the main pulse 
phase_start = 0.48
phase_end = 0.56

def ephem(n):
    return T0 + n*P

def pulsenum(mjd):
    return (mjd - T0) / (2*P)

def ipulsenum(mjd):
    return (mjd - T0) / (P)

def nicedate(t = None):
    ''' take a Time object and return a pleasantly formatted ISO string without excess precision on the seconds '''
    return "{0}-{1:02.0f}-{2:02.0f} {3:02.0f}:{4:02.0f}:{5:02.0f}".format(t.ymdhms[0], t.ymdhms[1], t.ymdhms[2], t.ymdhms[3], t.ymdhms[4], t.ymdhms[5])

def justdate(t = None):
    ''' take a Time object and return just the date '''
    return "{0}-{1:02.0f}-{2:02.0f}".format(t.ymdhms[0], t.ymdhms[1], t.ymdhms[2])

def hhmm(t = None):
    ''' take a Time object and return just the time with no seconds '''
    return "{0:02.0f}:{1:02.0f}".format(t.ymdhms[3], t.ymdhms[4])

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
        ax.axvline((ephem(i)*24*3600 - offset), alpha=0.4, color='orange')
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(vmin, vmax)
    ax.legend(loc=1)
    fig.savefig(outname, bbox_inches="tight")

class GaussianPB:
    
    def __init__(self,aperture = 12.0, expscaling = 4.0 * np.log(2.0), frequency = 1.1e9):
        
        self.aperture = aperture
        self.expScaling = expscaling
        self.frequency=frequency
        self.setXwidth(self.getFWHM())
        self.setYwidth(self.getFWHM())
        self.setAlpha(0.0)
        self.setXoff(0.0)
        self.setYoff(0.0)
        
    
    def getFWHM(self):
        sol = 299792458.0;
        fwhm = sol / self.frequency / self.aperture
        return fwhm
    
    def evaluate(self,offset = 0.0, freq = 0.0):
        if (freq > 0):
            self.frequency=freq
            
        pb = np.exp(-offset * offset * self.expScaling / (self.getFWHM() * self.getFWHM()))
        return pb
    
    def setXwidth(self,xwidth):
        self.xwidth = xwidth
        
    def setYwidth(self,ywidth):
        self.ywidth = ywidth
        
    def setAlpha(self,Angle):
        self.Alpha = Angle
        
    def setXoff(self,xoff):
        self.xoff = xoff
        
    def setYoff(self,yoff):
        self.yoff = yoff
        
    def evaluateAtOffset(self,offsetPAngle=0, offsetDist=0, freq=0):
            
        # x-direction is assumed along the meridian in the direction of north celestial pole
        # the offsetPA angle is relative to the meridian
        # the Alpha angle is the rotation of the beam pattern relative to the meridian
        # Therefore the offset relative to the
                
        if (freq > 0):
            self.frequency=freq
            
        x_angle = offsetDist * np.cos(offsetPAngle - self.Alpha)
        y_angle = offsetDist * np.sin(offsetPAngle - self.Alpha)
                    
        x_pb = np.exp(-1.0 * self.expScaling * np.power((x_angle - self.xoff) / self.xwidth, 2.0))
        y_pb = np.exp(-1.0 * self.expScaling * np.power((y_angle - self.yoff) / self.ywidth, 2.0))

        return x_pb * y_pb
XX = 0
XY = 1
YX = 2
YY = 3

cmap = { "I" : "viridis",
         "Q" : "RdBu_r",
         "U" : "RdBu",
         "V" : "PRGn" }

# Trying to make paper plots that show things
cmap = { "I" : "plasma",
         "Q" : "bwr_r",
         "U" : "bwr_r",
         "V" : "bwr_r"}

color = { "I" : "black",
         "Q" : "red",
         "U" : "blue",
         "V" : "green",
         "L" : "orange" }

# A better match to pulsar astronomy conventions
color = { "I" : "black",
         "Q" : "green",
         "U" : "orange",
         "V" : "blue",
         "L" : "red",
         "T" : "darkgrey" }

arr1 = np.load("dynspec/scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_averaged_cal.leakage.pkl", allow_pickle=True)
arr2 = np.load("dynspec/scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam09_averaged_cal.leakage.pkl", allow_pickle=True)
# This version has been run through FixMS, which is current ASKAP best practice
arr3 = np.load("dynspec/scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam09_averaged_cal.leakage_fixms.pkl", allow_pickle=True)
arr4 = np.load("dynspec/scienceData.EMU_1554-55_band2.SB40625.EMU_1554-55_band2.beam15_averaged_cal.leakage_fixms.pkl", allow_pickle=True)

freqs_a = arr1["FREQS"]/1.e9
times_a = arr1["TIMES"]
times_az = arr1["TIMES"] - arr1["TIMES"][0]

mkt = np.load("dynspec/1652551867-sdp-l0_2026-05-22T14-51-05_zBI.pkl", allow_pickle=True)
# Can't use this as it has been baseline-dependent-averaged!
mkt2 = np.load("dynspec/G326.31.8_Subbed.uvfits.pkl", allow_pickle=True)
#mkt = np.load("dynspec/G326.31.8_Subbed.uvfits.pkl", allow_pickle=True)

freqs_m = mkt["FREQS"]/1.e9
times_m = mkt["TIMES"]
times_m2 = mkt2["TIMES"]
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
# These are the values printed from the single-frequency code, in order to calibrate our expectations below
# Nearest beam to source is beam 15 with a separation of 0.404 deg
#Next-nearest beam to source is beam 09 with a separation of 0.576 deg

# Get the weights for each beam. Array 3 is beam 9 and Array 4 is beam 15.
seps = beams_sc.separation(source_sc)

#Beam 09
pb_vals = []
for freq in freqs_a:
    pb = GaussianPB(frequency = freq*1.e9)
    pb_vals.append(pb.evaluate(seps[0].rad, freq=freq*1.e9))
# This is an array of numbers >1 that you would need to multiply by, shaped into a time, frequency 2D array
pb_corr_3 = 1. / np.tile(np.array(pb_vals), (It3.shape[0],1))

# Sanity check
fig = plt.figure(figsize=(8,5))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.9])
cax = fig.add_axes([0.92, 0.1, 0.05, 0.9])
img = ax.imshow(pb_corr_3.T, origin='lower', aspect='auto')
plt.colorbar(img, cax=cax)
ax.set_xlabel("time")
ax.set_ylabel("channel")
fig.savefig("pb_corr_test.png", bbox_inches="tight")

# Beam 15
pb_vals = []
for freq in freqs_a:
    pb = GaussianPB(frequency = freq*1.e9)
    pb_vals.append(pb.evaluate(seps[1].rad, freq=freq*1.e9))
# This is an array of numbers >1 that you would need to multiply by, shaped into a time, frequency 2D array
pb_corr_4 = 1. / np.tile(np.array(pb_vals), (It4.shape[0],1))

# Correcting for the primary beam means multiplying by the primary beam correction (which is inversely proportional to distance to the phase centre)
# Weighting by the primary beam means dividing by the primary beam correction (as big numbers are bad!)
# So, effectively, those terms cancel out, and at the end we want to simply divide by the sum of the inverse of the primary beam corrections

w3 = 1/ pb_corr_3
w4 = 1/ pb_corr_4
It = (It3 + It4) / (w3 + w4)
Qt = (Qt3 + Qt4) / (w3 + w4)
Ut = (Ut3 + Ut4) / (w3 + w4)
Vt = (Vt3 + Vt4) / (w3 + w4)

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

if makePaperDS:
    # This will be a full-page plot
    fig = plt.figure(figsize=(17.9*cm, 8*cm))
    extent = [0, times_az[-1]/3600, freqs_a[16], freqs_a[-1]]
    # Top-left, Stokes I
    lw = 0.5
    alpha = 0.8
    vmin, vmax = -3, 20
    ax_I_ds = fig.add_axes([0.1, 0.53, 0.3, 0.3])
    cax_I = fig.add_axes([0.41, 0.53, 0.015, 0.3])
    Ids = ax_I_ds.imshow(1000*It[:,16:].T, interpolation='none', origin='lower',vmin = vmin, vmax=vmax, aspect='auto', cmap=cmap["I"], extent=extent)
    fig.colorbar(Ids, cax = cax_I)
    ax_I_lc = fig.add_axes([0.1, 0.83, 0.3, 0.1])
    ax_I_lc.plot(times_az/3600, 1000*ilc_a, lw=lw, color=color["I"], alpha=alpha, label="I")


    # Define a nice diverging colormap that de-emphasises the noisy values
    base_cmap = plt.get_cmap('bwr')
    # Increase the exponent (e.g., to 3 or 4) to make it fade even slower near white
    exponent = 2.0
    num_points = 256
    x = np.linspace(-1, 1, num_points)
    # Warp the steps using a power function, then shift back to a 0-to-1 range
    # This clusters points heavily around the center (0.5), pushing colors to the edges
    warped_x = np.sign(x) * (np.abs(x) ** exponent)
    colors_sampled = base_cmap((warped_x + 1) / 2)
    # Create the new colormap from these warped colors
    slow_fade_cmap = colors.ListedColormap(colors_sampled)
    # 4. Normalize symmetrically around 0 so the center is exactly white
    max_abs = 20
    my_norm = colors.Normalize(vmin=-max_abs, vmax=max_abs)

    # Top-right, Stokes Q
    ax_Q_ds = fig.add_axes([0.5, 0.53, 0.3, 0.3])
    cax_Q = fig.add_axes([0.81, 0.53, 0.015, 0.3])
    Qds = ax_Q_ds.imshow(1000*Qt[:,16:].T, interpolation='none', origin='lower', aspect='auto', extent=extent, cmap=slow_fade_cmap, norm=my_norm)
    fig.colorbar(Qds, cax = cax_Q, label='Flux density / mJy')
    ax_Q_lc = fig.add_axes([0.5, 0.83, 0.3, 0.1])
    ax_Q_lc.plot(times_az/3600, 1000*qlc_a, lw=lw, color=color["Q"], alpha=alpha, label="Q")
    # Bottom-left, Stokes U
    ax_U_ds = fig.add_axes([0.1, 0.1, 0.3, 0.3])
    cax_U = fig.add_axes([0.41, 0.1, 0.015, 0.3])
    Uds = ax_U_ds.imshow(1000*Ut[:,16:].T, interpolation='none', origin='lower', aspect='auto', extent=extent, cmap=slow_fade_cmap, norm=my_norm)
    fig.colorbar(Uds, cax = cax_U)
    ax_U_lc = fig.add_axes([0.1, 0.4, 0.3, 0.1])
    ax_U_lc.plot(times_az/3600, 1000*ulc_a, lw=lw, color=color["U"], alpha=alpha, label="U")
    # Bottom-right, Stokes V
    ax_V_ds = fig.add_axes([0.5, 0.1, 0.3, 0.3])
    cax_V = fig.add_axes([0.81, 0.1, 0.015, 0.3])
    Vds = ax_V_ds.imshow(1000*Vt[:,16:].T, interpolation='none', origin='lower', aspect='auto', extent=extent, cmap=slow_fade_cmap, norm=my_norm)
    fig.colorbar(Vds, cax = cax_V, label='Flux density / mJy')
    ax_V_lc = fig.add_axes([0.5, 0.4, 0.3, 0.1])
    ax_V_lc.plot(times_az/3600, 1000*vlc_a, lw=lw, color=color["V"], alpha=alpha, label="V")

    offset = times_a[0]
    for ax in [ax_I_lc, ax_Q_lc, ax_U_lc, ax_V_lc]:
        ax.set_xlim(times_az[0]/3600, times_az[-1]/3600)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        for i in range(-3, 35):
            ax.axvline((ephem(i)*24*3600 - offset)/3600, alpha=0.1, color='grey')
    for ax in [ax_I_ds, ax_U_ds]:
        ax.set_ylabel("Frequency / GHz")

    for ax in [ax_I_lc, ax_U_lc]:
        ax.set_ylabel("$S$ / mJy")

    tstart = nicedate(Time(times_a[0]/(24*3600), format='mjd', scale='utc'))
    for ax in [ax_U_ds, ax_V_ds]:
        ax.set_xlabel(f"Time / hours since {tstart}")

    for ax in [ax_Q_ds, ax_Q_lc, ax_I_ds, ax_I_lc, ax_U_lc, ax_V_lc]:
        ax.set_xticklabels([])

    for ax in [ax_Q_ds, ax_V_ds]:
        ax.set_yticklabels([])


    fig.savefig("ASKAP_dynamic_spectra_lcs.pdf", bbox_inches="tight", dpi=300)


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

# Original high-resolution data -- hopefully we will get a polarisation-calibrated version some day
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

### Sanity check
#fig = plt.figure(figsize=(8,5))
#ax = fig.add_axes([0.1, 0.1, 0.8, 0.9])
##cax = fig.add_axes([0.92, 0.1, 0.05, 0.9])
##img = ax.imshow(pb_corr_3.T, origin='lower', aspect='auto')
#ax.plot(times_m, label='From SDP')
#ax.plot(times_m2, label='From Bill')
#ax.set_xlabel("index")
#ax.set_ylabel("times (MJD seconds)")
#ax.legend()
#fig.savefig("time_axis.png", bbox_inches="tight")


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
    ax.plot(times_a, 1000*ilc_a, lw=2, color='darkgrey', alpha=0.8, label="ASKAP Stokes I")
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

    phase = np.mod(trange - T0 + P, 2*P)/(2*P)
    idx = np.argsort(phase)

    num_bins = 200

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
    L_binned_average = np.sqrt(Q_binned_average**2 + U_binned_average**2)
    L_frac = 100*L_binned_average/I_binned_average
    V_frac = 100*(np.abs(V_binned_average)/I_binned_average)
    T_frac = 100*np.sqrt((Q_binned_average**2 + U_binned_average**2 + V_binned_average**2)/(I_binned_average**2))

    phase_start_ip = 0.04
    phase_end_ip = 0.057
# Paper figure: aim for half an A4 column (8cm)
    fig = plt.figure(figsize=(8*cm,8*cm))
    ax1 = fig.add_subplot(211)
    ax1.axvspan(phase_start, phase_end, alpha=0.1, color='grey')
    ax1.axvspan(phase_start_ip, phase_end_ip, alpha=0.1, color='grey')
    ax1.plot(bin_centers, I_binned_average, color=color['I'], alpha=0.8, lw=0.5, label="Stokes I")
    ax1.plot(bin_centers, Q_binned_average, color=color['Q'], alpha=0.8, lw=0.5, label="Stokes Q")
    ax1.plot(bin_centers, U_binned_average, color=color['U'], alpha=0.8, lw=0.5, label="Stokes U")
    ax1.plot(bin_centers, V_binned_average, color=color['V'], alpha=0.8, lw=0.5, label="Stokes V")
    ax1.set_xlabel("Phase")
    ax1.set_ylabel("Mean brightness (mJy)")
    ax1.set_xlim(0, 1)
    ax1.legend(loc=1)
    ax2 = fig.add_subplot(212)
    ax2.axvspan(phase_start, phase_end, alpha=0.1, color='grey')
    ax2.axvspan(phase_start_ip, phase_end_ip, alpha=0.1, color='grey')

    I_cut = 0.5 #mJy
  # Main pulse
    ind1 = np.logical_and(np.abs(I_binned_average)>I_cut, np.logical_and(bin_centers > phase_start, bin_centers < phase_end))
  # Little circularly polarised pulse
    ind2 = np.logical_and(np.abs(I_binned_average)>I_cut, np.logical_and(bin_centers > phase_start_ip, bin_centers < phase_end_ip))
    ax2.plot(bin_centers[ind1], L_frac[ind1], color=color['L'], alpha=0.8, lw=0.5, label="Linear")
    ax2.plot(bin_centers[ind1], V_frac[ind1], color=color['V'], alpha=0.8, lw=0.5, label="Circular")
    ax2.plot(bin_centers[ind1], T_frac[ind1], color=color['T'], alpha=0.8, lw=0.5, label="Total")
    ax2.plot(bin_centers[ind2], L_frac[ind2], color=color['L'], alpha=0.8, lw=0.5)
    ax2.plot(bin_centers[ind2], V_frac[ind2], color=color['V'], alpha=0.8, lw=0.5)
    ax2.plot(bin_centers[ind2], T_frac[ind2], color=color['T'], alpha=0.8, lw=0.5)
    ax2.set_xlabel("Phase")
    ax2.set_ylim(0, 120)
    ax2.set_xlim(0, 1)
    ax2.set_ylabel("$|$Fractional$|$ polarisation (%)")
    ax2.legend(loc=1)
    n_p = times_az[-1] / (2*P*24*3600)
    len_ip = (phase_end_ip - phase_start_ip) * 2*P *24*3600
    len_mp = (phase_end - phase_start) * 2*P *24*3600
    print(f"Successfully stacked {n_p:2.2f} periods, boosting S/N by {np.sqrt(n_p):2.2f}")
    print(f"Main pulse is about {len_mp:2.0f}s wide.")
    print(f"Inter-pulse is about {len_ip:2.0f}s wide.")
    fig.savefig("Folded_EMU_light_curve.pdf", bbox_inches="tight")

    if makeRM is True:
# Try to fit the RM from the highest S/N phase bins in the EMU data
        ind = np.logical_and(phase>phase_start, phase<phase_end)
        Ilc_pulse = ilc_a[ind]
   
        #Sanity check
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.scatter(phase[ind], Ilc_pulse)
        ax.set_xlabel("Phase")
        ax.set_ylabel("Stokes I light curve just of main pulse")
        fig.savefig("test_pulse_capture.png", bbox_inches="tight")

        weights = np.tile(Ilc_pulse, (Qt.shape[1],1)).T
        weights[weights<0] = 0.
    # Normalise the weights to 1 as this will be useful later
        weights /= np.nanmax(weights)

        I_pulse = np.nansum(It[ind,:]*weights, axis=0)/np.nansum(weights,axis=0)
        Q_pulse = np.nansum(Qt[ind,:]*weights, axis=0)/np.nansum(weights,axis=0)
        U_pulse = np.nansum(Ut[ind,:]*weights, axis=0)/np.nansum(weights,axis=0)
        V_pulse = np.nansum(Vt[ind,:]*weights, axis=0)/np.nansum(weights,axis=0)
    # Removes RFI-flagged area
        I_pulse[I_pulse==0.0] = np.nan
        Q_pulse[Q_pulse==0.0] = np.nan
        U_pulse[U_pulse==0.0] = np.nan
        V_pulse[V_pulse==0.0] = np.nan

    # RMS is just some generic signal-free area
        rms = np.nanstd(It[np.logical_and(phase>0.2, phase<0.3),:])
        # Shape of weights is (phasebin, frequency)
    # RMS drops by the sqrt of the number of samples, if everything is equally weighted
    # But since the weights are fractional, it only drops by the sqrt of the sum of the weights
        rms /= np.sqrt(np.nansum(weights, axis=0))
        rms_arr = rms*np.ones(len(I_pulse))

        fig = plt.figure(figsize=(8*cm,8*cm))
        ax = fig.add_subplot(111)
        ax.scatter(freqs_a, 1000*I_pulse, color=color['I'], alpha=0.8, marker='.', s=4, lw=0.5, zorder=10, label="Stokes I")
        ax.errorbar(freqs_a, 1000*I_pulse, yerr=1000*rms_arr, color=color['I'], alpha=0.8, elinewidth=0.5, lw=0, zorder=5)
        ax.axhline(np.nanmean(1000*I_pulse), color=color['I'], lw=0.5)
        ax.scatter(freqs_a, 1000*Q_pulse, color=color['Q'], alpha=0.8, marker='.', s=4, lw=0.5, zorder=10, label="Stokes Q")
        ax.errorbar(freqs_a, 1000*Q_pulse, yerr=1000*rms_arr, color=color['Q'], alpha=0.8, elinewidth=0.5, lw=0, zorder=5)
        ax.axhline(np.nanmean(1000*Q_pulse), color=color['Q'], lw=0.5)
        ax.scatter(freqs_a, 1000*U_pulse, color=color['U'], alpha=0.8, marker='.', s=4, lw=0.5, zorder=10, label="Stokes U")
        ax.errorbar(freqs_a, 1000*U_pulse, yerr=1000*rms_arr, color=color['U'], alpha=0.8, elinewidth=0.5, lw=0, zorder=5)
        ax.axhline(np.nanmean(1000*U_pulse), color=color['U'], lw=0.5)
        ax.scatter(freqs_a, 1000*V_pulse, color=color['V'], alpha=0.7, marker='.', s=4, lw=0.5, zorder=11, label="Stokes V")
        ax.errorbar(freqs_a, 1000*V_pulse, yerr=1000*rms_arr, color=color['V'], alpha=0.7, elinewidth=0.5, lw=0, zorder=6)
        ax.axhline(np.nanmean(1000*V_pulse), color=color['V'], lw=0.5)
        ax.set_ylabel("Weighted brightness (mJy)")
        ax.set_xlabel("Frequency / GHz")
        ax.set_ylim(-20, 20)
        ax.legend()
        fig.savefig("EMU_weighted_Stokes_spectra.pdf", bbox_inches="tight")
        fig.savefig("EMU_weighted_Stokes_spectra.png", bbox_inches="tight")

        # [freq_Hz, I_Jy, Q_Jy, U_Jy, dI_Jy, dQ_Jy, dU_Jy]
    # Errors are ... the RMS of a normal bin, divided by the sqrt of the number of bins used -- but not all were fully used, some had less than unity weight, so it goes up by the difference between the weights and unity weighting -- i.e. only goes down by np.sqrt(sum of weights), where weights are normalised to 1
    #  But sqrt makes the errors too large
        # Just some random noise-dominated area -- use the light curve not the dynamic spectrum because that is closer to what we're calculating
        
        out = np.array([freqs_a[~np.isnan(I_pulse)]*1.e9, I_pulse[~np.isnan(I_pulse)], Q_pulse[~np.isnan(I_pulse)], U_pulse[~np.isnan(I_pulse)], rms_arr[~np.isnan(I_pulse)], rms_arr[~np.isnan(I_pulse)], rms_arr[~np.isnan(I_pulse)]])

        np.savetxt("EMU_folded_IQU_spectrum.txt", out.T)

    if makePhaseBin is True:
        # Try different phase binning to see if we can obtain a polarisation angle sweep constraint (we can't)
        step = 0.01
        phase_starts = np.arange(phase_start, phase_end+step, step)
        for phase_start in phase_starts:
            phase_end = phase_start + step
            ind = np.logical_and(phase>phase_start, phase<phase_end)
            Ilc_pulse = ilc_a[ind]
            weights = np.tile(Ilc_pulse, (Qt.shape[1],1)).T
            weights[weights<0] = 0.0
        # Normalise the weights to 1 as this will be useful later
            weights /= np.nanmax(weights)

            I_pulse = np.nansum(It[ind,:]*weights, axis=0)/np.nansum(weights[:,0])
            Q_pulse = np.nansum(Qt[ind,:]*weights, axis=0)/np.nansum(weights[:,0])
            U_pulse = np.nansum(Ut[ind,:]*weights, axis=0)/np.nansum(weights[:,0])
            V_pulse = np.nansum(Vt[ind,:]*weights, axis=0)/np.nansum(weights[:,0])
        # Removes RFI-flagged area
            I_pulse[I_pulse==0.0] = np.nan
            Q_pulse[Q_pulse==0.0] = np.nan
            U_pulse[U_pulse==0.0] = np.nan
            V_pulse[V_pulse==0.0] = np.nan

            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot(111)
            ax.scatter(freqs_a, 1000*I_pulse, color=color['I'], alpha=0.8, label="I")
            ax.axhline(np.nanmean(1000*I_pulse), color=color['I'])
            ax.scatter(freqs_a, 1000*Q_pulse, color=color['Q'], alpha=0.8, label="Q")
            ax.axhline(np.nanmean(1000*Q_pulse), color=color['Q'])
            ax.scatter(freqs_a, 1000*U_pulse, color=color['U'], alpha=0.8, label="U")
            ax.axhline(np.nanmean(1000*U_pulse), color=color['U'])
            ax.scatter(freqs_a, 1000*V_pulse, color=color['V'], alpha=0.8, label="V")
            ax.axhline(np.nanmean(1000*V_pulse), color=color['V'])
            ax.set_ylabel("Weighted brightness (mJy)")
            ax.set_xlabel("Frequency / GHz")
            ax.legend()
            fig.savefig(f"weighted_EMU_Stokes_phasebin{phase_start}.png", bbox_inches="tight")

  # I can reuse my rms_array from earlier
            
            out = np.array([freqs_a[~np.isnan(I_pulse)]*1.e9, I_pulse[~np.isnan(I_pulse)], Q_pulse[~np.isnan(I_pulse)], U_pulse[~np.isnan(I_pulse)], rms_arr[~np.isnan(I_pulse)], rms_arr[~np.isnan(I_pulse)], rms_arr[~np.isnan(I_pulse)]])

            np.savetxt(f"EMU_folded_IQU_spectrum_phasebin{phase_start:1.2f}.txt", out.T)

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
    sbids = []
    rmss = []
    freqs = []
    obslengths = [] # in minutes
    beams = []
    for pkl in pkls:
        arr = np.load(pkl, allow_pickle=True)
        freqs.append(arr["FREQS"][int(len(arr["FREQS"])/2)]/1.e6)
        sbids.append(pkl.split("SB")[1][0:5].replace("_",""))
        beams.append(pkl.split("beam")[1][0:2])
# For center of observation
#        mjds.append(arr["TIMES"][int(len(arr["TIMES"])/2)]/(24*3600))
# For start of observation
        mjds.append(arr["TIMES"][0]/(24*3600))
        obslengths.append((arr["TIMES"][-1]-arr["TIMES"][0])/60) # Because they want it in minutes
        rmss.append(np.nanstd(np.nanmean(np.real((arr["DS"][:,:,XX]+arr["DS"][:,:,YY])),axis=1)))

# TODO Combine beams at the point where you're measuring the RMS, not just later for the table
    sbids = np.array(sbids)
# Cut down to unique SBIDs and combine beams
#    uniq_sbids = np.unique(sbids)

    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    ax.scatter(mjds, 1000*np.array(rmss), marker='v', color='black', label='1-sigma RMS\n(10s time resolution)')
    ax.set_xlabel("MJD")
    ax.set_ylabel("Flux density (mJy)")
    ax.scatter(T0, 17., color='red', marker='*', label='EMU Pilot detection\n(brightest pulse)')
    ax.errorbar(T0, 17., yerr=1, color='red')
    ax.legend(loc=1)
    fig.savefig("Non-detections_ASKAP.png", bbox_inches="tight")

# Make a LaTeX table output for them
    times = Time(mjds, format='mjd', scale='utc')
    datestrs = np.array([justdate(t) for t in times])
    hhmmstrs = np.array([hhmm(t) for t in times])

# Sort everything by MJD
    ind = np.argsort(mjds)
    datestrs = datestrs[ind]
    hhmmstrs = hhmmstrs[ind]
    sbids = sbids[ind]
    beams = np.array(beams)[ind]
    freqs = np.array(freqs)[ind]
    obslengths = np.array(obslengths)[ind]

    with open("obs_table.tex", "w") as f:
        for i in range(0, len(datestrs)-1):
            date, time, sbid, freq, obslength, beam = datestrs[i], hhmmstrs[i], sbids[i], freqs[i], obslengths[i], beams[i]
# Almost every observation has two beam measurements
# TODO fix for the one that doesn't
            if sbid == sbids[i+1]:
                beams_to_print = beams[i:i+2]
                sorted_beams = beams_to_print[np.argsort(beams_to_print.astype('int'))]
                f.write(f"{date} & {time} & ASKAP & SB{sbid} beams:{sorted_beams[0]},{sorted_beams[1]} & {freq:4.0f} & {obslength:2.0f} \\\\\n")

# Look at the accuracy of the ephemeris by plotting each pulse

trange = times_a / (24*3600)
# Note the addition of 1 period here
phase = np.mod(trange - T0 + P, 2*P)/(2*P)
# and half a turn of phase here, +1 to get away from +/-zero where everything gets labelled zero
pulsenums = (2.5 + pulsenum(trange)).astype('int')
# Makes it easier to see the lightcurves
minpulsenum = pulsenums[0]
maxpulsenum = pulsenums[-1]

# Now include MeerKAT data
trange_m = times_m / (24*3600)
phase_m = np.mod(trange_m - T0 + P, 2*P)/(2*P)
# and half a turn of phase here, +1 to get away from +/-zero where everything gets labelled zero
pulsenums_m = (2.5 + pulsenum(trange_m)).astype('int')
num_extra_mkt_panels = len(np.unique(pulsenums_m))

num_panels = maxpulsenum - minpulsenum + num_extra_mkt_panels

ind = 1
fig = plt.figure(figsize=(5,20))
#Plot the EMU data
for n in range(minpulsenum, maxpulsenum):
    ax = fig.add_subplot(num_panels,1,ind)
    ax.plot(phase[pulsenums==n], 1000*ilc_a[pulsenums==n], color=color["I"], alpha=0.8, label=f"{n}")
#ax.set_ylabel("brightness (mJy/beam)")
#ax.set_xlabel("time / s")
    ax.set_ylim(-3, 20)
    ax.set_xlim(-0.05, 1.05)
    ax.axvline(0.05, alpha=0.4, color='orange')
    ax.axvline(0.52, alpha=0.8, color='orange')
    ax.legend()
    ax.tick_params(axis='x', labelbottom=False)
    ind += 1
# Plot the MeerKAT data
for n in np.unique(pulsenums_m):
    ax = fig.add_subplot(num_panels,1,ind)
    ax.plot(phase_m[pulsenums_m==n], 1000*ilc_m[pulsenums_m==n], color='purple', alpha=0.8, label=f"{n}")
    ax.set_ylim(-3, 20)
    ax.set_xlim(-0.05, 1.05)
    ax.axvline(0.05, alpha=0.4, color='orange')
    ax.axvline(0.52, alpha=0.8, color='orange')
    ax.legend()
    if ind != (num_panels):
        ax.tick_params(axis='x', labelbottom=False)
    ind += 1
ax.set_xlabel("Phase")
#ax.set_xlim(tstart, tend)
#ax.legend(loc=1)
fig.savefig("Ephemeris_lightcurve.png", bbox_inches="tight")

