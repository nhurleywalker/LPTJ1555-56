#!/usr/bin/env python

import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
from astroplan import Observer
from astropy import units as u

# 1. Define your observing location (e.g., Perth, WA)
#perth_location = EarthLocation(lat=-31.9523*u.deg, lon=115.8613*u.deg, height=0*u.m)
#perth_observer = Observer(location=perth_location, name='Perth')

mkt = Observer.at_site("salt")
askap = Observer.at_site("mwa")

# 2. Define the exact date and time of the observation
obs_time = Time((5.15927e9+1900)/(24*3600), format='mjd', scale='utc')

# 3. Define the celestial target coordinates (Right Ascension, Declination)
target = SkyCoord(ra= 238.93204167*u.deg, dec= -56.51733333*u.deg, frame='fk5')

# 4. Calculate the parallactic angle
# Note: The output is an astropy Angle object, retrieved in degrees
pa_mkt = mkt.parallactic_angle(obs_time, target)
pa_askap = askap.parallactic_angle(obs_time, target)

print(f"MeerKAT Parallactic Angle: {pa_mkt.deg:.2f} degrees")
print(f"ASKAP Parallactic Angle: {pa_askap.deg:.2f} degrees")
