import glob
from pathlib import Path
import os
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u
 
import numpy as np

from primary_beams import GaussianPB, get_beam_pos, MKCosBeam, get_beam_pos_mkt

#logger = logging.getLogger(__name__)
cm = 1/2.54

# All L-band data so it's not necessary
# Spectral index to use for correcting to same flux density scale
SPECIND = -1.5
# Which frequency to correct to
CALFREQ = 1.e9 # (1 GHz)

j1555_coord = SkyCoord(
    ra="15h55m43.69s",
    dec="-56d31m02.4s",
    unit="hourangle,deg",
    frame="fk5",
)

images = glob.glob("images/1*.fits")
vimages = glob.glob("images/StokesV/1*.fits")
images = images + vimages
for image in images:
    cbid = int(Path(image).stem[0:10])
    print(cbid)
    hdu = fits.open(image)
    freq = hdu[0].header["CRVAL3"]
    pb_val = MKCosBeam(get_beam_pos_mkt(cbid).separation(j1555_coord).deg, freq)
    hdu[0].data = hdu[0].data / pb_val
    hdu.writeto(image.replace(".fits", "_pbcorr.fits"), overwrite=False)
