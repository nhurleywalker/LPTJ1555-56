import glob
from pathlib import Path
import os
from astropy.io import fits
 
import numpy as np

images = glob.glob("images/*pbcorr.fits")
vimages = glob.glob("images/StokesV/1*pbcorr.fits")
images = images + vimages
for image in images:
    cbid = int(Path(image).stem[0:10])
    print(cbid)
    hdu = fits.open(image)
    bkg = fits.open(image.replace(".fits", "_bkg.fits"))
    hdu[0].data = hdu[0].data - bkg[0].data
    hdu.writeto(image.replace(".fits", "_bkgsub.fits"), overwrite=False)
