from __future__ import absolute_import, division, print_function
# nav to site packages
import sys
sys.path.append('home/u6/jashcraft/emccd_venvP/lib/python3.8/site-packages') 
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from emccd_detect.emccd_detect import EMCCDDetect, emccd_detect
from PIL import Image
from astropy.io import fits
import scipy

# Import Disk File, gain, and exposure time
diskfile = 'zodipic_10pc03mas._coremasked._HLC.fits'
framegain = 6000
frametime = 5

# meta_path will depend on the directory you installed emccd_detect in
meta_path = '/home/u6/jashcraft/Photon-Counting/os9_hlc/metadata.yaml'
here = os.path.abspath(os.path.dirname('Photon-Counting'))

# Begin file loading
fluxmap = fits.getdata(diskfile).astype(float)*1e14 # scale up from Janskys
speckwmuf = fits.getdata('muf_os9_ccd_images_no_planets.fits').astype(float)
meanspeck = np.mean(speckwmuf[60])*1e-2
print('max fluxmap val = ',max(fluxmap[:,33]))
print('mean speckle val = ',meanspeck)

# Scale to the mean values of the speckles
fluxmap *= meanspeck/max(fluxmap[:,75])

# Put fluxmap in 1024x1024 image section
full_fluxmap = np.zeros((1024, 1024)).astype(float)
full_fluxmap[0:fluxmap.shape[0], 0:fluxmap.shape[1]] = fluxmap

# Option to save fluxmap as text file
# np.savetxt('origflux.txt',fluxmap,delimiter=',')

# Run Nemati's EMCCDDetect detector model, parameters from the ipac website
emccd = EMCCDDetect(
            meta_path=meta_path,
            em_gain=framegain, # gain (e-/photoelectron)
            full_well_image=50000, # Image area full well capacity e-
            dark_current=0.000213888889, # e-/pix/s, taken from 0.76 e-/pix/hr
            cic=0.01, # e-/pix/frame
            read_noise=100, # e-/pix/frame
            bias=10000, # e-
            qe=0.9, 
            cr_rate=0., # cosmic ray rate hits/cm^2/s
            pixel_pitch=13e-6, # distance between centers m
            shot_noise_on=False,
            cic_gain_register=0., 
            numel_gain_register=604, 
            nbits=14
        )
        
# Simulate EMCCD frames from the given input fluxmap - this tends to run for quite some time
for ijk in range(25):
    sim_frame = emccd.sim_sub_frame(full_fluxmap, frametime)
    if ijk == 0:
        diskar = sim_frame[0:200,0:200]
        print('first disk!')
    else:
        diskar = np.dstack([diskar,sim_frame[0:200,0:200]])
        print('disk ',ijk,' counted')

# write stack to fits file
hdu = fits.PrimaryHDU(data=diskar)
hdul = fits.HDUList([hdu])
hdul.writeto('zodi_pc_10pc03mas_stack2.fits')