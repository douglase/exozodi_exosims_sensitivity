import numpy as  np
import samplerate
import matplotlib.pyplot as plt
from astropy.io import fits

def resamp2D(array,ratio):
	
	ydim = 200
	
	box = []
	
	for yind in range(200):
	
		if yind == 0:
			box = samplerate.resample(array[yind,:],ratio,'linear')
			
		else:
			box = np.vstack((box,samplerate.resample(array[yind,:],ratio,'linear')))
	
	nbox = []
	
	for xind in range(int(ratio*ydim)):
	
		if xind == 0:
			nbox = samplerate.resample(box[:,xind],ratio,'linear')
			
		else:
			nbox = np.vstack((nbox,samplerate.resample(box[:,xind],ratio,'linear')))
		
	return nbox
