import numpy as np
import matplotlib.pyplot as plt
#import resample2D
from astropy.io import fits
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from scipy.ndimage import rotate
from scipy.ndimage import zoom

"""

"""

def threshold(factor,readnoise,frame):
    """
    Binary Threshold
    factor = 5*readnoise
    readnoise = 100 e-
    frame = frame to threshold
    """

    thresh = factor*readnoise

    frame[frame<thresh] = 0
    frame[frame>thresh] = 1

    return frame

def photCorrPC ( nobs, nfr, t, g ) : # Nemati (2020) Appendix
    #     print("nfr", nfr, "t", t, "g", g)
    """
    # nobs = number of observed and counted photons.
    #        "The number of _counts_ of an above-threshold photon, that have been summed up."
    # nfr  = number of frames.
    #        "The number of _frames_ across which the number of observations of above-threshold photons have occured."
    # t    = threshold.
    #        "The threshold of electrons that determines whether a pixel is considered to have recorded a _countable_ photon."
    # g    = gain.
    #        "Electro Multiplying Gain = How many electrons are mobilized by one photon, using this version of CCD pixel..."
    #        "Therefore, how many electrons need to be counted to signify that a countable photon has occured."
            # https://www.photometrics.com/learn/imaging-topics/on-chip-multiplication-gain
            # "The level of EM gain can be controlled by either increasing or decreasing the voltage;
            # the gain is exponentially proportional to the voltage. 
            # Multiplying the signal above the read noise of the output amplifier enables ultra-low-light detection at high operation speeds. 
            # EM gain can exceed 1000x."
    """
    lam_est = -np.log ( 1 - ( nobs / nfr )   *   np.exp ( t / g ) )
    lam_est = lam_est - deltaLam ( lam_est, t, g, nfr, nobs )
    lam_est = lam_est - deltaLam ( lam_est, t, g, nfr, nobs )
    return lam_est
    
def deltaLam ( lam, t, g, nfr, nobs) : # Nemati 2020
    """
    # lam  = AKA "lam_est" in photCorrPC = mean expected rate per pixel per frame
    #        "A value less than one, representing the expected rate that a photon will hit that pixel in that frame inside the frame's exposure time."
    # t    = threshold [measured in electrons] chosen for photon counting
    # g    = EM gain
    # nfr  = number of frames.
    # nobs = number of observed and counted photons.
    """
    ft1 = lam**2            # frequent term #1
    ft2 = 6 + 3 * lam + ft1 # frequent term #2
    ft3 = 2 * g**2 * ft2    # frequent term #3 ; ft3 = 2 * g**2 * ( 6 + 3 * lam + ft1 )
    
    # Epsilon PC = Epsilon Photon Counting = Thresholding Efficiency
    epsThr3 = np.exp( - t / g ) * ( t**2 * ft1 + 2 * g * t * lam * ( 3 + lam ) + ft3 ) / ft3 

    # Epsilon Coincidence Loss = Coincidence Loss (Efficiency)
    epsCL = ( 1 - np.exp ( - lam ) ) / lam
    
    func = lam * nfr * epsThr3 * epsCL - nobs
    
    # dfdlam
    dfdlam_1tN  = np.exp ( - t / g - lam) * nfr # First term numerator
    dfdlam_1tD  = 2 * g**2 * ft2**2             # 1t denominator ; { dfdlam_1tD  = 2 * g**2 * ( 6 + 3 * lam * ft1 )**2 } 
    dfdlam_2ts1 = dfdlam_1tD                 # 2t, 1 summand ; { dfdlam_2ts1 = 2 * g**2 * ( 6 + 3 * lam * ft1 )**2 }
    #dfdlam_2ts2 = t**2 * lam * ( -12 + 3 * lam + 3 * ft1 + lam**3 + 3 * np.exp ( lam ) * (4 + lam) ) # 2t, 2s
    dfdlam_2ts2 = t**2 * lam * ( -12 + 3 * lam + 3 * ft1 + ft1*lam + 3 * np.exp ( lam ) * (4 + lam) ) # 2t, 2s
    #dfdlam_2ts3 = 2 * g * t * ( -18 + 6 * lam + 15 * ft1 + 6 * lam**3 + lam**4 + 6 * np.exp ( lam ) * ( 3 + 2 * lam ) ) # 2t, 3s
    dfdlam_2ts3 = 2 * g * t * ( -18 + 6 * lam + 15 * ft1 + 6 * ft1*lam + ft1**2 + 6 * np.exp ( lam ) * ( 3 + 2 * lam ) ) # 2t, 3s
    dfdlam      = dfdlam_1tN * dfdlam_1tD * ( dfdlam_2ts1 + dfdlam_2ts2 + dfdlam_2ts3 )   
    
    dlam = func / dfdlam
    
#     print("dlam",dlam)
    return dlam

def processcube(data,ID,diskfile=None,mode=None):
    
    

    vmin = 0
    vmax = 1e3

    if ID == 1:
        mind = 0
        mand = 59

    elif ID == 2:
        mind = 60
        mand = 1079

    elif ID == 3:
        mind = 1080
        mand = 2339

    elif ID == 4:
        mind = 2340
        mand = 3601

    elif ID == 5:
        mind = 3602
        mand = 4860

    elif ID == 6:
        mind = 4861
        mand = 4925

    elif ID == 7:
        mind = 4926
        mand = 4985

    elif ID == 8:
        mind = 4986
        mand = 6005

    elif ID == 9:
        mind = 6006
        mand = 7265

    elif ID == 10:
        mind = 7626
        mand = 8525

    elif ID == 11:
        mind = 8526
        mand = 9785

    elif ID == 12:
        mind = 9786
        mand = 9850

    elif ID == 13:
        mind = 9851
        mand = 9910

    elif ID == 14:
        mind = 9911
        mand = 10930

    elif ID == 15:
        mind = 10931
        mand = 12190

    elif ID == 16:
        mind = 12191
        mand = 13450

    elif ID == 17:
        mind = 13451
        mand = 14710

    elif ID == 18:
        mind = 14711
        mand = 14735
        
    x = np.linspace(-1,1,len(data[0,:,:]))
    y = np.linspace(-1,1,len(data[0,:,:]))
    x,y = np.meshgrid(x,y)
    r = np.sqrt(x**2 + y**2)

    maskarray = (r<=0.75) & (r>= 0.2)

    if diskfile is not None:
        # Add debris disks
        disk = fits.getdata(diskfile).astype(float)
        print('disk array shape = ',disk.shape)
    
        # Resample the disk to fit the HLC resolution - Noisy
        box = np.zeros([67,67,len(disk[0,0,:])])
        box[int(67/2)-24:int(67/2)+24,int(67/2)-24:int(67/2)+24] = disk

        # Noiseless
#         box = disk

        # scale to ansay of disk
        scalar = 300/10000#4e13
#         scalar *= 2/1.85 # for face-on disk
#         scalar *= 1/50000 # for noiseless
#         if mode == 'Analog':
#             scalar *= 1.5

        # Case for +11
        if ID in [3,5,8,10,15,17]:
            
            rotdisk = rotate(scalar*box,-22,reshape=False)
            rotdisk[rotdisk == np.NaN] = 1e-12
            rotdisk[rotdisk <= 0] = 1e-12
            
            for dlen in range(mand-mind):
                choice = np.random.randint(low=0,high=len(disk)+1)
                data[mind+dlen,:,:] += rotdisk[:,:,choice]

        # Case for -11
        elif ID in [2,4,9,11,14,16]:
            
            for dlen in range(mand-mind):
                choice = np.random.randint(low=0,high=len(disk)+1)
                data[mind+dlen,:,:] += scalar*box[:,:,choice]

    if ID in [2,3,4,5,8,9,10,11,14,15,16,17]:

        if mode == 'Photon-Counting':

            # Now Photon Count
            data_t = threshold(5,100,data[mind:mand,:,:]) # e- to photon
            data_s = np.sum(data_t,axis=0) # photons in an observation
            data_c = photCorrPC(data_s,mand-mind+1,500,6000)/5 # photon/pix*s
            data_c -= np.mean(data_c[0:5,0:5]) # photon/pix*s
            
            # Filter invalid values for NMF
            data_c[data_c <= 0] = 1e-20
            data_c[data_c == np.nan] = 1e-20
            data_c[data_c == np.inf] = 1e-20
            

        elif mode == 'Analog':
            
            data_c = np.sum(data[mind:mand,:,:],axis=0)
            data_c = data_c/(5*6000*(mand-mind))#data[mind:mand]/(1*6000) # photon/pix*sec
            data_c -= np.mean(data_c[0:5,0:5])

            # Filter invalid values for NMF
            data_c[data_c <= 0] = 1e-20
            data_c[data_c == np.nan] = 1e-20
            data_c[data_c == np.inf] = 1e-20

        plt.figure()
        plt.title('{} Data'.format(mode))
        plt.imshow(data_c,vmin=vmin)
        plt.colorbar()
        plt.show()

    # Case for Reference
    elif ID in [1,6,7,12,13,18]:
        data_c = data[mind:mand]/(60*100)
        data_c -= np.mean(data_c[:,0:5,0:5])

        data_c[data_c <= 0] = 1e-20
        data_c[data_c == np.nan] = 1e-20
        

    return data_c


