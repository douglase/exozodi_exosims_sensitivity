import numpy as np
import matplotlib.pyplot as plt
import resample2D
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

def photCorrPC(nobs,nfr,t,g):
    """
    Photon counting to 3rd order correction, typical parameters are
    nobs =
    t = 5*100 e-/frame
    g = 6000 e-/photoelectron
    nbr = lam_br * Nfr * eth * ecl

    This should return the mean expected photon rate per-pixel


    """

    lam_est = -np.log(1-(nobs/nfr)*np.exp(t/g))
    plt.figure()
    plt.title('First Photon Counted Frame')
    plt.imshow(lam_est,vmin=0)
    plt.colorbar()
    plt.show()
    lam_est -= deltaLam(lam_est,t,g,nfr,nobs)
    plt.figure()
    plt.title('First Photometric Correction')
    plt.imshow(lam_est,vmin=0)
    plt.colorbar()
    plt.show()
    lam_est -= deltaLam(lam_est,t,g,nfr,nobs)
    plt.figure()
    plt.title('Second Photometric Correction')
    plt.imshow(lam_est,vmin=0)
    plt.colorbar()
    # plt.show()
    return lam_est

def deltaLam(lam,t,g,nfr,nobs):
    """

    Parameters
    ----------
    lam : float
        mean expected rate per pixel per frame from photCorrPC
    t : float
        threshold chosen for photon counting
    g : float
        EM Gain
    nfr : float
        number of frames
    nobs : float
        sum of counts across all frames after thresholding

    From B. Nemati 7 Nov 2020

    Returns
    -------
    dlam

    """

    epsThr_a = np.exp(-t/g)*(t**2 * lam**2 + 2*g*t*lam*(3+lam) + 2*g**2 *(6+3*lam + lam**2))
    epsThr_b = (2*g**2 *(6+3*lam + lam**2))
    epsThr3  = epsThr_a/epsThr_b

    epsCL = (1-np.exp(lam))/lam
    func  = lam*nfr*epsThr3*epsCL - nobs

    dfdlam_a = (1/(2*g**2 *(6+3*lam+lam**2)**2))
    dfdlam_b = np.exp(-t/g - lam)*nfr
    dfdlam_c = 2*g**2 *(6+3*lam+lam**2)**2 + t**2 *lam*(-12 + 3*lam +3*lam**2 + lam**3 +3*np.exp(lam)*(4+lam))
    dfdlam_d = 2*g*t*(-18 +6*lam +15*lam**2 + 6*lam**3 +lam**4 + 6*np.exp(lam)*(3+2*lam))

    dfdlam   = dfdlam_a*dfdlam_b*(dfdlam_c+dfdlam_d)

    dlam = func/dfdlam

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
        scalar = 250/10000#4e13
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
            data_t = threshold(5,100,data[mind:mand,:,:]) # e- to photoelectron
            data_s = np.sum(data_t,axis=0)
            data_c = photCorrPC(data_s,mand-mind+1,500,6000)/1 # photoelectron/sec
            data_c = data_c-np.mean(data_c[0:5,0:5]) # SNR/sec
            
            # Filter invalid values for NMF
            data_c[data_c <= 0] = 1e-20
            data_c[data_c == np.nan] = 1e-20
            data_c[data_c == np.inf] = 1e-20
            

        elif mode == 'Analog':

            data_c = data[mind:mand]/(1*6000) # photon/sec
            data_c -= np.mean(data_c[:,0:5,0:5])
            data_c = np.sum(data_c,axis=0)
            data_c = data_c # kind of a fudge factor

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
        data_c = data[mind:mand]/(60*10)
        data_c -= np.mean(data_c[:,0:5,0:5])

        data_c[data_c <= 0] = 1e-20
        data_c[data_c == np.nan] = 1e-20
        

    return data_c
