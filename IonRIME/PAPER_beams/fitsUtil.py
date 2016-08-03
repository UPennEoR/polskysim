"""
FITS Utility Functions
"""

import pyfits as pf
import pywcs

def readFITS(fn,hdr=False,axis=False):
    """Read a FITS image file and returns a numpy array"""
    hdulist=pf.open(fn)
    im=hdulist[0].data
    #image data format: [frequency, polarization, x, y]
    hdulist.close()
    rhdr=None
    raxis=None
    if hdr: rhdr=getFITSInfo(fn)
    if axis: raxis=getAxisInfo(fn)
    if hdr and axis: return im,rhdr,raxis
    elif hdr: return im,rhdr
    elif axis: return im,raxis
    else: return im

def getFITSInfo(fn):
    """Parse the FITS header for pointing and pixel size information
    return [RA,DEC], pixel resolution, pixel of [RA,DEC]
    generates a WCS instance for converting between sky and pixels
    """
    hdulist=pf.open(fn)
    hdr=hdulist[0].header
    #CTYPE1: RA---[PROJ], projection SIN/TAN/ARC
    #CRVAL1: reference RA position in degrees
    #CRPIX1: location of reference pixel
    #CDELT1: delta RA/pixel size in degrees
    #CTYPE2: DEC--[PROJ], projection SIN/TAN/ARC
    #CRVAL2: reference DEC position in degrees
    #CRPIX2: location of reference pixel
    #CDELT2: delta DEC/pixel size in degrees
    #LATPOL: latitude of array centre
    ra=hdr['CRVAL1']
    dra=hdr['CDELT1']
    raPix=hdr['CRPIX1']
    dec=hdr['CRVAL2']
    ddec=hdr['CDELT2']
    decPix=hdr['CRPIX2']
    latPole=hdr.get('LATPOLE',hdr['CRVAL2'])
    #CTYPE3: FREQ
    #CRVAL3: centre bandwidth in Hz
    #CRPIX3: number of freq channels
    #CDELT3: bandwith of channel
    centreFreq=hdr['CRVAL3']
    nchans=int(hdr['CRPIX3'])
    bw=hdr['CDELT3']
    #CTYPE4: STOKES
    #CRVAL4: number of Stokes values
    #CRPIX4: Initial Stokes label (1:I 2:Q 3:U 4:V)
    #CDELT4: Stokes label step size
    nstokes=hdr['CRVAL4']
    stokes0=hdr['CRPIX4']
    #Generate a WCS structure, using the normal method creates errors due to header formating
    wcs = pywcs.WCS(naxis=2)
    wcs.wcs.crval = [ra,dec]
    wcs.wcs.crpix = [raPix,decPix]
    wcs.wcs.cdelt = [dra,ddec]
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]

    hdulist.close()
    return {'hdr':hdr,'ra':ra,'dec':dec,'dra':dra,'ddec':ddec,'raPix':raPix,'decPix':decPix, 'latPole':latPole, 'wcs':wcs,'centreFreq':centreFreq,'nchans':nchans,'bw':bw,'nstokes':nstokes,'stokes0':stokes0}

def getAxisInfo(fn):
    hdulist=pf.open(fn)
    hdr=hdulist[0].header
    axisInfo=[]
    cIdx=1
    while ('CTYPE%i'%cIdx) in hdr:
        axisInfo.append(hdr.get('CTYPE%i'%cIdx))
        cIdx+=1
    hdulist.close()
    return axisInfo[::-1] #I think the axis reversal is due to a C/FORTRAN array indexing issue

def getKeyword(fn,kw):
    """Return the value of a keyword in a FITS file header
    """
    hdulist=pf.open(fn)
    hdr=hdulist[0].header
    val=hdr[kw.upper()]
    hdulist.close()
    return val

