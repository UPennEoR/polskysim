import numpy as np, healpy as hp, os, sys, optparse, pyfits
# import matplotlib.pyplot as plt
from datetime import datetime as datetime
"""
Making maps from angular power spectra
"""

def mk_map_onescale(nside,lmin=0,alpha=-2.17,smooth=1.):
    """
    One power law model -> random field realization
    """
    lmax = 3*nside - 1
    ll = np.array(range(lmin,lmax,1))
    Cl = np.power(ll,alpha)
    Cl[0] = 0. #zero mean

    Q = hp.synfast(Cl,nside,lmax=lmax)
    Qsm = hp.smoothing(Q,fwhm=np.radians(smooth))

    return Qsm

def mk_map_SCK(nside,nu,A,beta,alpha,lmin=0,l_f=1000,nu_f=130,smooth=1.):
    """
    Santos Cooray and Knox (2005) diffuse foreground model -> random field realization

    #Fiducial parameters at l_f=1000, nu_f=130 MHz
    #
    # Src (i)                        A (mK^2)  beta  \bar{alpha}  xi
    #
    #Extragalactic point sources     57        1.1   2.07         1.0
    #Extragalactic bremstrahlung     0.014     1.0   2.10         35
    #Galactic synchrotron            700       2.4   2.80         4.0
    #Galactic bremstrahlung          0.088     3.0   2.15         35
    """
    lmax = 3*nside - 1
    ll = np.array(range(lmin,lmax,1))
    Cl = A*np.power(float(l_f)/ll,beta)*np.power(nu_f/nu,2*alpha)
    Cl[0] = 0. #zero mean

    Q = hp.synfast(Cl,nside,lmax=lmax)
    Qsm = hp.smoothing(Q,fwhm=np.radians(smooth))

    return Qsm

def mk_map_GSM(f=150.,frange=None,nbins=None,write2fits=True):
    """
    Make a Stokes I map/cube using pyGSM
    Can also provide frange=[lo,hi] and nbins to get an image cube (ignores f)
    Both f arguments need to be  in MHz
    """
    from pygsm import GlobalSkyModel
    gsm = GlobalSkyModel()

    if frange==None:
        gsm = gsm.generate(f)
        if write2fits: gsm.write_fits("gsm_%sMHz.fits"%str(int(f)))
        M = gsm
    else:
        flo,fhi=frange[0],frange[1]
        try: freqs = np.linspace(flo,fhi,nbins)
        except:
            print 'Did you provide "nbins"?'
            return None
        map_cube = gsm.generate(freqs)
        #if write2fits: map_cube.write_fits("gsm_cube_%s-%sMHz"%(str(int(flo)), str(int(fhi))))
        M = np.swapaxes(map_cube,0,1) #conform to other map making routines
    return M

def mk_map_gsm2016_mod(freqs, nside_out):
    import sys
    sys.path.append('/data4/paper/zionos/polskysim')
    import gsm2016_mod
    import astropy.coordinates as coord
    import astropy.units as units

    def harmonic_ud_grade(m, nside_in, nside_out):
        """
        Decompose a map at a resolution nside_in into spherical harmonic components
        and then resynthesize the map at nside_out.
        """
        lmax = 3 * nside_in - 1
        alm = hp.map2alm(m, lmax=lmax)
        return hp.alm2map(alm, nside_out, lmax=lmax, verbose=False)

    def rotate_healpix_map(m, R):
        """
        Performs a scalar rotation of the map relative to the Healpix coordinate
        frame by interpolating the map at the coordinates of new coordinate frame.
        """
        npix = len(m)
        nside = hp.npix2nside(npix)
        hpxidx = np.arange(npix)
        c, a = hp.pix2ang(nside, hpxidx)
        t, p = rotate_sphr_coords(R, c, a)
        return hp.get_interp_val(m, t, p)

    nside_in = 64
    npix_in = hp.nside2npix(nside_in)
    nfreq = len(freqs)

    npix_out = hp.nside2npix(nside_out)
    I_gal = np.zeros((nfreq, npix_in))
    for fi, f in enumerate(freqs):
        I_gal[fi] = gsm2016_mod.get_gsm_map_lowres(f/1e3) # freqs is in MHz, gsm2016 generates

    x_c = np.array([1.,0,0]) # unit vectors to be transformed by astropy
    y_c = np.array([0,1.,0])
    z_c = np.array([0,0,1.])

    # The GSM is given in galactic coordinates. We will rotate it to J2000 equatorial coordinates.
    axes_icrs = coord.SkyCoord(x=x_c, y=y_c, z=z_c, frame='icrs', representation='cartesian')
    axes_gal = axes_icrs.transform_to('galactic')
    axes_gal.representation = 'cartesian'

    R = np.array(axes_gal.cartesian.xyz) # The 3D rotation matrix that defines the coordinate transformation.

    npix_out = hp.nside2npix(nside_out)
    I = np.zeros((nfreq, npix_out))

    for i in range(nfreq):
        I[i] = rotate_healpix_map(I_gal[i], R)
        if nside_out != nside_in:
            I[i] = harmonic_ud_grade(I_gal[i], nside_in, nside_out)

    return I

def mk_fg_cube(onescale=True, pfrac=0.002, flo=100., fhi=200., nbins=203, alo=-2.7, ahi=-2.3, alpha_map=None, raw_map=None, intermediates=True, save_cubes=True, verbose=False):
    """
    Make a Stokes IQUV cube.

    pfrac: fraction Q,U/I
    flo: lowest frequency
    fhi: highest frequency
    nbins: number of frequency bins
    alo: lowest spectral index
    ahi: highest spectral index

    alpha_map: a healpix map of spectral indices. Useful if splitting-up frequency runs to avoid memory errors

    onescale: I'm not feeling smart enough to make an argument interpreter for
    different "mk_map" cases. If onescale=True, diffuse emission is modelled using a single power law. Otherwise, we use the SCK model. XXX currently only Galactic Synchrotron

    intermediates: Save Q and U realizations @ flo MHz, and spectral index maps



    I'm making the :
        - large assumption that Q and U trace the spectral distribution of
          diffuse Stokes I power at a fixed polarization fraction.
        - assumption that spectral indicies are randomly distributed on scales > 3deg
        - small assumption that Stokes V power is vanishingly small.

    TODO:
        - map Q<->U using polarization angle map
        - maybe move away from assumption that Galactic Sync. is dominant source of polarization?
        - could use a "correlation length" ala Shaw et al. 2014 to get more realistic spectral index distribution

    """
    nu = np.linspace(flo,fhi,num=nbins, endpoint=True)
    nside=512 #to match GSM nside
    npix = hp.nside2npix(nside)
    ipix = np.arange(npix)

    #Spectral indices -2.7 to -2.3 are the 2sigma range of Rogers & Bowman '08 (in K)
    if alpha_map==None:
        alpha = np.random.uniform(low=alo,high=ahi,size=npix)
        alpha = hp.smoothing(alpha,fwhm=np.radians(3.))
        if intermediates:
            #XXX bug -- need to init date
            np.savez('alpha_%i-%i-%i.npz'%(date[0],date[1],date[2]),maps=alpha)
    else:
        if verbose: '    Loading %s'%alpha_map
        alpha = np.load(alpha_map)['maps']

    if verbose: print 'Creating Stokes I spectral cube with PyGSM...'
    I = np.transpose(mk_map_GSM(frange=[flo,fhi],nbins=nbins), (-1,0)) #use GSM for Stokes I
    ## I.shape = (nfreq, npix)

    ## eh, this will require more work, might not be necessary
    # freqs = np.linspace(flo,fhi, nbins, endpoint=True)
    # I = mk_map_gsm2016_mod
    if verbose: print 'Stokes I done'

    if raw_map==None:
        if onescale:
            Q0 = mk_map_onescale(512)
            U0 = mk_map_onescale(512) #different realizations of same scaling
            #XXX with a polarization angle map, I could link Q,U instead of having them independent
        else:
            Q0 = mk_map_SCK(512,flo,700,2.4,2.80)
            U0 = mk_map_SCK(512,flo,700,2.4,2.80)
            #XXX as above wrt pol angle, but also this currently assumes we are dominated by Galactic Synchrotron

        _Q0,_U0 = Q0 - Q0.min(),U0 - U0.min()
        #XXX is this OK?
        Q0 = (2*_Q0/_Q0.max()) - 1 #scale to be -1 to 1
        U0 = (2*_U0/_U0.max()) - 1 #scale to be -1 to 1

        if intermediates:
            if verbose: plot_maps([Q0,U0],titles=['raw Q','raw U'])
            np.savez('rawcube_%sMHz.npz'%str(flo),maps=[I[:,0],Q0,U0])

    else:
        if verbose: print '    Loading %s'%raw_map
        raw = np.load(raw_map)['maps']
        #Stokes I is deterministic, Q and U are not
        Q0 = raw[1]
        U0 = raw[2]

    Qmaps,Umaps,Vmaps = np.zeros((len(nu), npix)),np.zeros((len(nu), npix)),np.zeros((len(nu), npix))

    #If only I could take the log! Then this would be vectorizable
    #stoopid Q and U with their non +ve definition
    if verbose: print 'Begin loop over spectral index for frequency scaling'

    # for i in ipix:
    #     Qmaps[i,:] = Q0[i] * np.power(nu/130.,alpha[i]) #XXX BEWARE HARDCODED 130 MHz
    #     Umaps[i,:] = U0[i] * np.power(nu/130.,alpha[i])

    freq_scale = np.power(np.outer(nu, np.ones(npix)) / 130., alpha) #XXX BEWARE HARDCODED 130 MHz
    Qmaps = Q0 * freq_scale
    Umaps = U0 * freq_scale

    date = datetime.now().timetuple()

    if verbose: print 'Frequency scaling done'

    #XXX is this OK?
    #impose polarization fraction as fraction of sky-average Stokes I power per frequency
    Qmaps *= np.outer(np.nanmean(I,axis=1)*pfrac, np.ones(npix))
    Umaps *= np.outer(np.nanmean(I,axis=1)*pfrac, np.ones(npix))

    spols = ['I','Q','U','V']
    cube = [I,Qmaps,Umaps,Vmaps]
    if save_cubes == True:
        for i,m in enumerate(cube):
            N = 'cube_%s_%s-%sMHz_%sbins.npz'%(spols[i],str(flo),str(fhi), str(len(nu)))
            print '    Saving %s'%N
            np.savez(N, maps=m)

    return np.array(cube)

def propOpp(cube=None,flo=100.,fhi=200.,lmax=100,npznamelist=None, save_cubes=True):
    """
    Propogate the Q and U components of an IQUV cube through
    the Oppermann et al. 2012 RM map.

    The cube must be 4 or 2 by npix by nfreq. If 4, the middle two arrays will be
    assumed to be Q & U IN THAT ORDER.
    Or if fromnpz=True, then provide an array of npz names (cube length assumptionsremain)
    """
    ## load the maps
    if npznamelist!=None:
        assert(cube==None)
        nNpz = len(npznamelist)
        assert(nNpz == 2 or nNpz == 4)

        if nNpz == 2:
            Q = np.load(npznamelist[0])['maps']
            U = np.load(npznamelist[1])['maps']
        else:
            Q = np.load(npznamelist[1])['maps']
            U = np.load(npznamelist[2])['maps']
    elif cube!=None:
        Q = cube[1]
        U = cube[2]
    else:
        raise ImplmentationError('No map information provided.')

    ##
    nbins = Q.shape[0]
    nu = np.linspace(flo,fhi,num=nbins)
    lam = 3e8/(nu*1e6)
    lam2 = np.power(lam,2)

    d = pyfits.open('opp2012.fits')
    RM = d[3].data.field(0)
    """
    The RM map is nside=128. Everything else is nside=512.
    We're smoothing on scales larger than the pixellization
    this introduces, so no worries.

    RMmap=hp.ud_grade(RM,nside_out=512)
    hp.mollview(RMmap,title='Oppermann map')

    RMmap=hp.smoothing(RMmap,fwhm=np.radians(1.))

    hp.mollview(RMmap,title='Oppermann map smoothed')
    plt.show()
    """
    #Downsample RM variance in alm space
    #Upsample it in pixellization
    #Is this kosher?
    Qn, Un = [np.zeros((nbins,hp.nside2npix(128))) for i in range(2)]
    for i in range(Q.shape[0]):
        # print Q[:,i].shape
        Qn[i] = hp.alm2map(hp.map2alm(Q[i],lmax=3 * 512 -1),nside=128)
        Un[i] = hp.alm2map(hp.map2alm(U[i],lmax=3*512 - -1),nside=128)

    RMmap = RM

    # RMmap = hp.alm2map(hp.map2alm(RM,lmax=lmax),nside=512)
    Qmaps_rot = np.zeros_like(Qn)
    Umaps_rot = np.zeros_like(Un)

    # phi = np.outer(RMmap,lam2)
    for i in range(nbins):
        phi = RMmap * lam2[i]
        fara_rot = (Qn[i] + 1.j*Un[i])*np.exp(-2.j*phi) #Eq. 9 of Moore et al. 2013
        Qmaps_rot[i] = fara_rot.real
        Umaps_rot[i] = fara_rot.imag

    QU = [Qmaps_rot,Umaps_rot]

    if save_cubes == True:
        print 'Saving Q U rotated'
        np.savez('cube_Qrot_%s-%sMHz.npz'%(str(flo),str(fhi)),maps=Qmaps_rot)
        np.savez('cube_Urot_%s-%sMHz.npz'%(str(flo),str(fhi)),maps=Umaps_rot)

    return np.array(QU)
"""
#MEMORY ERRORS
def concat_cube_npzs(Ilist,Qlist,Ulist):
    #V can just be an array of zeros like the concatenated others
    mI,mQ,mU = [],[],[]
    for i,a in enumerate(Ilist):
        mI.append(np.load(a)['maps'])
        mQ.append(np.load(Qlist[i])['maps'])
        mU.append(np.load(Ulist[i])['maps'])

    Icube = np.concatenate(mI,axis=1)
    Qcube = np.concatenate(mQ,axis=1)
    Ucube = np.concatenate(mU,axis=1)

    return [Icube,Qcube,Ucube]
"""

def plot_maps(maps,titles=None):
    """
    Plots a list of healpix maps in mollweide projection
    """
    s = int(np.ceil(np.sqrt(float(len(maps)))))
    for i,m in enumerate(maps):
        if titles is not None: hp.mollview(m,title=titles[i],sub=(s,s,i+1))
        else: hp.mollview(m,sub=(s,s,i+1))
    plt.show()
