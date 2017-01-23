import numpy as np
import healpy as hp
import astropy.coordinates as coord
import astropy.units as units
from astropy.time import Time
import ionRIME_funcs as irf
import os
import sys

class SkyConstructor(object):
    def __init__(self, params):
        self.p = params

        skyGenerators = self.collect_full_skies()

        self.stokes_parameters = skyGenerators[self.p.sky_selection]()

    def StokesParameters(self):
        if any(x == None for x in [self.I,self.Q,self.U]):
            raise Exception("I,Q, or U is not set.")
        return self.I,self.Q,self.U,self.V

    def collect_full_skies(self):

        skyGenerators = {
            'skyA': self.skyA,
            'skyB': self.skyB,
            'skyC': self.skyC,
            'skyD': self.skyD,
            'skyE': self.skyE,
            'skyF': self.skyF,
            'skyG': self.skyG,
            'skyH': self.skyH,
            'skyI': self.skyI,
            'skyJ': self.skyJ,
            'skyK': self.skyK,
            'skyL': self.skyL,
            'skyM': self.skyM,
            'skyN': self.skyN
        }

        return skyGenerators

    def skyN(self):
        """fgs SCK generated
        nfreq 201
        band 100-200 MHz
        pfrac 0.01
        """
        file_dir = '/data4/paper/zionos/polskysim/fgs/pfrac001/'

        I = np.load(file_dir + 'I_use.npy')
        if self.p.unpolarized == False:
            Q = np.load(file_dir + 'Q_use.npy')
            U = np.load(file_dir + 'U_use.npy')
        else:
            Q = np.zeros_like(I)
            U = np.zeros_like(I)

        V = np.zeros_like(I)

        return I,Q,U,V

    def skyM(self):
        """fgs SCK generated
        nfreq 201
        band 100-200 MHz
        pfrac 0.05
        """
        file_dir = '/data4/paper/zionos/polskysim/fgs/pfrac005/'

        I = np.load(file_dir + 'I_use.npy')
        if self.p.unpolarized == False:
            Q = np.load(file_dir + 'Q_use.npy')
            U = np.load(file_dir + 'U_use.npy')
        else:
            Q = np.zeros_like(I)
            U = np.zeros_like(I)

        V = np.zeros_like(I)

        return I,Q,U,V

    def skyL(self):
        """fgs SCK generated
        nfreq 201
        band 100-200 MHz
        pfrac 0.5
        """
        file_dir = '/data4/paper/zionos/polskysim/fgs/pfrac05/'

        I = np.load(file_dir + 'I_use.npy')
        if self.p.unpolarized == False:
            Q = np.load(file_dir + 'Q_use.npy')
            U = np.load(file_dir + 'U_use.npy')
        else:
            Q = np.zeros_like(I)
            U = np.zeros_like(I)

        V = np.zeros_like(I)

        return I,Q,U,V

    def skyK(self):
        pfrac = 0.5
        I = get_gsm_cube(self.p)


    def skyA(self):
        I = get_gsm_cube()
        Q,U,V = [np.zeros((p.nfreq, npix)) for x in range(3)]
        return I,Q,U,V

    def skyB(self):
        I,Q,U,V = get_cora_polsky()
        if self.p.unpolarized == True:
            Q,U,V = [np.zeros((self.p.nfreq, npix)) for x in range(3)]

    def skyC(self):
        if (self.p.nside != 128) or (self.p.nfreq != 241): raise ValueError("The nside or nfreq of the simulation does not match the requested sky maps.")

        import h5py

        fpath = '/data4/paper/zionos/cora_maps/cora_polgalaxy1_nside128_nfreq241_band140_170.h5'
        print 'Using ' + fpath
        data = h5py.File(fpath)
        if self.p.unpolarized == True:
            I,_,_,_ = [data['map'][:,i,:] for i in [0,1,2,3]]
            Q,U,V = [np.zeros((self.p.nfreq, npix)) for x in range(3)]
        else:
            I,Q,U,V = [data['map'][:,i,:] for i in [0,1,2,3]]

        return I,Q,U,V

    def skyD(self):
        if (self.p.nside != 128) or (self.p.nfreq != 241): raise ValueError("The nside or nfreq of the simulation does not match the requested sky maps.")

        import h5py

        fpath = '/data4/paper/zionos/cora_maps/cora_polforeground1_nside128_nfreq241_band140_170.h5'
        print 'Using ' + fpath
        data = h5py.File(fpath)
        if self.p.unpolarized == True:
            I,_,_,_ = [data['map'][:,i,:] for i in [0,1,2,3]]
            Q,U,V = [np.zeros((self.p.nfreq, npix)) for x in range(3)]
        else:
            I,Q,U,V = [data['map'][:,i,:] for i in [0,1,2,3]]

        return I,Q,U,V

    def skyE(self):
        if (self.p.nside != 128) or (self.p.nfreq != 201): raise ValueError("The nside or nfreq of the simulation does not match the requested sky maps.")

        import h5py

        fpath = '/data4/paper/zionos/cora_maps/cora_polforeground2_nside128_nfreq201_band100_200.h5'
        print 'Using ' + fpath
        data = h5py.File(fpath)
        if self.p.unpolarized == True:
            I,_,_,_ = [data['map'][:,i,:] for i in [0,1,2,3]]
            Q,U,V = [np.zeros((self.p.nfreq, npix)) for x in range(3)]
            I = np.abs(I)
        else:
            I,Q,U,V = [data['map'][:,i,:] for i in [0,1,2,3]]
            I = np.abs(I)

        return I,Q,U,V

    def skyF(self):
        if (self.p.nside != 256) or (self.p.nfreq != 201): raise ValueError("The nside or nfreq of the simulation does not match the requested sky maps.")

        import h5py

        fpath = '/data4/paper/zionos/cora_maps/cora_polforeground3_nside256_nfreq201_band100_200.h5'
        print 'Using ' + fpath
        data = h5py.File(fpath)
        if self.p.unpolarized == True:
            I,_,_,_ = [data['map'][:,i,:] for i in [0,1,2,3]]
            Q,U,V = [np.zeros((self.p.nfreq, npix)) for x in range(3)]
            I = np.abs(I)
        else:
            I,Q,U,V = [data['map'][:,i,:] for i in [0,1,2,3]]
            I = np.abs(I)

        return I,Q,U,V

    def skyG(self):
        if (self.p.nside != 128) or (self.p.nfreq != 201): raise ValueError("The nside or nfreq of the simulation does not match the requested sky maps.")

        import h5py

        fpath = '/data4/paper/zionos/cora_maps/cora_21cm1_nside128_nfreq201_band100_200.h5'
        print 'Using ' + fpath
        data = h5py.File(fpath)
        if self.p.unpolarized == True:
            I,_,_,_ = [data['map'][:,i,:] for i in [0,1,2,3]]
            Q,U,V = [np.zeros((self.p.nfreq, self.p.npix)) for x in range(3)]
            I = np.abs(I)
        else:
            I,Q,U,V = [data['map'][:,i,:] for i in [0,1,2,3]]
            I = np.abs(I)

        return I,Q,U,V

    def skyH(self):
        if (self.p.nside != 128) or (self.p.nfreq != 241): raise ValueError("The nside or nfreq of the simulation does not match the requested sky maps.")

        import h5py
        fpath = '/data4/paper/zionos/cora_maps/cora_21cm1_nside128_nfreq241_band140_170.h5'
        print 'Using ' + fpath
        data = h5py.File(fpath)

        I = data['map'][:,0,:]
        Q,U,V = [np.zeros((self.p.nfreq, npix)) for x in range(3)]

        return I,Q,U,V

    def skyI(self):
        if (self.p.nside != 64) or (self.p.nfreq != 31): raise ValueError("The nside or nfreq of the simulation does not match the requested sky maps.")

        import h5py

        fpath = '/data4/paper/zionos/cora_maps/cora_polgalaxy1_nside64_nfreq31_band140_170.h5'
        print 'Using ' + fpath
        data = h5py.File(fpath)
        if self.p.unpolarized == True:
            I,_,_,_ = [data['map'][:,i,:] for i in [0,1,2,3]]
            Q,U,V = [np.zeros((self.p.nfreq, npix)) for x in range(3)]
        else:
            I,Q,U,V = [data['map'][:,i,:] for i in [0,1,2,3]]

        return I,Q,U,V

    def skyJ(self):
        ## unpolarized Point sources
        # src_ra = np.radians(np.array([120.,120.]))
        # src_dec = np.radians(np.array([-30.,-30.]))
        src_ra = np.radians(np.array([180.])) # these must be arrays
        src_dec = np.radians(np.array([-31.]))
        src_cza = np.pi/2. - src_dec

        src_idx = hp.ang2pix(self.p.nside, src_cza, src_ra)
        I = np.ones((self.p.nfreq, len(src_idx)))
        # I = np.ones(self.p.nfreq)
        I *= 150. # Jy?
        Q,U,V = [np.zeros((self.p.nfreq, npix)) for x in range(3)]

        return I,Q,U,V

    def point_skyA(self):
        ## Polarized Point sources
        # src_ra = np.radians(np.array([120.,120.]))
        # src_dec = np.radians(np.array([-30.,-30.]))
        src_ra = np.radians(np.array([155.])) # these must be arrays
        src_dec = np.radians(np.array([16.]))
        src_cza = np.pi/2. - src_dec

        src_idx = hp.ang2pix(self.p.nside, src_cza, src_ra)
        I = np.ones((self.p.nfreq, len(src_idx)))
        # I = np.ones(p.nfreq)
        I *= 150. # Jy?
        Q,U,V = [np.zeros((self.p.nfreq, len(src_idx))) for x in range(3)]

        Q = np.zeros((self.p.nfreq, len(src_idx)))
        # Q = np.ones((p.nfreq, len(src_idx)))
        # Q *= 0.5 * I

        # U = np.zeros((self.p.nfreq, len(src_idx)))
        U = np.ones((self.p.nfreq, len(src_idx)))
        U *=0.5 * I

        V = np.zeros((self.p.nfreq, len(src_idx)))

        if self.p.unpolarized == True:
            Q,U,V = [np.zeros((self.p.nfreq, len(src_idx))) for x in range(3)]

        return I,Q,U,V

    def point_skyB(self):
        ## Point sources
        src_ra = np.radians(np.array([120. + 130.,120. + 130.]))
        src_dec = np.radians(np.array([-10.,-10.]))
        src_cza = np.pi/2. - src_dec

        src_idx = hp.ang2pix(self.p.nside, src_cza, src_ra)
        I = np.ones((self.p.nfreq, len(src_idx)))
        I *= 150. # Jy?
        Q,U,V = [np.zeros((self.p.nfreq, npix)) for x in range(3)]

        return I,Q,U,V



def Hz2GHz(freq):
    return freq / 1e9

def get_gsm_cube(p):
    sys.path.append('/data4/paper/zionos/polskysim')
    import gsm2016_mod
    # import astropy.coordinates as coord
    # import astropy.units as units

    nside_in = 64
    npix_in = hp.nside2npix(nside_in)
    I_gal = np.zeros((p.nfreq, npix_in))
    for fi, f in enumerate(p.nu_axis):
        I_gal[fi] = gsm2016_mod.get_gsm_map_lowres(Hz2GHz(f))

    x_c = np.array([1.,0,0]) # unit vectors to be transformed by astropy
    y_c = np.array([0,1.,0])
    z_c = np.array([0,0,1.])

    # The GSM is given in galactic coordinates. We will rotate it to J2000 equatorial coordinates.
    axes_icrs = coord.SkyCoord(x=x_c, y=y_c, z=z_c, frame='icrs', representation='cartesian')
    axes_gal = axes_icrs.transform_to('galactic')
    axes_gal.representation = 'cartesian'

    R = np.array(axes_gal.cartesian.xyz) # The 3D rotation matrix that defines the coordinate transformation.

    npix_out = hp.nside2npix(p.nside)
    I = np.zeros((p.nfreq, npix_out))

    for i in range(p.nfreq):
        I[i] = irf.rotate_healpix_map(I_gal[i], R)
        I[i] = irf.harmonic_ud_grade(I[i], nside_in, p.nside)

    return I

def get_cora_polsky(pfrac_max=None):
    from cora.foreground import galaxy

    gal = galaxy.ConstrainedGalaxy()
    gal.nside = p.nside
    gal.frequencies = p.nu_axis

    stokes_cubes = gal.getpolsky()
    I,Q,U,V = [stokes_cubes[:,i,:] for i in range(4)]

    return I,Q,U,V
