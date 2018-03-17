import numpy as np
import healpy as hp
import astropy.coordinates as coord
import astropy.units as units
from astropy.time import Time
import pygsm
import ionRIME_funcs as irf
import os
import sys
import h5py

def Hz2GHz(freq):
    return freq / 1e9

class SkyConstructor(object):
    def __init__(self, params):

        for key in params:
            setattr(self, key, params[key])

        self.npix = hp.nside2npix(self.nside)

        self.lmax = 3 * self.nside -1

        self.nu_axis = np.linspace(self.nu_0, self.nu_f, num=self.nfreq, endpoint=True)

        self.hpxidx = np.arange(self.npix)

        attribute_test = getattr(self, 'z0_cza',None)
        if attribute_test is None:
            self.z0_cza = np.radians(120.7215) # latitude of HERA/PAPER

        skyGenerators = self.collect_full_skies()

        self.stokes_parameters = skyGenerators[self.sky_selection]()

    def collect_full_skies(self):

        skyGenerators = {
            'unpol_GSM2008': self.unpol_GSM2008,
            'MartaFGS': self.MartaFGS,
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
            'skyN': self.skyN,
            'skyPolNoiseA': self.skyPolNoiseA,
            'skyUniformI':self.skyUniformI,
            'skyCORA_pol_galaxy': self.skyCORA_pol_galaxy,
            'point_skyC': self.point_skyC,
            'point_skyD': self.point_skyD,
            'point_skyE': self.point_skyE,
            'point_skyF': self.point_skyF,
            'point_skyG': self.point_skyG,
            'point_skyH': self.point_skyH,
            'sim_group_test0': self.sim_group_test0,
            'get_cora_polsky': self.get_cora_polsky,
            'unit_Q': self.unit_Q,
            'unit_U': self.unit_U,
        }

        return skyGenerators

    def MartaFGS(self):
        if self.nside not in [128, 256]:
            raise Exception("nside is not 128 or 256. The only available maps are nside 128 or 256")

        if self.nfreq != 201:
            raise Exception("nfreq must be 201")

        nu0 = int(self.nu_axis[0]*1e-6)
        nuf = int(self.nu_axis[-1]*1e-6)

        if (nu0, nuf) not in [(100,200), (50,150)]:
            raise Exception("frequency band must be 100-200MHz or 50-150MHz")

        fname = 'nside{}_nfreq{}_{}-{}MHz.h5'.format(self.nside, self.nfreq, nu0, nuf)
        data_dir = '/lustre/aoc/projects/hera/zmartino/zionos/marta_foregrounds/'

        rescale_factor = 1. # rescale the polarization amplitude

        if self.unpolarized == True:
            I,Q,U,V = self.unpol_GSM2008()
        else:
            h5f = h5py.File(data_dir + fname, 'r')
            Q = h5f['Q'][:] * rescale_factor
            U = h5f['U'][:] * rescale_factor
            I = self.GSM2008()
            V = np.zeros_like(I)
        return I,Q,U,V

    def skyCORA_pol_galaxy(self):
        if (self.nside != 128) or (self.nfreq != 201): raise ValueError('nside or nfreq does not match this sky model')

        import h5py
        if os.path.exists('/data4/paper/'):
            raise Exception('Wrong system')
        else:
            fpath = '/lustre/aoc/projects/hera/zmartino/zionos/cora_models/galaxy_nside128_100-200MHz_nfreq201/map.h5'

        data = h5py.File(fpath)
        if self.unpolarized == True:
            I = data['map'][:,0,:]
            Q,U,V = [np.zeros((self.nfreq, self.npix)) for x in range(3)]
        else:
            I,Q,U,V = [data['map'][:,i,:] for i in [0,1,2,3]]

        return I,Q,U,V

    def skyPolNoiseA(self):
        I = self.GSM2008()
        V = np.zeros_like(I)
        seed = 873470
        np.random.seed(seed)

        pfrac = 0.1
        chi = np.random.rand(*I.shape)
        Q = pfrac * I * np.cos(2. * chi)
        U = pfrac * I * np.cos(2. * chi)

        return I,Q,U,V
    def skyUniformI(self):
        I = np.ones((self.nfreq, self.npix))
        Q,U,V = [np.zeros_like(I) for k in range(3)]
        return I,Q,U,V

    def skyN(self):
        """fgs SCK generated
        nfreq 201
        band 100-200 MHz
        pfrac 0.01
        """
        if os.path.exists('/lustre/aoc/projects/hera/zmartino'):
            file_dir = '/lustre/aoc/projects/hera/zmartino/zionos/polskysim/fgs/pfrac001/'
        elif os.path.exists('/data4/paper'):
            file_dir = '/data4/paper/zionos/polskysim/fgs/pfrac001/'

        I = np.load(file_dir + 'I_use.npy')
        if self.unpolarized == False:
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
        if os.path.exists('/data4/paper/'):
            file_dir = '/data4/paper/zionos/polskysim/fgs/pfrac005/'
        else:
            file_dir = '/lustre/aoc/projects/hera/zmartino/zionos/polskysim/fgs/pfrac005/'

        I = np.load(file_dir + 'I_use.npy')
        if self.unpolarized == False:
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
        if os.path.exists('/data4/paper/'):
            file_dir = '/data4/paper/zionos/polskysim/fgs/pfrac05/'
        else:
            file_dir = '/lustre/aoc/projects/hera/zmartino/zionos/polskysim/fgs/pfrac05/'

        I = np.load(file_dir + 'I_use.npy')
        if self.unpolarized == False:
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
        if self.unpolarized == True:
            Q,U,V = [np.zeros((self.nfreq, npix)) for x in range(3)]

    def skyC(self):
        if (self.nside != 128) or (self.nfreq != 241): raise ValueError("The nside or nfreq of the simulation does not match the requested sky maps.")

        import h5py
        if os.path.exists('/data4/paper/'):
            fpath = '/data4/paper/zionos/cora_maps/cora_polgalaxy1_nside128_nfreq241_band140_170.h5'
        else:
            fpath = '/lustre/aoc/projects/hera/zmartino/zionos/cora_maps/cora_polgalaxy1_nside128_nfreq241_band140_170.h5'
        print 'Using ' + fpath
        data = h5py.File(fpath)
        if self.unpolarized == True:
            I,_,_,_ = [data['map'][:,i,:] for i in [0,1,2,3]]
            Q,U,V = [np.zeros((self.nfreq, npix)) for x in range(3)]
        else:
            I,Q,U,V = [data['map'][:,i,:] for i in [0,1,2,3]]

        return I,Q,U,V

    def skyD(self):
        if (self.nside != 128) or (self.nfreq != 241): raise ValueError("The nside or nfreq of the simulation does not match the requested sky maps.")

        import h5py
        if os.path.exists('/data4/paper/'):
            fpath = '/data4/paper/zionos/cora_maps/cora_polforeground1_nside128_nfreq241_band140_170.h5'
        else:
            fpath = '/lustre/aoc/projects/hera/zmartino/zionos/cora_maps/cora_polforeground1_nside128_nfreq241_band140_170.h5'
        print 'Using ' + fpath
        data = h5py.File(fpath)
        if self.unpolarized == True:
            I,_,_,_ = [data['map'][:,i,:] for i in [0,1,2,3]]
            Q,U,V = [np.zeros((self.nfreq, npix)) for x in range(3)]
        else:
            I,Q,U,V = [data['map'][:,i,:] for i in [0,1,2,3]]

        return I,Q,U,V

    def skyE(self):
        if (self.nside != 128) or (self.nfreq != 201): raise ValueError("The nside or nfreq of the simulation does not match the requested sky maps.")

        import h5py
        if os.path.exists('/data4/paper/'):
            fpath = '/data4/paper/zionos/cora_maps/cora_polforeground2_nside128_nfreq201_band100_200.h5'
        else:
            fpath = '/lustre/aoc/projects/hera/zmartino/zionos/cora_maps/cora_polforeground2_nside128_nfreq201_band100_200.h5'

        print 'Using ' + fpath
        data = h5py.File(fpath)
        if self.unpolarized == True:
            I,_,_,_ = [data['map'][:,i,:] for i in [0,1,2,3]]
            Q,U,V = [np.zeros((self.nfreq, npix)) for x in range(3)]
            I = np.abs(I)
        else:
            I,Q,U,V = [data['map'][:,i,:] for i in [0,1,2,3]]
            I = np.abs(I)

        return I,Q,U,V

    def skyF(self):
        if (self.nside != 256) or (self.nfreq != 201): raise ValueError("The nside or nfreq of the simulation does not match the requested sky maps.")

        import h5py
        if os.path.exists('data4/paper/'):
            fpath = '/data4/paper/zionos/cora_maps/cora_polforeground3_nside256_nfreq201_band100_200.h5'
        else:
            fpath = '/lustre/aoc/projects/hera/zmartino/zionos/cora_maps/cora_polforeground3_nside256_nfreq201_band100_200.h5'
        print 'Using ' + fpath
        data = h5py.File(fpath)
        if self.unpolarized == True:
            I,_,_,_ = [data['map'][:,i,:] for i in [0,1,2,3]]
            Q,U,V = [np.zeros((self.nfreq, npix)) for x in range(3)]
            I = np.abs(I)
        else:
            I,Q,U,V = [data['map'][:,i,:] for i in [0,1,2,3]]
            I = np.abs(I)

        return I,Q,U,V

    def skyG(self):
        if (self.nside != 128) or (self.nfreq != 201): raise ValueError("The nside or nfreq of the simulation does not match the requested sky maps.")

        import h5py
        if os.path.exists('/data4/paper/'):
            fpath = '/data4/paper/zionos/cora_maps/cora_21cm1_nside128_nfreq201_band100_200.h5'
        else:
            fpath = '/lustre/aoc/projects/hera/zmartino/zionos/cora_maps/cora_21cm1_nside128_nfreq201_band100_200.h5'
        print 'Using ' + fpath
        data = h5py.File(fpath)
        if self.unpolarized == True:
            I,_,_,_ = [data['map'][:,i,:] for i in [0,1,2,3]]
            Q,U,V = [np.zeros((self.nfreq, self.npix)) for x in range(3)]
            I = np.abs(I)
        else:
            I,Q,U,V = [data['map'][:,i,:] for i in [0,1,2,3]]
            I = np.abs(I)

        return I,Q,U,V

    def skyH(self):
        if (self.nside != 128) or (self.nfreq != 241): raise ValueError("The nside or nfreq of the simulation does not match the requested sky maps.")

        import h5py
        if os.path.exists('/data4/paper/'):
            fpath = '/data4/paper/zionos/cora_maps/cora_21cm1_nside128_nfreq241_band140_170.h5'
        else:
            fpath = '/lustre/aoc/projects/hera/zmartino/zionos/cora_maps/cora_21cm1_nside128_nfreq241_band140_170.h5'
        print 'Using ' + fpath
        data = h5py.File(fpath)

        I = data['map'][:,0,:]
        Q,U,V = [np.zeros((self.nfreq, npix)) for x in range(3)]

        return I,Q,U,V

    def skyI(self):
        if (self.nside != 64) or (self.nfreq != 31): raise ValueError("The nside or nfreq of the simulation does not match the requested sky maps.")

        import h5py
        if os.path.exists('/data4/paper'):
            fpath = '/data4/paper/zionos/cora_maps/cora_polgalaxy1_nside64_nfreq31_band140_170.h5'
        else:
            fpath = '/lustre/aoc/projects/hera/zmartino/zionos/cora_maps/cora_polgalaxy1_nside64_nfreq31_band140_170.h5'
        print 'Using ' + fpath
        data = h5py.File(fpath)
        if self.unpolarized == True:
            I,_,_,_ = [data['map'][:,i,:] for i in [0,1,2,3]]
            Q,U,V = [np.zeros((self.nfreq, npix)) for x in range(3)]
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

        src_idx = hp.ang2pix(self.nside, src_cza, src_ra)
        I = np.ones((self.nfreq, len(src_idx)))
        # I = np.ones(self.nfreq)
        I *= 150. # Jy?
        Q,U,V = [np.zeros((self.nfreq, npix)) for x in range(3)]

        return I,Q,U,V

    def point_skyH(self):
        self.src_ra = np.radians(np.array([5. * 15.]))
        self.src_dec = np.radians(np.array([-32.72]))
        self.src_cza = np.pi/2. - self.src_dec

        src_idx = hp.ang2pix(self.nside, self.src_cza, self.src_ra)
        I = np.ones((self.nfreq, len(src_idx)))
        U = 0.2 * np.ones((self.nfreq, len(src_idx)))
        Q,V = [np.zeros((self.nfreq, len(src_idx))) for x in range(2)]

        return I,Q,U,V

    def point_skyG(self):
        self.src_ra = np.radians(np.array([5. * 15.]))
        self.src_dec = np.radians(np.array([-32.72]))
        self.src_cza = np.pi/2. - self.src_dec

        src_idx = hp.ang2pix(self.nside, self.src_cza, self.src_ra)
        I = np.ones((self.nfreq, len(src_idx)))
        Q = 0.2 * np.ones((self.nfreq, len(src_idx)))
        U,V = [np.zeros((self.nfreq, len(src_idx))) for x in range(2)]

        return I,Q,U,V

    def point_skyE(self):
        self.src_ra = np.radians(np.array([5. * 15.]))
        self.src_dec = np.radians(np.array([-15.]))
        self.src_cza = np.pi/2. - self.src_dec

        src_idx = hp.ang2pix(self.nside, self.src_cza, self.src_ra)
        I = np.ones((self.nfreq, len(src_idx)))
        U = 0.2 * np.ones((self.nfreq, len(src_idx)))
        Q,V = [np.zeros((self.nfreq, len(src_idx))) for x in range(2)]

        return I,Q,U,V

    def point_skyD(self):
        self.src_ra = np.radians(np.array([5. * 15.]))
        self.src_dec = np.radians(np.array([-15.]))
        self.src_cza = np.pi/2. - self.src_dec

        src_idx = hp.ang2pix(self.nside, self.src_cza, self.src_ra)
        I = np.ones((self.nfreq, len(src_idx)))
        Q = 0.2 * np.ones((self.nfreq, len(src_idx)))
        U,V = [np.zeros((self.nfreq, len(src_idx))) for x in range(2)]

        return I,Q,U,V

    def point_skyF(self):
        """
        unpolarized point source that transits zenith
        """
        self.src_ra = np.radians(np.array([5. * 15.]))
        self.src_dec = np.radians(np.array([-32.72]))
        self.src_cza = np.pi/2. - self.src_dec

        src_idx = hp.ang2pix(self.nside, self.src_cza, self.src_ra)
        I = np.ones((self.nfreq, len(src_idx)))
        Q,U,V = [np.zeros((self.nfreq, len(src_idx))) for x in range(3)]

        return I,Q,U,V

    def point_skyC(self):
        self.src_ra = np.radians(np.array([5. * 15.]))
        self.src_dec = np.radians(np.array([-15.]))
        self.src_cza = np.pi/2. - self.src_dec

        src_idx = hp.ang2pix(self.nside, self.src_cza, self.src_ra)
        I = np.ones((self.nfreq, len(src_idx)))
        Q,U,V = [np.zeros((self.nfreq, len(src_idx))) for x in range(3)]

        return I,Q,U,V

    def point_skyA(self):
        ## Polarized Point sources
        # src_ra = np.radians(np.array([120.,120.]))
        # src_dec = np.radians(np.array([-30.,-30.]))
        src_ra = np.radians(np.array([155.])) # these must be arrays
        src_dec = np.radians(np.array([16.]))
        src_cza = np.pi/2. - src_dec

        src_idx = hp.ang2pix(self.nside, src_cza, src_ra)
        I = np.ones((self.nfreq, len(src_idx)))
        # I = np.ones(p.nfreq)
        I *= 150. # Jy?
        Q,U,V = [np.zeros((self.nfreq, len(src_idx))) for x in range(3)]

        Q = np.zeros((self.nfreq, len(src_idx)))
        # Q = np.ones((p.nfreq, len(src_idx)))
        # Q *= 0.5 * I

        # U = np.zeros((self.nfreq, len(src_idx)))
        U = np.ones((self.nfreq, len(src_idx)))
        U *=0.5 * I

        V = np.zeros((self.nfreq, len(src_idx)))

        if self.unpolarized == True:
            Q,U,V = [np.zeros((self.nfreq, len(src_idx))) for x in range(3)]

        return I,Q,U,V

    def point_skyB(self):
        ## Point sources
        src_ra = np.radians(np.array([120. + 130.,120. + 130.]))
        src_dec = np.radians(np.array([-10.,-10.]))
        src_cza = np.pi/2. - src_dec

        src_idx = hp.ang2pix(self.nside, src_cza, src_ra)
        I = np.ones((self.nfreq, len(src_idx)))
        I *= 150. # Jy?
        Q,U,V = [np.zeros((self.nfreq, npix)) for x in range(3)]

        return I,Q,U,V

    def get_gsm_cube(self):
        if os.path.exists('data4/paper/'):
            sys.path.append('/data4/paper/zionos/polskysim')
        else:
            sys.path.append('/lustre/aoc/projects/hera/zmartino/zionos/polskysim')
        import gsm2016_mod
        # import astropy.coordinates as coord
        # import astropy.units as units

        nside_in = 64
        npix_in = hp.nside2npix(nside_in)
        I_gal = np.zeros((self.nfreq, npix_in))
        for fi, f in enumerate(self.nu_axis):
            I_gal[fi] = gsm2016_mod.get_gsm_map_lowres(Hz2GHz(f))

        x_c = np.array([1.,0,0]) # unit vectors to be transformed by astropy
        y_c = np.array([0,1.,0])
        z_c = np.array([0,0,1.])

        # The GSM is given in galactic coordinates. We will rotate it to J2000 equatorial coordinates.
        axes_icrs = coord.SkyCoord(x=x_c, y=y_c, z=z_c, frame='icrs', representation='cartesian')
        axes_gal = axes_icrs.transform_to('galactic')
        axes_gal.representation = 'cartesian'

        R = np.array(axes_gal.cartesian.xyz) # The 3D rotation matrix that defines the coordinate transformation.

        npix_out = hp.nside2npix(self.nside)
        I = np.zeros((self.nfreq, npix_out))

        for i in range(self.nfreq):
            I[i] = irf.rotate_healpix_map(I_gal[i], R)
            I[i] = irf.harmonic_ud_grade(I[i], nside_in, self.nside)

        return I

    def get_cora_polsky(self, pfrac_max=None):
        from cora.foreground import galaxy

        gal = galaxy.ConstrainedGalaxy()
        gal.nside = self.nside
        gal.frequencies = self.nu_axis * 1e-6 # MHz - self.nu_axis is in Hz.

        stokes_cubes = gal.getpolsky()
        I,Q,U,V = [stokes_cubes[:,i,:] for i in range(4)]
        
        if self.unpolarized == True:
            Q *= 0
            U *= 0
            V *= 0

        return I,Q,U,V

    def GSM2008(self):
        x_c = np.array([1.,0,0]) # unit vectors to be transformed by astropy
        y_c = np.array([0,1.,0])
        z_c = np.array([0,0,1.])

        # The GSM is given in galactic coordinates. We will rotate it to J2000 equatorial coordinates.
        axes_icrs = coord.SkyCoord(x=x_c, y=y_c, z=z_c, frame='icrs', representation='cartesian')
        axes_gal = axes_icrs.transform_to('galactic')
        axes_gal.representation = 'cartesian'

        R = np.array(axes_gal.cartesian.xyz) # The 3D rotation matrix that defines the coordinate transformation.

        I = np.zeros((self.nfreq, self.npix))

        gsm = pygsm.GlobalSkyModel(freq_unit='MHz', interpolation='cubic')
        for i,nu in enumerate(self.nu_axis):
            I_nu = gsm.generate(nu * 1e-6)
            rI_nu = irf.rotate_healpix_mindex(I_nu, R)
            I[i,:] = irf.harmonic_ud_grade(rI_nu, self.nside)

        return I

    def unpol_GSM2008(self):
        I = self.GSM2008()
        Q,U,V = [np.zeros_like(I) for k in range(3)]
        return I,Q,U,V

    def sim_group_test0(self):
        x_c = np.array([1.,0,0]) # unit vectors to be transformed by astropy
        y_c = np.array([0,1.,0])
        z_c = np.array([0,0,1.])

        # The GSM is given in galactic coordinates. We will rotate it to J2000 equatorial coordinates.
        axes_icrs = coord.SkyCoord(x=x_c, y=y_c, z=z_c, frame='icrs', representation='cartesian')
        axes_gal = axes_icrs.transform_to('galactic')
        axes_gal.representation = 'cartesian'

        R = np.array(axes_gal.cartesian.xyz) # The 3D rotation matrix that defines the coordinate transformation.

        I = np.zeros((self.nfreq, self.npix))

        gsm = pygsm.GlobalSkyModel(freq_unit='MHz', interpolation='cubic')
        I_150 = gsm.generate(150.)
        rI_150 = irf.rotate_healpix_mindex(I_150, R)
        urI_150 = irf.harmonic_ud_grade(rI_150, self.nside)

        for i,nu in enumerate(self.nu_axis):
            I[i,:] = urI_150

        Q,U,V = [np.zeros_like(I) for k in range(3)]

        return I,Q,U,V

    def unit_Q(self):
        Q = np.ones((self.nfreq, self.npix))
        I,U,V = [np.zeros_like(Q) for k in range(3)]
        return I,Q,U,V

    def unit_U(self):
        U = np.ones((self.nfreq, self.npix))
        I,Q,V = [np.zeros_like(U) for k in range(3)]
        return I,Q,U,V
