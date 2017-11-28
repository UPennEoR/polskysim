import numpy as np, healpy as hp
import astropy.io.fits as fits
import ionRIME_funcs as irf

data_dir = '/users/zmartino/zmartino/HERA-Beams/NicolasFagnoniBeams/'

def udgrade(x,nside_out):
    return hp.alm2map(hp.map2alm(x),nside_out, verbose=False)

def AzimuthalRotation(hmap):
    """
    Azimuthal clockwise(?) rotation of a healpix map by pi/2 about the z-axis

    """
    npix = len(hmap)
    nside= hp.npix2nside(npix)
    hpxidx = np.arange(npix)
    t2,p2 = hp.pix2ang(nside, hpxidx)

    p = p2 - np.pi/2
    p[p < 0] += 2. * np.pi
    t = t2

    idx = hp.ang2pix(nside, t, p)

    hout = hmap[idx]
    return hout

def horizon_mask(jones, z0_cza):
    npix = jones.shape[0]
    nside = hp.npix2nside(npix)
    hpxidx = np.arange(npix)
    cza, ra = hp.pix2ang(nside, hpxidx)

    if z0_cza == 0.:
        tb, pb = cza, ra
    else:

        z0 = irf.r_hat_cart(z0_cza, 0.)

        RotAxis = np.cross(z0, np.array([0,0,1.]))
        RotAxis /= np.sqrt(np.dot(RotAxis,RotAxis))
        RotAngle = np.arccos(np.dot(z0, [0,0,1.]))

        R_z0 = irf.rotation_matrix(RotAxis, RotAngle)

        tb, pb = irf.rotate_sphr_coords(R_z0, cza, ra)

    hm = np.zeros((npix,2,2))
    hm[np.where(tb < np.pi/2.)] = 1.

    return hm

class Parameters:
    def __init__(self, param_dict):
        for key in param_dict:
            setattr(self, key, param_dict[key])

def make_ijones_spectrum(parameters, verbose=False):
    p = Parameters(parameters)


    d = fits.open(data_dir + 'healpix_beam.fits')
    beams = d[0].data
    freqs = d[1].data

    # select only 100-200 MHz data
    freq_select = np.where((freqs >= 100) & (freqs <=200))[0]
    beams = beams[:, freq_select]
    freqs = freqs[freq_select]
    Nfreqs = len(freqs)

    # take East pol and rotate to get North pol
    beam_theta, beam_phi = hp.pix2ang(64, np.arange(64**2 * 12))
    R = hp.Rotator(rot=[0,0,-np.pi/2], deg=False)
    beam_theta2, beam_phi2 = R(beam_theta, beam_phi)
    beam_rot = np.array(map(lambda x: hp.get_interp_val(x, beam_theta2, beam_phi2), beams.T))
    beam_data = np.array([beams.T, beam_rot])
    beam_data.resize(2, Nfreqs, 49152)

    # normalize each frequency to max of 1
    for i in range(beam_data.shape[-2]):
        beam_data[:, i, :] /= beam_data[:, i, :].max()

    uJee = udgrade(np.sqrt(beam_data[0,50,:]), p.nside)

    uJnn = AzimuthalRotation(uJee)
    Jee = irf.rotate_healpix_mapHPX(uJee, [0., -np.degrees(p.z0_cza),0])
    Jee /= np.amax(Jee)
    Jnn = irf.rotate_healpix_mapHPX(uJnn, [0., -np.degrees(p.z0_cza),0])
    Jnn /= np.amax(Jnn)

    jones = np.array([
        [np.abs(Jee), np.zeros(len(Jee))],
        [np.zeros(len(Jee)), np.abs(Jnn)]
    ], dtype=np.complex).transpose(2,0,1)

    hm = horizon_mask(jones, p.z0_cza)
    return np.broadcast_to(jones * hm, (p.nfreq,) + jones.shape)
