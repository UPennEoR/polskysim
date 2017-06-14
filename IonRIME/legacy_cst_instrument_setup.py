import numpy as np
import healpy as hp
import os
from scipy import interpolate
import ionRIME_funcs as irf
import sys
import time
import numba_funcs as irnf

def transform_basis(nside, jones, z0_cza, R_z0):
    """
    At zenith in the local frame the 'x' feed is aligned with 'theta' and
    the 'y' feed is aligned with 'phi'
    """
    npix = hp.nside2npix(nside)
    hpxidx = np.arange(npix)
    cza, ra = hp.pix2ang(nside, hpxidx)

    # Rb is the rotation relating the E-field basis coordinate frame to the local horizontal zenith.
    # (specific to this instrument response simulation data)
    Rb = np.array([
    [0,0,-1],
    [0,-1,0],
    [-1,0,0]
    ])

    fR = np.einsum('ab,bc->ac', Rb, R_z0) # matrix product of two rotations

    tb, pb = irf.rotate_sphr_coords(fR, cza, ra)

    cza_v = irf.t_hat_cart(cza, ra)
    ra_v = irf.p_hat_cart(cza, ra)

    tb_v = irf.t_hat_cart(tb, pb)

    fRcza_v = np.einsum('ab...,b...->a...', fR, cza_v)
    fRra_v = np.einsum('ab...,b...->a...', fR, ra_v)

    cosX = np.einsum('a...,a...', fRcza_v, tb_v)
    sinX = np.einsum('a...,a...', fRra_v, tb_v)

    basis_rot = np.array([[cosX, sinX],[-sinX, cosX]])
    basis_rot = np.transpose(basis_rot,(2,0,1))

    return irnf.M(jones, basis_rot)

def instrument_setup(z0_cza, freqs):
    """
    This is the CST simulation using the efield basis of z' = -x, y' = -y, x' = -z
    frequencies are every 10MHz, from 100-200
    Each file contains 8 columns which are ordered as:
          (Re(xt),Re(xp),Re(yt),Re(yp),Im(xt),Im(xp),Im(yt),Im(yp)).
    Each column is a healpix map with resolution nside = 2**8
    """

    nu0 = str(int(p.nu_axis[0] / 1e6))
    nuf = str(int(p.nu_axis[-1] / 1e6))
    band_str = nu0 + "-" + nuf

    # restore_name = p.interp_type + "_" + "band_" + band_str + "mhz_nfreq" + str(p.nfreq)+ "_nside" + str(p.nside) + ".npy"
    #
    # if os.path.exists('jones_save/' + restore_name) == True:
    #     return np.load('jones_save/' + restore_name)
    #
    local_jones0_file = 'local_jones0/nside' + str(p.nside) + '_band' + band_str + '_Jdata.npy'

    if os.path.exists(local_jones0_file) == True:
        return np.load(local_jones0_file)

    fbase = '/data4/paper/zionos/HERA_jones_data/HERA_Jones_healpix_'
    # fbase = '/home/zmart/radcos/polskysim/IonRIME/HERA_jones_data/HERA_Jones_healpix_'

    nside_in = 2**8
    fnames = [fbase + str(int(f / 1e6)) + 'MHz.txt' for f in freqs]
    nfreq_nodes = len(freqs)

    npix = hp.nside2npix(nside_in)
    hpxidx = np.arange(npix)
    cza, ra = hp.pix2ang(nside_in, hpxidx)

    z0 = irf.r_hat_cart(z0_cza, 0.)

    RotAxis = np.cross(z0, np.array([0,0,1.]))
    RotAxis /= np.sqrt(np.dot(RotAxis,RotAxis))
    RotAngle = np.arccos(np.dot(z0, [0,0,1.]))

    R_z0 = irf.rotation_matrix(RotAxis, RotAngle)

    t0, p0 = irf.rotate_sphr_coords(R_z0, cza, ra)

    hm = np.zeros(npix)
    hm[np.where(cza < (np.pi / 2. + np.pi / 20.))] = 1 # Horizon mask; is 0 below the local horizon.
    # added some padding. Idea being to allow for some interpolation near the horizon. Questionable.
    npix_out = hp.nside2npix(p.nside)

    Jdata = np.zeros((nfreq_nodes,npix_out,2,2),dtype='complex128')
    for i,f in enumerate(fnames):
        J_f = np.loadtxt(f) # J_f.shape = (npix_in, 8)

        J_f = J_f * np.tile(hm, 8).reshape(8, npix).transpose(1,0) # Apply horizon mask

        # Could future "rotation" of these zeroed-maps have small errors at the
        # edges of the horizon? due to the way healpy interpolates.
        # Unlikely to be important.
        # Comment update: Yep, it turns out this happens, BUT it is approximately
        # power-preserving. The pixels at the edges of the rotated mask are not
        # identically 1, but the sum over the mask is maintained to about a part
        # in 1e-5

        # Perform a scalar rotation of each component so that the instrument's boresight
        # is pointed toward (z0_cza, 0), the location of the instrument on the
        # earth in the Current-Epoch-RA/Dec coordinate frame.
        J_f = irf.rotate_jones(J_f, R_z0, multiway=False)

        if p.nside != nside_in:
            # Change the map resolution as needed.

            #d = lambda m: hp.ud_grade(m, nside=p.nside, power=-2.)
                # I think these two ended up being (roughly) the same?
                # The apparent normalization problem was really becuase of an freq. interpolation problem.
                # irf.harmonic_ud_grade is probably better for increasing resolution, but hp.ud_grade is
                # faster because it's just averaging/tiling instead of doing SHT's
            d = lambda m: irf.harmonic_ud_grade(m, nside_in, p.nside)
            J_f = (np.asarray(map(d, J_f.T))).T
                # The inner transpose is so that correct dimension is map()'ed over,
                # and then the outer transpose returns the array to its original shape.

        J_f = irf.inverse_flatten_jones(J_f) # Change shape to (nfreq,npix,2,2), complex-valued
        J_f = transform_basis(p.nside, J_f, z0_cza, R_z0) # right-multiply by the basis transformation matrix from RA/CZA to the Local CST basis.
                                                          # Note that CZA = pi/2 - Dec! So this is not quite the RA/Dec basis. But the difference
                                                          # in the Stoke parameters between the two is only U -> -U
        Jdata[i,:,:,:] = J_f
        print i

    # If the model at the current nside hasn't been generated before, save it for future reuse.
    if os.path.exists(local_jones0_file) == False:
        np.save(local_jones0_file, Jdata)

    return Jdata

def _interpolate_jones_freq(J_in, freqs, multiway=True, interp_type='spline'):
    """
    A scheme to interpolate the spherical harmonic components of jones matrix elements.
    Does not seem to work well, and is unused.
    """
    nfreq_in = len(freqs)

    if multiway == True:
        J_flat = np.zeros((nfreq_in, npix, 8), dtype='float64')
        for i in range(nfreq_in):
            J_flat[i] = irf.flatten_jones(J_in[i])
        J_in = J_flat

    lmax = 3 * nside -1
    nlm = hp.Alm.getsize(lmax)
    Jlm_in = np.zeros(nfreq_in, nlm, 8)
    for i in range(nfreq_in):
        sht = lambda m: hp.map2alm(m, lmax=lmax)
        Jlm_in[i,:,:] = (np.asarray(map(sht, J_in.T))).T

    Jlm_out = np.zeros(p.nfreq, nlm, 8)
    for lm in range(nlm):
        for j in range(8):
            Jlmj_re = np.real(Jlm_in[:,lm,j])
            Jlmj_im = np.imag(Jlm_in[:,lm,j])

            a = interpolate_pixel(Jlmj_re, freqs, p.nu_axis, interp_type=p.interp_type) # note! interpolate_pixel function no longer exists
            b = interpolate_pixel(Jlmj_im, freqs, p.nu_axis, interp_type=p.interp_type)
            Jlm_out[:, lm, j] = a + 1j*b

    # J_in.shape = (p.nfreq_in, ??, 8)

    # Now, return alm's? or spatial maps?

def interpolate_jones_freq(J_in, freqs, multiway=True, interp_type='cubic'):
    #nfreq_out = len(nu_axis)
    nfreq_in = len(freqs)
    npix = len(J_in[0,:,0])
    #nside = hp.npix2nside(npix)

    if multiway == True:
        J_flat = np.zeros((nfreq_in, npix, 8), dtype='float64')
        for i in range(nfreq_in):
            J_flat[i] = irf.flatten_jones(J_in[i])
        J_in = J_flat

    # J_in.shape = (nfreq_in,npix, 8)

    interpolant = interpolate.interp1d(freqs, J_in, kind=interp_type,axis=0)
    J_out = interpolant(p.nu_axis)

    if multiway == True:
        J_m = np.zeros((p.nfreq, npix, 2,2), dtype='complex128')
        for i in range(p.nfreq):
            J_m[i] = irf.inverse_flatten_jones(J_out[i])
        J_out = J_m

    for i in range(p.nfreq):
        Bx_max = np.amax(np.absolute(J_out[i,:,0,0])**2. + np.absolute(J_out[i,:,0,1])**2.)
        By_max = np.amax(np.absolute(J_out[i,:,1,0])**2. + np.absolute(J_out[i,:,1,1])**2.)
        J_out[i,:,0,0] /= np.sqrt(Bx_max)
        J_out[i,:,0,1] /= np.sqrt(Bx_max)
        J_out[i,:,1,0] /= np.sqrt(By_max)
        J_out[i,:,1,1] /= np.sqrt(By_max)

    # Bah, figure it out later
    # Bx_max = np.amax(
    #     np.absolute(J_out[:,:,0,0])**2. + np.absolute(J_out[:,:,0,1])**2.,
    #     axis=1)
    # By_max = np.amax(
    #     np.absolute(J_out[:,:,1,0])**2. + np.absolute(J_out[:,:,1,1])**2.,
    #     axis=1)
    # # Bx_max.shape = By_max = (nfreq,)
    #
    # J_out[:,:,0,:] /= Bx_max[:,None,None]
    #
    # J_out[:,:,1,:] /= By_max[:,None,None]

    nu0 = str(int(p.nu_axis[0] / 1e6))
    nuf = str(int(p.nu_axis[-1] / 1e6))
    fname = p.interp_type + "_" + "band_" + nu0 + "-" + nuf + "mhz_nfreq" + str(p.nfreq)+ "_nside" + str(p.nside) + ".npy"
    if p.instrument == 'paper':
        np.save('jones_save/PAPER/' + fname, J_out)
    else:
        np.save('jones_save/' + fname, J_out)

    return J_out
