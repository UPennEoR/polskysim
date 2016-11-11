import numpy as np
import healpy as hp
import os
from scipy import interpolate
import ionRIME_funcs as irf
import sys
import time
import numba_funcs as irnf
from sky_setup import SkyConstructor
import radiono

import astropy.coordinates as coord
import astropy.units as units
from astropy.time import Time
import yaml

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

def map2alm(marr, lmax):
    """
    Vectorized hp.map2alm
    """
    return np.apply_along_axis(lambda m: hp.map2alm(m, lmax=lmax),1,marr)

def alm2map(almarr, nside):
    """
    Vectorized hp.alm2map
    """
    return np.apply_along_axis(lambda alm: hp.alm2map(alm, nside, verbose=False), 1, almarr)

def compute_pointsource_visibility(p,d,t,ijones,ijonesH,sky_list,source_track,fringe_track,src_cza,src_ra,K,Vis):

    if p.circular_pol == True:
        I,Q,U,V = sky_list
    else:
        I,Q,U = sky_list

    print "t is " + str(t)
    total_angle = float(p.nhours * 15) # degrees
    offset_angle = float(p.hour_offset * 15) # degrees
    zl_ra = ((float(t) / float(p.ntime)) * np.radians(total_angle) + np.radians(offset_angle)) % (2.*np.pi)# radians

    npix = hp.nside2npix(p.nside)

    RotAxis = np.array([0.,0.,1.])
    RotAngle = -zl_ra # basicly Hour Angle with t=0

    # R_t = irf.rotation_matrix(RotAxis, RotAngle)
    # s0 = hp.ang2vec(src_cza,src_ra)
    # sf = np.einsum('...ab,...b->...b', R_t, s0)

    src_ra_f = src_ra + RotAngle
    sf = hp.ang2vec(src_cza, src_ra_f)

    # s0 = hp.ang2vec(src_cza,src_ra)
    # hpR_t = hp.Rotator(rot=[RotAngle])

    # sf = np.array(hpR_t(s0[:,0],s0[:,1],s0[:,2])).T

    if len(sf.shape) > 1:
        inds_f = hp.vec2pix(p.nside, sf[:,0],sf[:,1],sf[:,2])
    else:
        inds_f = hp.vec2pix(p.nside, sf[0],sf[1],sf[2])

    It = I #XXX ??

    ijones_t = ijones[:,inds_f,:,:]
    ijonesH_t = ijonesH[:,inds_f,:,:]
    Kt = K[:,inds_f]

    source_track[inds_f] += 1.
    fringe_track[inds_f] = Kt[0]

    if p.unpolarized == True:
        Ut = np.zeros((p.nfreq,It.shape[-1]))

        sky_t = np.array([
            [It, Ut],
            [Ut, It]]).transpose(2,3,0,1) # sky_t.shape = (p.nfreq, p.npix, 2, 2)

    else:
        Ut = U
        Qt = Q

        sky_t = np.array([
            [It + Qt, Ut],
            [Ut, It - Qt]]).transpose(2,3,0,1) # sky_t.shape = (p.nfreq, p.npix, 2, 2)

    irnf.instrRIME_integral(ijones_t, sky_t, ijonesH_t, Kt, Vis[d,t,:,:,:].squeeze())

# @profile
def compute_visibility(p,d,t,h,m,ionRM_out,ijones,ijonesH,sky_alms,K,Vis):

    print "t is " + str(t)
    total_angle = float(p.nhours * 15) # degrees
    offset_angle = float(p.hour_offset * 15) # degrees
    zl_ra = (float(t) / float(p.ntime)) * np.radians(total_angle) + np.radians(offset_angle) % 2*np.pi# radians

    npix = hp.nside2npix(p.nside)

    RotAxis = np.array([0.,0.,1.])
    RotAngle = -zl_ra # basicly Hour Angle with t=0

    mrot = np.exp(1j * m * RotAngle)

    temp = np.zeros((3,p.nfreq,p.npix))
    for xi,x in enumerate(sky_alms):
        for fi in range(x.shape[0]):
            temp1 = x[fi] * mrot
            temp[xi,fi,:] = hp.alm2map(temp1, p.nside, verbose=False)

    It,Qt,Ut = temp[0],temp[1],temp[2]

    ## Ionosphere
    """
    ionRM.shape = (p.nhours, p.nfreq, p.npix)
    ionRM_t.shape = (p.nfreq,p.npix)
    """
    if p.ionosphere == True:
        ionRM_t = ionRM_out[h] # pick out the map corresponding to this hour

        c = 299792458. # meters / sec
        lbda2 = (c / p.nu_axis)**2.
        ionAngle = np.outer(lbda2, np.ones(p.npix)) * ionRM_t

        ion_cos2 = irnf.numbap_cos(2. * ionAngle) # numba can multithread a numpy ufunc, no fuss no muss!
        ion_sin2 = irnf.numbap_sin(2. * ionAngle)

        QUout = irnf.complex_rotation(Qt,Ut, ion_cos2, ion_sin2)
        Qt = QUout.real
        Ut = QUout.imag

    sky_t = np.array([
        [It + Qt, Ut],
        [Ut, It - Qt]]).transpose(2,3,0,1) # sky_t.shape = (p.nfreq, p.npix, 2, 2)

    irnf.instrRIME_integral(ijones, sky_t, ijonesH, K, Vis[d,t,:,:,:].squeeze())

def main(p):

    npix = hp.nside2npix(p.nside)
    hpxidx = np.arange(npix)
    cza, ra = hp.pix2ang(p.nside, hpxidx)
    l,m = hp.Alm.getlm(p.lmax)

    z0_cza = np.radians(120.7215) # latitude of HERA/PAPER
    z0_ra = np.radians(0.)

    ## sky
    """
    sky.shape = (p.nfreq, npix, 2,2)
    """
    
    sky_init = SkyConstructor(p)
    I,Q,U,V = sky_init.stokes_parameters

    if p.point_source_sim == True:
        pass
    else:
        nside_use = hp.npix2nside(I.shape[1])
        I_alm, Q_alm, U_alm = map(lambda marr: map2alm(marr, p.lmax), [I,Q,U])
        if p.circular_pol == True:
            V_alm = map2alm(V, p.lmax)

    ## Instrument
    """
    Jdata.shape = (nfreq_in, p.npix, 2, 2)
    ijones.shape = (p.nfreq, p.npix, 2, 2)
    """
    freqs = [x * 1e6 for x in range(140,171)] # Hz
    # freqs = [(100 + 10 * x) * 1e6 for x in range(11)] # Hz. Must be converted to MHz for file list.
    #freqs = [140, 150, 160]
    tmark0 = time.time()

    nu0 = str(int(p.nu_axis[0] / 1e6))
    nuf = str(int(p.nu_axis[-1] / 1e6))
    fname = "ijones" + "band_" + nu0 + "-" + nuf + "mhz_nfreq" + str(p.nfreq)+ "_nside" + str(p.nside) + ".npy"

    # Ugh, this block makes baby jesus cry. l2oop
    if p.instrument == 'paper_feko':
        if os.path.exists('jones_save/PAPER/' + fname) == True:
            ijones = np.load('jones_save/PAPER/' + fname)
            print "Restored Jones model"
        else:
            ijones = irf.PAPER_instrument_setup(p, z0_cza)
            np.save('jones_save/PAPER/' + fname, ijones)

            tmark_inst = time.time()
            print "Completed instrument_setup(), in " + str(tmark_inst - tmark0)

    elif p.instrument == 'hera_hfss':
        if os.path.exists('jones_save/HERA_HFSS/' + fname) == True:
            ijones = np.load('jones_save/HERA_HFSS/' + fname)
            print "Restored Jones model"
        else:
            import hera_hfss_instrument_setup as hhis
            ijones = hhis.make_ijones_spectrum(p, verbose=True)
            np.save('jones_save/HERA_HFSS/' + fname, ijones)

            tmark_inst = time.time()
            print "Completed instrument_setup(), in " + str(tmark_inst - tmark0)

    elif p.instrument == 'paper_hfss':
        if os.path.exists('jones_save/PAPER_HFSS/' + fname) == True:
            ijones = np.load('jones_save/PAPER_HFSS/' + fname)
            print "Restored Jones model"
        else:
            import paper_hfss_instrument_setup as phis
            ijones = phis.make_ijones_spectrum(p, verbose=True)
            np.save('jones_save/PAPER_HFSS/' + fname, ijones)

            tmark_inst = time.time()
            print "Completed instrument_setup(), in " + str(tmark_inst - tmark0)

    elif p.instrument == 'hera':
        if os.path.exists('jones_save/' + fname) == True:
            ijones = np.load('jones_save/' + fname)
            print "Restored Jones model"
        else:
            Jdata = instrument_setup(z0_cza, freqs)

            tmark_inst = time.time()
            print "Completed instrument_setup(), in " + str(tmark_inst - tmark0)

            ijones = interpolate_jones_freq(Jdata, freqs, interp_type=p.interp_type)

            tmark_interp = time.time()
            print "Completed interpolate_jones_freq(), in " + str(tmark_interp - tmark_inst)

    else:
        raise ValueError('Instrument parameter is not a valid option.')

    ijonesH = np.transpose(ijones.conj(),(0,1,3,2))

    ## Baselines
    bl_eq = irf.transform_baselines(p.baselines) # get baseline vectors in equatorial coordinates

    l,m = hp.Alm.getlm(p.lmax)
    if p.point_source_sim == True:
        pass
    else:
        sky_alms = [I_alm, Q_alm, U_alm]
        if p.circular_pol == True:
            sky_alms.append(V_alm)

    sky_list = [I,Q,U]
    if p.circular_pol == True:
        sky_list.append(V)

    tmark_loopstart = time.time()

    ##
    """
    Fringe
    K.shape = (nfreq,npix)
    """
    c = 299792458. # meters / sec
    b = bl_eq[0]# meters, in the Equatorial basis
    s = hp.pix2vec(p.nside, hpxidx)
    b_dot_s = np.einsum('a...,a...',b,s)
    tau = b_dot_s / c
    K = np.exp(-2. * np.pi * 1j * np.outer(np.ones(p.nfreq), tau) * np.outer(p.nu_axis, np.ones(npix)) )

    if p.ionosphere == False:
        p.ndays = 1

    ## For each (t,f):
    # V[t,f,0,0] == V_xx[t,f]
    # V[t,f,0,1] == V_xy[t,f]
    # V[t,f,1,0] == V_yx[t,f]
    # V[t,f,1,1] == V_yy[t,f]
    Vis = np.zeros(p.ndays * p.ntime * p.nfreq * 2 * 2, dtype='complex128')
    Vis = Vis.reshape(p.ndays, p.ntime, p.nfreq, 2, 2)

    for d in range(p.ndays):
        if p.ionosphere == True:
            time_str = [irf.get_time_string(d, p.day0)] # the time string needed by radiono
            print "d is " + str(d) + ", day is " + time_str[0]

            heraRM = radiono.rm.HERA_RM(time_str)
            heraRM.make_radec_RM_maps()

            c_local = coord.AltAz(az=0. * units.degree, alt=90. * units.degree, obstime=Time(time_str[0] + 'T00:00:00', format='isot'), location=heraRM.location)

            c_local_Zeq = c_local.transform_to(coord.ICRS)
            z0_ra = c_local_Zeq.ra.radian

            hour0 = int(np.ceil(np.degrees(z0_ra/15.)))

            # the time axis on RMs starts at local midnight, by definition of the altaz<->radec transformation in radionopy
            # but my time loop does not start at local midnight, it starts t=0 when local zenith is at ra=0
            # the local time is hour0 ~= 11 hours after midnight at that point.
            # so the first RM map that should be seen is RM[hour0-1, :]


            # hour_inds = [(hour0 + p.hour_offset + x * p.nhours/p.ntime) % 24 for x in range(p.ntime)]
            hour_axis = [hour0 + x % 24 for x in range(p.nhours)]

            # if we aren't going to use all 24 hours of RM data, we don't want to be rotating and
            # resampling all 24 hours, just the hours that will be used

            ionRM_out = np.zeros((p.nhours, p.npix))
            for i, hr in enumerate(hour_axis):
                hrAngle = -z0_ra - np.radians(hr * 15.) # did i need to add the hour offset here? fuck
                lh,mh = hp.Alm.getlm(3*heraRM.nside -1)
                mh_rot = np.exp(1j * mh * hrAngle)
                ionRM_out[i] = hp.alm2map(hp.map2alm(heraRM.RMs[0,hr,:], lmax=3*heraRM.nside -1) * mh_rot, p.nside, verbose=False)
        else:
            ionRM_out = None
        ionRM_index = [(x * p.nhours/p.ntime) % 24 for x in range(p.ntime)]

        ## Debugging stuff
        if debug == True and p.point_source_sim == True:
            source_track = np.zeros(p.npix)
            fringe_track = np.zeros((p.npix), dtype='complex128')

        for t, h in enumerate(ionRM_index):
            if p.point_source_sim == True:
                compute_pointsource_visibility(p,d,t,ijones,ijonesH,sky_list,source_track,fringe_track,src_cza,src_ra,K,Vis)
            else:
                compute_visibility(p,d,t,h,m,ionRM_out,ijones,ijonesH,sky_alms,K,Vis)

    Vis /= hp.nside2npix(p.nside) # normalization
    tmark_loopstop = time.time()

    print "Visibility loop completed in " + str(tmark_loopstop - tmark_loopstart)
    print "Full run in " + str(tmark_loopstop -tmark0) + " seconds."

    # out_name = "Vis_band" + str(int(p.nu_0 / 1e6)) + "-" + str(int(p.nu_f /1e6)) + "MHz_nfreq" + str(p.nfreq) + "_ntime" + str(p.ntime) + "_nside" + str(p.nside) + ".npz"
    # if p.unpolarized == True:
    #     out_name = "unpol" + out_name
    # if (p.unpolarized == False) and (p.ionosphere == False):
    #     out_name = "noion" + out_name

    # out_name = p.outname_prefix + out_name

    #if os.path.exists(out_name) == False:

    out_name = "visibility.npz"

    np.savez(p.out_dir_use + out_name, Vis=Vis, param_dict=p.__dict__)

    if debug == True and p.point_source_sim == True:
        np.save('debug/source_track.npy', source_track)
        np.save('debug/fringe_track.npy', fringe_track)
    # np.savez('output_vis/' + 'param_dict_' + out_name, param_dict=p.__dict__)

class Parameters:
    def __init__(self, param_dict):
        for key in param_dict:
            setattr(self, key, param_dict[key])

if __name__ == '__main__':

    yamlfile = file('parameters.yaml', 'r')
    param_dict = yaml.load(yamlfile)
    p = Parameters(param_dict)

    p.npix = hp.nside2npix(p.nside)

    p.lmax = 3 * p.nside - 1

    p.nu_axis = np.linspace(p.nu_0,p.nu_f,num=p.nfreq,endpoint=True)

    for key in param_dict:
        print key,":",param_dict[key],":",type(param_dict[key])

    print p.interp_type
    if (p.unpolarized == True) and (p.ionosphere == True):
        raise Exception('Simulation includes the ionosphere with an unpolarized sky! Dummy!')

    if p.instrument == 'paper':
        out_dir = os.path.join(p.output_base_directory, 'PAPER/', p.output_directory)
    else:
        out_dir = os.path.join(p.output_base_directory, 'HERA/', p.output_directory)

    while os.path.exists(out_dir):
        out_dir = out_dir[:-1] + "B/"

    p.out_dir_use = out_dir

    os.makedirs(p.out_dir_use)
    from shutil import copyfile
    copyfile('parameters.yaml', p.out_dir_use + 'parameters.yaml')

    global debug
    debug = False

    if p.unpolarized == True:
        print "Polarization turned off"
    if p.ionosphere == False:
        print "Ionospheric rotation turned off"

    if p.circular_pol == False:
        print "No circular polarization! Booooo."

    main(p)
    print "Compiled successfully"
