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

import math
import itertools
from fractions import Fraction

import yaml


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
    zl_ra = ((float(t) / float(p.ntime)) * np.radians(total_angle) + np.radians(offset_angle)) % (2*np.pi) # radians

    npix = hp.nside2npix(p.nside)

    RotAxis = np.array([0.,0.,1.])
    RotAngle = zl_ra

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

    # if p.point_source_sim == True:
    #     pass
    # else:
    if p.point_source_sim == False:
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
    if p.instrument == 'hera_NicCST':
        if os.path.exists('jones_save/HERA_NicCST/' + fname) == True:
            ijones = np.load('jones_save/HERA_NicCST/' + fname)
            print "Restored Jones model"
        else:
            import hera_NicCST_instrument_setup as hnis
            ijones = hnis.make_ijones_spectrum(p, verbose=True)
            np.save('jones_save/HERA_NicCST/' + fname, ijones)

            tmark_inst = time.time()
            print "Completed instrument_setup(), in " + str(tmark_inst - tmark0)

    elif p.instrument == 'paper_feko':
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
            import legacy_cst_instrument_setup as lcis
            Jdata = lcis.instrument_setup(z0_cza, freqs)

            tmark_inst = time.time()
            print "Completed instrument_setup(), in " + str(tmark_inst - tmark0)

            ijones = lcis.interpolate_jones_freq(Jdata, freqs, interp_type=p.interp_type)

            tmark_interp = time.time()
            print "Completed interpolate_jones_freq(), in " + str(tmark_interp - tmark_inst)

    elif p.instrument == 'analytic_dipole':
        ijones = irf.analytic_dipole_setup(p.nside, p.nfreq, z0_cza=z0_cza)

        tmark_inst = time.time()
        print "Completed instrument_setup(), in " + str(tmark_inst - tmark0)

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

    p.RMs = []

    if p.ionosphere_type == 'constant':
        RMs_npz_path = '/data4/paper/zionos/polskysim/IonRIME/RM_sim_data/' + p.RMs_sequence_file_name
        RMs_npz = np.load(RMs_npz_path)
        p.RMs = RMs_npz['RMs']

    d0 = (p.sim_part-1)*p.ndays
    d1 = (p.sim_part-1)*p.ndays + p.ndays
    date_strs = [irf.get_time_string(x,p.day0) for x in range(d0-1,d1+1)]
    if p.ionosphere == True and p.ionosphere_type == 'radionopy':
        heraRM = radiono.rm.HERA_RM(date_strs)
        heraRM.make_radec_RM_maps() # compute all the radionopy RM maps that will be used. The RM array for each day is only ~0.7 MB.

    ## I think this is unneeded now?
    # c_local = coord.AltAz(az=0. * units.degree, alt=90. * units.degree, obstime=Time(date_strs[0] + 'T00:00:00', format='isot'), location=heraRM.location)
    # # the Alt/Az coordinates of the local zenith at midnight on p.day0
    #
    # c_local_Zeq = c_local.transform_to(coord.ICRS)
    # z0_ra = c_local_Zeq.ra.radian # the ra coordinate of zenith at midnight on day0

    p.ra0 = np.radians(p.hour_offset * 15.) # the starting RA of the simulation window i.e. when t=0, p.ra0 is at the beam's zenith meridian

    UT_hour_start_str = radiono.utils.nextTransit(date_strs[1], np.degrees(p.ra0),0.).split(' ')[1] # an extra day has been added before "day0",
    UT_hour_start = int(UT_hour_start_str[:2]) # the UT hour of the first RM map on the first day

    for d in range(p.ndays):
        if p.ionosphere == True:
            if p.ionosphere_type == 'radionopy':
                print "d is " + str(d) + ", day is " + date_strs[d+1]

                ds = d + (p.sim_part-1)*p.ndays

                bins_per_hour = float(p.ntime)/p.nhours
                hours_per_degree = 1./15.     # 1 deg ~= 1/15 hours
                bins_per_degree = bins_per_hour * hours_per_degree

                dec_part, _ = math.modf(bins_per_degree)

                frac = Fraction(dec_part).limit_denominator(max_denominator=100)

                time_bin_shift = ds * int(bins_per_degree) + ds/frac.denominator

                ionRM_index = [(x * p.nhours/p.ntime) + UT_hour_start + 24 for x in range(-time_bin_shift, p.ntime)] # the +24 is to shift the UTs to the middle of the heraRM_use axis

                ionRM_index = np.roll(ionRM_index, time_bin_shift)[-p.ntime:]

                UT_index = list(set(ionRM_index))
                UT_index.sort()

                ionRM_index = [x - min(ionRM_index) for x in ionRM_index]

                ionRM_out = np.zeros((p.nhours+1, p.npix))
                heraRM_use = np.concatenate((heraRM.RMs[d,:,:], heraRM.RMs[d+1,:,:]), axis=0) # 48 hours of RM maps, starting at midnight on day "d"

                if max(UT_index) > 47:
                    heraRM_use = np.concatenate((heraRM_use, heraRM.RMs[d+2,:,:]), axis=0)

                for i, hr in enumerate(UT_index):
                    ##OLD
                    # hrAngle = -p.ra0 + np.radians(hr * 15.) + np.pi # approximately true?
                    # lh,mh = hp.Alm.getlm(3*heraRM.nside -1)
                    # mh_rot = np.exp(1j * mh * hrAngle)
                    # ionRM_out[i] = hp.alm2map(hp.map2alm(heraRM_use[hr,:], lmax=3*heraRM.nside -1)* mh_rot, p.nside, verbose=False)

                    RotAngle = -p.ra0 - np.radians((hr-2) * 15.)
                    # the (hr-2) has something to do with the local time zone being UT+2.
                    # Found that it should be (hr-2) by trial-and-error matching the radionopy altaz map...
                    temp1 = hp.ud_grade(heraRM_use[hr,:], nside_out=p.nside)
                    ionRM_out[i] = irf.rotate_healpix_mapHPX(temp1,[-np.degrees(RotAngle)])

            elif p.ionosphere_type == 'constant':
                print "d is " + str(d)
                # ionRM_out = np.ones((p.nhours, p.npix))
                # # p.RMs.append(np.random.rand(1)[0] * 2. * np.pi)
                # RM_use = p.RM_sigma * np.random.randn(1)[0]
                # p.RMs.append(RM_use)
                #
                # ionRM_out *= p.RMs[d]
                # ionRM_index = [0 for x in range(p.ntime)]

                ionRM_out = p.RMs[d] * np.ones((p.nhours, p.npix))
                ionRM_index = [0 for x in range(p.ntime)]


        else:
            ionRM_out = None
            ionRM_index = range(p.ntime)

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

    if p.unpolarized == False and p.ionosphere == False and p.ndays > 1:
        raise Exception('Multiple days are set for a non-ionosphere run.')

    if p.instrument == 'paper':
        out_dir = os.path.join(p.output_base_directory, 'PAPER/', p.output_directory)
    elif p.instrument == 'paper_hfss':
        out_dir = os.path.join(p.output_base_directory, 'PAPER/', p.output_directory)
    elif p.instrument == 'paper_feko':
        out_dir = os.path.join(p.output_base_directory, 'PAPER/', p.output_directory)
    else:
        out_dir = os.path.join(p.output_base_directory, 'HERA/', p.output_directory)

    while os.path.exists(out_dir):
        out_dir = out_dir[:-1] + "B/"

    p.out_dir_use = out_dir

    os.makedirs(p.out_dir_use)
    from shutil import copyfile
    copyfile('parameters.yaml', p.out_dir_use + 'parameters.yaml')
    print "Parameters file copied."

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
