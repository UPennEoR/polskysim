import numpy as np
import healpy as hp
import os
from scipy import interpolate
import ionRIME_funcs as irf
import sys
import time
import numba_funcs as irnf
from sky_setup import SkyConstructor
from instrument_setup import InstrumentConstructor
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

class VisibilitySimulation(object):
    """
    A class that implements a visibility computation as a function of the set of
    input parameters.
    """

    def __init__(self, parameters):

        # parameters is a dictionary of variables read from the parameters.yaml file
        for key in parameters:
            setattr(self, key, parameters[key])

        self.npix = hp.nside2npix(self.nside)
        self.lmax = 3 * self.nside -1
        self.nu_axis = np.linspace(self.nu_0, self.nu_f, num=self.nfreq, endpoint=True)
        self.hpxidx = np.arange(self.npix)

        self.l, self.m = hp.Alm.getlm(self.lmax)

        attribute_test = getattr(self, 'z0_cza',None)
        if attribute_test is None:
            self.z0_cza = np.radians(120.7215) # latitude of HERA/PAPER
            parameters.update({'z0_cza':self.z0_cza})

        ## Sky
        ## sky.shape = (self.nfreq, self.npix, 2,2)

        print "Setting up sky."
        sky_init = SkyConstructor(parameters)
        self.I, self.Q, self.U, self.V = sky_init.stokes_parameters

        if self.point_source_sim is False:
            self.I_alm, self.Q_alm, self.U_alm = [map2alm(S, self.lmax) for S in [self.I, self.Q, self.U]]
            if self.circular_pol is True:
                self.V_alm = map2alm(self.V, self.lmax)

        ## Instrument
        ## ijones.shape = (self.nfreq, self.npix, 2, 2)

        tmark0 = time.time()

        print "Setting up instrument."
        instrument_init = InstrumentConstructor(parameters)
        self.ijones = instrument_init.iJonesModel

        self.ijonesH = np.transpose(self.ijones.conj(),(0,1,3,2))

        tmark_inst = time.time()
        print "Completed instrument setup in " + str(tmark_inst - tmark0)

        ## Fringe
        # K.shape = (self.nfreq,self.npix)

        c = 299792458. # meters / sec
        b = irf.transform_baselines(self.baseline) # meters, in the Equatorial basis
        s = hp.pix2vec(self.nside, self.hpxidx)
        b_dot_s = np.einsum('a...,a...',b,s)
        tau = b_dot_s / c

        self.K = np.exp(-2. * np.pi * 1j * np.outer(np.ones(self.nfreq), tau) * np.outer(self.nu_axis, np.ones(self.npix)) )

        if self.ionosphere == 'none':
            self.ndays = 1

        self.Vis = np.zeros(self.ndays * self.ntime * self.nfreq * 2 * 2, dtype='complex128')
        self.Vis = self.Vis.reshape(self.ndays, self.ntime, self.nfreq, 2, 2)

        self.RMs = []

        self.ra0 = np.radians(self.hour_offset * 15.) # the starting RA of the simulation window i.e. when t=0, p.ra0 is at the beam's zenith meridian

    def run(self):
        if self.final_day_average is False:
            self.compute_visibility_for_each_day()
        else:
            self.compute_average_visibility()

    def compute_visibility_for_each_day(self):
        tmark_loopstart = time.time()

        if self.ionosphere == 'radionopy':
            d0 = (self.sim_part-1)*self.ndays
            d1 = (self.sim_part-1)*self.ndays + self.ndays
            date_strs = [irf.get_time_string(x, self.day0) for x in range(d0-1,d1+1)]

            UT_hour_start_str = radiono.utils.nextTransit(date_strs[1], np.degrees(self.ra0),0.).split(' ')[1] # an extra day has been added before "day0",
            UT_hour_start = int(UT_hour_start_str[:2]) # the UT hour of the first RM map on the first day

            print "Getting ionosphere data..."
            heraRM = radiono.rm.HERA_RM(date_strs)
            heraRM.make_radec_RM_maps() # compute all the radionopy RM maps that will be used. The RM array for each day is only ~0.7 MB.

        elif self.ionosphere == 'constant':
            RMs_npz_path = '/data4/paper/zionos/polskysim/IonRIME/RM_sim_data/'
            RMs_npz_path += self.RMs_sequence_file_name

            RMs_npz = np.load(RMs_npz_path)
            self.RMs = RMs_npz['RMs']

        for d in range(self.ndays):
            if self.ionosphere == 'radionopy':

                print "d is " + str(d) + ", day is " + date_strs[d+1]

                ds = d + (self.sim_part-1)*self.ndays

                bins_per_hour = float(self.ntime)/self.nhours
                hours_per_degree = 1./15.     # 1 deg ~= 1/15 hours
                bins_per_degree = bins_per_hour * hours_per_degree

                dec_part, _ = math.modf(bins_per_degree)

                frac = Fraction(dec_part).limit_denominator(max_denominator=100)

                time_bin_shift = ds * int(bins_per_degree) + ds/frac.denominator

                ionRM_index = [(x * self.nhours/self.ntime) + UT_hour_start + 24 for x in range(-time_bin_shift, self.ntime)] # the +24 is to shift the UTs to the middle of the heraRM_use axis

                ionRM_index = np.roll(ionRM_index, time_bin_shift)[-self.ntime:]

                UT_index = list(set(ionRM_index))
                UT_index.sort()

                ionRM_index = [x - min(ionRM_index) for x in ionRM_index]
                self.ionRM_index = ionRM_index

                self.ionRM_out = np.zeros((self.nhours+1, self.npix))
                heraRM_use = np.concatenate((heraRM.RMs[d,:,:], heraRM.RMs[d+1,:,:]), axis=0) # 48 hours of RM maps, starting at midnight on day "d"

                if max(UT_index) > 47:
                    heraRM_use = np.concatenate((heraRM_use, heraRM.RMs[d+2,:,:]), axis=0)

                for i, hr in enumerate(UT_index):
                    ##OLD
                    # hrAngle = -p.ra0 + np.radians(hr * 15.) + np.pi # approximately true?
                    # lh,mh = hp.Alm.getlm(3*heraRM.nside -1)
                    # mh_rot = np.exp(1j * mh * hrAngle)
                    # ionRM_out[i] = hp.alm2map(hp.map2alm(heraRM_use[hr,:], lmax=3*heraRM.nside -1)* mh_rot, p.nside, verbose=False)

                    RotAngle = -self.ra0 - np.radians((hr-2) * 15.)
                    # the (hr-2) has something to do with the local time zone being UT+2.
                    # Found that it should be (hr-2) by trial-and-error matching the radionopy altaz map...
                    temp1 = hp.ud_grade(heraRM_use[hr,:], nside_out=self.nside)
                    self.ionRM_out[i] = irf.rotate_healpix_mapHPX(temp1,[-np.degrees(RotAngle)])

            elif self.ionosphere == 'constant':
                print "d is " + str(d)
                # ionRM_out = np.ones((p.nhours, p.npix))
                # # p.RMs.append(np.random.rand(1)[0] * 2. * np.pi)
                # RM_use = p.RM_sigma * np.random.randn(1)[0]
                # p.RMs.append(RM_use)
                #
                # ionRM_out *= p.RMs[d]
                # ionRM_index = [0 for x in range(p.ntime)]

                self.ionRM_out = self.RMs[d] * np.ones((self.nhours, self.npix))
                ionRM_index = [0 for x in range(self.ntime)]


            else:
                self.ionRM_out = None
                ionRM_index = range(self.ntime)

            for t, h in enumerate(ionRM_index):
                if self.point_source_sim is True:
                    self.from_point_sources(d,t,h)
                else:
                    self.from_healpix_grid(d,t,h)

        self.Vis /= self.npix # normalization
        tmark_loopstop = time.time()

        print "Visibility loop completed in " + str(tmark_loopstop - tmark_loopstart)

    def compute_average_visibility(self):
        if self.ionosphere == 'none':
            raise Exception('Idiocy')

        tmark_loopstart = time.time()

        UT_hour_start_str = radiono.utils.nextTransit(date_strs[1], np.degrees(self.ra0),0.).split(' ')[1] # an extra day has been added before "day0",
        UT_hour_start = int(UT_hour_start_str[:2]) # the UT hour of the first RM map on the first day

        if self.ionosphere == 'radionopy':
            d0 = (self.sim_part-1)*self.ndays
            d1 = (self.sim_part-1)*self.ndays + self.ndays
            date_strs = [irf.get_time_string(x, self.day0) for x in range(d0-1,d1+1)]

            heraRM = radiono.rm.HERA_RM(date_strs)
            heraRM.make_radec_RM_maps() # compute all the radionopy RM maps that will be used. The RM array for each day is only ~0.7 MB.

        elif self.ionosphere == 'constant':
            RMs_npz_path = '/data4/paper/zionos/polskysim/IonRIME/RM_sim_data/' +self.RMs_sequence_file_name
            RMs_npz = np.load(RMs_npz_path)
            self.RMs = RMs_npz['RMs']

        self.RM_series = np.zeros((self.ntime, self.ndays, self.npix))

        for d in range(self.ndays):
            if self.ionosphere == 'radionopy':

                print "d is " + str(d) + ", day is " + date_strs[d+1]

                ds = d + (self.sim_part-1)*self.ndays

                bins_per_hour = float(self.ntime)/self.nhours
                hours_per_degree = 1./15.     # 1 deg ~= 1/15 hours
                bins_per_degree = bins_per_hour * hours_per_degree

                dec_part, _ = math.modf(bins_per_degree)

                frac = Fraction(dec_part).limit_denominator(max_denominator=100)

                time_bin_shift = ds * int(bins_per_degree) + ds/frac.denominator

                ionRM_index = [(x * self.nhours/self.ntime) + UT_hour_start + 24 for x in range(-time_bin_shift, self.ntime)] # the +24 is to shift the UTs to the middle of the heraRM_use axis

                ionRM_index = np.roll(ionRM_index, time_bin_shift)[-self.ntime:]

                UT_index = list(set(ionRM_index))
                UT_index.sort()

                ionRM_index = [x - min(ionRM_index) for x in ionRM_index]
                self.ionRM_index = ionRM_index

                self.ionRM_out = np.zeros((p.nhours+1, p.npix))
                heraRM_use = np.concatenate((heraRM.RMs[d,:,:], heraRM.RMs[d+1,:,:]), axis=0) # 48 hours of RM maps, starting at midnight on day "d"

                if max(UT_index) > 47:
                    heraRM_use = np.concatenate((heraRM_use, heraRM.RMs[d+2,:,:]), axis=0)

                RM_window = heraRM_use[UT_index]

                for t, hr in enumerate(ionRM_index):
                    self.RM_series[t,d,:] = RM_window[hr]

            elif self.ionosphere == 'constant':
                print "d is " + str(d)
                # ionRM_out = np.ones((p.nhours, p.npix))
                # # p.RMs.append(np.random.rand(1)[0] * 2. * np.pi)
                # RM_use = p.RM_sigma * np.random.randn(1)[0]
                # p.RMs.append(RM_use)
                #
                # ionRM_out *= p.RMs[d]
                # ionRM_index = [0 for x in range(p.ntime)]

                self.ionRM_out = self.RMs[d] * np.ones((self.nhours, self.npix))
                ionRM_index = [0 for x in range(self.ntime)]

                for t, hr in enumerate(ionRM_index):
                    self.RM_series[t,d,:] = self.ionRM_out[hr]

        d = 1
        for t in range(self.ntime):
            self.from_healpix_grid(d,t,h)

        self.Vis /= self.npix # normalization
        tmark_loopstop = time.time()

    def from_healpix_grid(self,d,t,h):

        print "t is " + str(t)
        total_angle = float(self.nhours * 15) # degrees
        offset_angle = float(self.hour_offset * 15) # degrees
        zl_ra = ((float(t) / float(self.ntime)) * np.radians(total_angle) + np.radians(offset_angle)) % (2*np.pi) # radians

        # npix = hp.nside2npix(p.nside)

        RotAxis = np.array([0.,0.,1.])
        RotAngle = zl_ra

        mrot = np.exp(1j * self.m * RotAngle)

        It, Qt, Ut = [np.zeros((self.nfreq, self.npix)) for n in range(3)]

        for fi in range(self.nfreq):
            It[fi] = hp.alm2map(self.I_alm[fi] * mrot, self.nside, verbose=False)
            Qt[fi] = hp.alm2map(self.Q_alm[fi] * mrot, self.nside, verbose=False)
            Ut[fi] = hp.alm2map(self.U_alm[fi] * mrot, self.nside, verbose=False)

        ## Ionosphere
        """
        ionRM.shape = (self.nhours, self.nfreq, self.npix)
        ionRM_t.shape = (self.nfreq, self.npix)
        """
        if self.ionosphere != 'none':
            if self.final_day_average is False:
                ionRM_t = self.ionRM_out[h] # pick out the map corresponding to this hour

                c = 299792458. # meters / sec
                lbda2 = (c / self.nu_axis)**2.
                ionAngle = np.outer(lbda2, np.ones(self.npix)) * ionRM_t

                ion_cos2 = irnf.numbap_cos(2. * ionAngle) # numba can multithread a numpy ufunc, no fuss no muss!
                ion_sin2 = irnf.numbap_sin(2. * ionAngle)

                QUout = irnf.complex_rotation(Qt,Ut, ion_cos2, ion_sin2)
                Qt = QUout.real
                Ut = QUout.imag

            elif self.final_day_average is True:
                ionRM_t = self.RM_series[t]

                c = 299792458. # meters / sec
                lbda2 = (c / self.nu_axis)**2.
                # ionAngle = np.outer(lbda2, np.ones(self.npix)) * ionRM_t

                ionAngle = np.zeros((self.ndays, self.nfreq, self.npix))

                for k in range(self.ndays):
                    ionAngle[k] = np.outer(lbda2, np.ones(self.npix)) * ionRM_t[k]

                A_re = np.mean(irnf.numbap_cos(2. * ionAngle), axis=0)
                A_im = np.mean(irnf.numbap_sin(2. * ionAngle), axis=0)

                QUout = irnf.complex_rotation(Qt,Ut, A_re, A_im)
                Qt = QUout.real
                Ut = QUout.imag

        sky_t = np.array([
            [It + Qt, Ut],
            [Ut, It - Qt]]).transpose(2,3,0,1) # sky_t.shape = (p.nfreq, p.npix, 2, 2)

        irnf.instrRIME_integral(self.ijones, sky_t, self.ijonesH, self.K, self.Vis[d,t,:,:,:].squeeze())

    def from_point_sources(self,d,t,h):
        # if p.circular_pol == True:
        #     I,Q,U,V = sky_list
        # else:
        #     I,Q,U = sky_list

        print "t is " + str(t)
        total_angle = float(self.nhours * 15) # degrees
        offset_angle = float(self.hour_offset * 15) # degrees
        zl_ra = ((float(t) / float(self.ntime)) * np.radians(total_angle) + np.radians(offset_angle)) % (2.*np.pi)# radians

        RotAxis = np.array([0.,0.,1.])
        RotAngle = -zl_ra # basicly Hour Angle with t=0

        # R_t = irf.rotation_matrix(RotAxis, RotAngle)
        # s0 = hp.ang2vec(src_cza,src_ra)
        # sf = np.einsum('...ab,...b->...b', R_t, s0)

        src_ra_f = self.src_ra + RotAngle
        sf = hp.ang2vec(self.src_cza, self.src_ra_f)

        # s0 = hp.ang2vec(src_cza,src_ra)
        # hpR_t = hp.Rotator(rot=[RotAngle])

        # sf = np.array(hpR_t(s0[:,0],s0[:,1],s0[:,2])).T

        if len(sf.shape) > 1:
            inds_f = hp.vec2pix(self.nside, sf[:,0],sf[:,1],sf[:,2])
        else:
            inds_f = hp.vec2pix(self.nside, sf[0],sf[1],sf[2])

        It = self.I #XXX ??

        ijones_t = self.ijones[:,inds_f,:,:]
        ijonesH_t = self.ijonesH[:,inds_f,:,:]
        Kt = self.K[:,inds_f]

        if p.unpolarized == True:
            Ut = np.zeros((p.nfreq,It.shape[-1]))

            sky_t = np.array([
                [It, Ut],
                [Ut, It]]).transpose(2,3,0,1) # sky_t.shape = (p.nfreq, p.npix, 2, 2)

        else:
            Ut = self.U
            Qt = self.Q

            sky_t = np.array([
                [It + Qt, Ut],
                [Ut, It - Qt]]).transpose(2,3,0,1) # sky_t.shape = (p.nfreq, p.npix, 2, 2)

        irnf.instrRIME_integral(ijones_t, sky_t, ijonesH_t, Kt, self.Vis[d,t,:,:,:].squeeze())
