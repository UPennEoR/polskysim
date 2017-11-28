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
from astropy.time import Time, TimeDelta

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
        # older input parameters files will not have an instrument_b setting
        attribute_test = getattr(self, 'instrument_b',None)
        if attribute_test is None:
            self.instrument_b = 'same'
            parameters.update({'instrument_b':self.instrument_b})


        ## Sky
        ## sky.shape = (self.nfreq, self.npix, 2,2)

        print "Setting up sky."
        sky_init = SkyConstructor(parameters)
        self.I, self.Q, self.U, self.V = sky_init.stokes_parameters

        if self.point_source_sim is False:
            self.I_alm, self.Q_alm, self.U_alm = [map2alm(S, self.lmax) for S in [self.I, self.Q, self.U]]
            if self.circular_pol is True:
                self.V_alm = map2alm(self.V, self.lmax)

        if self.point_source_sim is True:
            self.src_ra = sky_init.src_ra
            self.src_cza = sky_init.src_cza

        ## Instrument
        ## ijones.shape = (self.nfreq, self.npix, 2, 2)

        tmark0 = time.time()

        print "Setting up instrument."
        instrument_a_init = InstrumentConstructor(parameters, antenna='a')
        self.ijones = instrument_a_init.iJonesModel

        instrument_b_init = InstrumentConstructor(parameters, antenna='b')
        self.ijonesH = np.transpose(instrument_b_init.iJonesModel.conj(),(0,1,3,2))

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
        if self.final_day_average is False:
            self.Vis = np.zeros(self.ndays * self.ntime * self.nfreq * 2 * 2, dtype='complex128')
            self.Vis = self.Vis.reshape(self.ndays, self.ntime, self.nfreq, 2, 2)
        else:
            self.Vis = np.zeros(1 * self.ntime * self.nfreq * 2 * 2, dtype='complex128')
            self.Vis = self.Vis.reshape(1, self.ntime, self.nfreq, 2, 2)

        self.RMs = []

        self.ra0 = np.radians(self.hour_offset * 15.) # the starting RA of the simulation window i.e. when t=0, p.ra0 is at the beam's zenith meridian

    def run(self):
        self.compute_visibilities()

    def compute_visibilities(self):
        tmark_loopstart = time.time()

        if self.ionosphere == 'radionopy':

            init_transit_times = []
            day0_str = '-'.join([str(x) for x in self.day0])
            ra0 = np.radians(15. * self.hour_offset)
            raf = ra0 + np.radians(15. * self.nhours)
            ra_axis = np.linspace(ra0, raf, self.ntime)

            for ra in ra_axis:
                td = radiono.utils.nextTransit(day0_str, np.degrees(ra), 0.)
                td = radiono.utils.eph2ionDate(td)
                init_transit_times.append(Time(td, format='iso', scale='utc'))

            self.UT_times = []
            for di in range(self.ndays):
                self.UT_times.append([])
                for ti in range(self.ntime):
                    td = str(init_transit_times[ti] + di * TimeDelta(1. * units.sday))
                    self.UT_times[di].append(td)

            UT_strs = [item for sublist in self.UT_times for item in sublist] # flattened

            tmark_ion_start = time.time()
            print "Getting ionosphere data..."
            heraRM = radiono.rm.HERA_RM(UT_strs)
            heraRM.calc_ionRIME_rm()

            self.RMs = heraRM.RMs

            tmark_ion_stop = time.time()

            print "Ionosphere data aquirred in " + str(tmark_ion_stop - tmark_ion_start)


        elif self.ionosphere == 'constant':
            if os.path.exists('/data4/paper/zionos/'):
                RMs_npz_path = '/data4/paper/zionos/polskysim/IonRIME/RM_sim_data/'
                RMs_npz_path += self.RMs_sequence_file_name
            elif os.path.exists('/lustre/aoc/projects/hera/zmartino/zionos/'):
                RMs_npz_path = '/lustre/aoc/projects/hera/zmartino/zionos/polskysim/IonRIME/RM_sim_data/'
                RMs_npz_path += self.RMs_sequence_file_name

            RMs_npz = np.load(RMs_npz_path)
            self.RMs = {k:RM for k, RM in enumerate(RMs_npz['RMs'])}
            self.UT_times = []
            for di in range(self.ndays):
                self.UT_times.append([])
                for ti in range(self.ntime):
                    self.UT_times[di].append(di)

        if self.final_day_average is True:
            Nd_use = 1
        else:
            Nd_use = self.ndays

        for d in range(Nd_use):
            print "d is " + str(d)
            for t in range(self.ntime):
                if self.point_source_sim is True:
                    self.from_point_sources(d,t)
                else:
                    self.from_healpix_grid(d,t)

        self.Vis *= (4. * np.pi / self.npix)
        tmark_loopstop = time.time()

        print "Visibility loop completed in " + str(tmark_loopstop - tmark_loopstart)

    def from_healpix_grid(self,d,t):

        print "t is " + str(t)
        total_angle = float(self.nhours * 15) # degrees
        offset_angle = float(self.hour_offset * 15) # degrees
        zl_ra = ((float(t) / float(self.ntime)) * np.radians(total_angle) + np.radians(offset_angle)) % (2*np.pi) # radians

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

                if self.ionosphere == 'constant':
                    ionRM_t = self.RMs[self.UT_times[d][t]] * np.ones(self.npix)
                else:
                    ionRM_t = self.RMs[self.UT_times[d][t]]
                    ionRM_t = hp.alm2map(hp.map2alm(ionRM_t), self.nside, verbose=False)

                c = 299792458. # meters / sec
                lbda2 = (c / self.nu_axis)**2.
                ionAngle = np.outer(lbda2, np.ones(self.npix)) * ionRM_t

                ion_cos2 = irnf.numbap_cos(2. * ionAngle) # numba can multithread a numpy ufunc, no fuss no muss!
                ion_sin2 = irnf.numbap_sin(2. * ionAngle)

                QUout = irnf.complex_rotation(Qt,Ut, ion_cos2, ion_sin2)
                Qt = QUout.real
                Ut = QUout.imag

            elif self.final_day_average is True:

                if self.ionosphere == 'constant':
                    ionRM_t = self.RMs[self.UT_times[d][t]] * np.ones(self.npix)
                else:
                    ionRM_t = self.RMs[self.UT_times[d][t]]
                    ionRM_t = hp.alm2map(hp.map2alm(ionRM_t), self.nside, verbose=False)

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

    def from_point_sources(self,d,t):
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

        self.src_ra_f = self.src_ra + RotAngle
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

        if self.unpolarized == True:
            Ut = np.zeros((self.nfreq,It.shape[-1]))

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
