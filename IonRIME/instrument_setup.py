import numpy as np
import healpy as hp
import ionRIME_funcs as irf
import os
import sys
import errno

import hera_NicCST_instrument_setup as hnis
import hera_CTP_instrument_setup as hcis
import hera_vivaldi_instrument_setup as hvis
import hera_nicsinuous_instrument_setup as hnsis
import hera_hfss_instrument_setup as hhis
import paper_hfss_instrument_setup as phis
import legacy_cst_instrument_setup as lcis
import sim_group_test0_instrument_setup as sgtest0


class InstrumentConstructor(object):
    def __init__(self, parameters, antenna='a'):
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

        parameters.update({
            'npix':self.npix,
            'lmax':self.lmax,
            'nu_axis': self.nu_axis,
            'hpxidx': self.hpxidx,
            'l': self.l,
            'm': self.m,
            'z0_cza': self.z0_cza})

        if antenna == 'b':
            if self.instrument_b != 'same':
                self.antenna_model = self.instrument_b
            else:
                self.antenna_model = self.instrument
        else:
            self.antenna_model = self.instrument

        nu0 = str(int(self.nu_axis[0] / 1e6))
        nuf = str(int(self.nu_axis[-1] / 1e6))
        self.fname = "ijones" + "band_" + nu0 + "-" + nuf + "mhz_nfreq" + str(self.nfreq)+ "_nside" + str(self.nside) + ".npy"

        self.InstrumentDirectories = {
            'hera_NicCST': 'jones_save/HERA_NicCST/',
            'hera_ctp': 'jones_save/HERA_CTP/',
            'hera_vivaldi': 'jones_save/HERA_Vivaldi/',
            'hera_nicsinuous': 'jones_save/HERA_NicSinuous/',
            'hera_hfss': 'jones_save/HERA_HFSS/',
            'paper_hfss': 'jones_save/PAPER_HFSS/',
            'paper_feko': 'jones_save/PAPER/',
            'hera_legacy': 'jones_save/',
            'analytic_dipole': 'analytic_dipole/',
            'airy_dipole': 'jones_save/airy_dipole/',
            'sim_group_test0': 'jones_save/sim_group_test0/'
        }

        self.relative_path = self.InstrumentDirectories[self.antenna_model] + self.fname

        self.InstrumentGenerators = {
            'hera_NicCST': self.hera_NicCST,
            'hera_ctp': self.hera_ctp,
            'hera_vivaldi': self.hera_vivaldi,
            'hera_nicsinuous': self.hera_nicsinuous,
            'hera_hfss': self.hera_hfss,
            'paper_hfss': self.paper_hfss,
            'paper_feko': self.paper_feko,
            'hera_legacy': self.hera_legacy,
            'analytic_dipole': self.analytic_dipole,
            'airy_dipole': self.airy_dipole,
            'sim_group_test0': self.sim_group_test0,
        }

        try:
            os.makedirs(self.InstrumentDirectories[self.antenna_model])
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        if os.path.exists(self.relative_path) is True:
            self.iJonesModel = np.load(self.relative_path)
            print "Restored Jones model"
        else:
            self.iJonesModel = self.InstrumentGenerators[self.antenna_model](parameters)
            if self.antenna_model != 'analytic_dipole':
                np.save(self.relative_path, self.iJonesModel)


    def hera_NicCST(self, parameters):
        ijones, solid_angle, peak_norms = hnis.make_ijones_spectrum(parameters, verbose=True)

        nu0 = str(int(self.nu_axis[0] / 1e6))
        nuf = str(int(self.nu_axis[-1] / 1e6))
        fname = "norms_ijones" + "band_" + nu0 + "-" + nuf + "mhz_nfreq" + str(self.nfreq)+ "_nside" + str(self.nside) + '.npz'

        save_path = self.InstrumentDirectories[self.antenna_model] + fname
        np.savez(save_path, solid_angle=solid_angle, peak_norms=peak_norms)
        return ijones

    def hera_ctp(self, parameters):
        ijones, solid_angle, peak_norms = hcis.make_ijones_spectrum(parameters, verbose=True)

        nu0 = str(int(self.nu_axis[0] / 1e6))
        nuf = str(int(self.nu_axis[-1] / 1e6))
        fname = "norms_ijones" + "band_" + nu0 + "-" + nuf + "mhz_nfreq" + str(self.nfreq)+ "_nside" + str(self.nside) + '.npz'

        save_path = self.InstrumentDirectories[self.antenna_model] + fname
        np.savez(save_path, solid_angle=solid_angle, peak_norms=peak_norms)
        return ijones

    def hera_vivaldi(self, parameters):
        ijones, solid_angle, peak_norms = hvis.make_ijones_spectrum(parameters, verbose=True)

        nu0 = str(int(self.nu_axis[0] / 1e6))
        nuf = str(int(self.nu_axis[-1] / 1e6))
        fname = "norms_ijones" + "band_" + nu0 + "-" + nuf + "mhz_nfreq" + str(self.nfreq)+ "_nside" + str(self.nside) + '.npz'

        save_path = self.InstrumentDirectories[self.antenna_model] + fname
        np.savez(save_path, solid_angle=solid_angle, peak_norms=peak_norms)
        return ijones

    def hera_nicsinuous(self, parameters):
        ijones, solid_angle, peak_norms = hnsis.make_ijones_spectrum(parameters, verbose=True)

        nu0 = str(int(self.nu_axis[0] / 1e6))
        nuf = str(int(self.nu_axis[-1] / 1e6))
        fname = "norms_ijones" + "band_" + nu0 + "-" + nuf + "mhz_nfreq" + str(self.nfreq)+ "_nside" + str(self.nside) + '.npz'

        save_path = self.InstrumentDirectories[self.antenna_model] + fname
        np.savez(save_path, solid_angle=solid_angle, peak_norms=peak_norms)
        return ijones

    def hera_hfss(self, parameters):
        return hhis.make_ijones_spectrum(parameters, verbose=False)

    def paper_hfss(self, parameters):
        return phis.make_ijones_spectrum(parameters, verbose=True)

    def paper_feko(self, parameters):
        return irf.PAPER_instrument_setup(parameters, self.z0_cza)

    def hera_legacy(self, parameters):
        freqs = [x * 1e6 for x in range(140,171)] # Hz
        Jdata = lcis.instrument_setup(self.z0_cza, freqs)
        return lcis.interpolate_jones_freq(Jdata, freqs, interp_type=self.interp_type)

    def analytic_dipole(self, parameters):
        return irf.analytic_dipole_setup(self.nside, self.nfreq,sigma=self.dipole_hpbw, z0_cza=self.z0_cza)

    def airy_dipole(self, parameters):
        a = 14.6/2. # HERA dish radius ...or was it 14.0?
        return irf.airy_dipole_setup(self.nside, self.nu_axis, a, z0_cza=self.z0_cza)

    def sim_group_test0(self, parameters):
        return sgtest0.make_ijones_spectrum(parameters)
