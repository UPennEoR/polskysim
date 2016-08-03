"""
An importable module adaptation of the command-line script create_map.py in
https://github.com/jeffzheng/gsm2016/
"""

import numpy as np
import healpy as hp
import sys, os

script_path = os.path.dirname(os.path.realpath(__file__))
labels = ['Synchrotron', 'CMB', 'HI', 'Dust1', 'Dust2', 'Free-Free']
n_comp = len(labels)
kB = 1.38065e-23
C = 2.99792e8
h = 6.62607e-34
T = 2.725
hoverk = h / kB

def get_gsm_map_lowres(frequency):
    freq = frequency
    nside = 64
    unit = 'MJysr'
    convert_ring = 'True'

    map_ni = np.loadtxt(script_path + '/data/lowres_maps.txt')

    spec_nf = np.loadtxt(script_path + '/data/spectra.txt')

    nfreq = spec_nf.shape[1]

    left_index = -1
    for i in range(nfreq - 1):
        if freq >= spec_nf[0, i] and freq <= spec_nf[0, i + 1]:
            left_index = i
            break
    if left_index < 0:
        print "FREQUENCY ERROR: %.2e GHz is outside supported frequency range of %.2e GHz to %.2e GHz."%(freq, spec_nf[0, 0], spec_nf[0, -1])

    interp_spec_nf = np.copy(spec_nf)
    interp_spec_nf[0:2] = np.log10(interp_spec_nf[0:2])
    x1 = interp_spec_nf[0, left_index]
    x2 = interp_spec_nf[0, left_index + 1]
    y1 = interp_spec_nf[1:, left_index]
    y2 = interp_spec_nf[1:, left_index + 1]
    x = np.log10(freq)
    interpolated_vals = (x * (y2 - y1) + x2 * y1 - x1 * y2) / (x2 - x1)
    result = np.sum(10.**interpolated_vals[0] * (interpolated_vals[1:, None] * map_ni), axis=0)

    result = hp.reorder(result, n2r=True)

    result *= 1e6 # convert from MJysr to Jysr

    return result
