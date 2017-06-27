#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:38:12 2017

@author: jaguirre
"""

import numpy as np
from astropy import units as u
from astropy import constants as c
from astropy import cosmology
from scipy.special import spherical_jn as jn
import pylab as plt

cosmo = cosmology.Planck15

def k_para_rm(nu,RM):
    lmbda = (c.c/nu).to(u.m)
    z = (1420.*u.MHz/nu).to(u.dimensionless_unscaled) - 1.
    k_para = 4.*cosmo.H(z)/c.c/(1+z)*RM*np.power(lmbda,2)
    print lmbda
    print z
    
    return k_para.to(u.Mpc**-1)


nu = np.linspace(100,200,num=1000)*u.MHz
tau = 30*u.m/c.c

x = (2*np.pi*(tau*nu).to(u.dimensionless_unscaled)).value

plt.figure(1)
plt.clf()
plt.plot(nu.value,jn(0,x))
plt.plot(nu.value,jn(1,x))
plt.plot(nu.value,jn(2,x))
plt.plot(nu.value,jn(3,x))

sm = np.array(jn(1,x),dtype='complex128')
for l in np.arange(2,10):
    sm += np.power(-1j,l)*jn(l,x)