# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 17:21:16 2016

@author: jaguirre
"""

import numpy as np
#import pylab as plt
import matplotlib.pyplot as plt
from astropy import units as u

from astropy.io import fits
from astropy.table import Table

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', weight='normal')

def bin_centers(bin_edges):
    """ With histograms, sometimes more useful to plot at the 
    center than to have the edges """
    db = np.diff(bin_edges)
    bc = bin_edges[0:-1] + db
    return bc

# Supposedly a lot of this is in David's PolSim, but it's buried down.
def dNdS_models(S,model='6C'):
    if model=='6C':
        S0 = 0.88  
        dNdS = np.zeros_like(S)
        dNdS[S<S0] = 4000.*np.power(S0,-0.76)*np.power(S[S<S0],-1.75)
        dNdS[S>=S0] = 4000. * np.power(S[S>=S0],-2.81)
    if model == 'VLSS':
        dNdS = 4865.*np.power(S,-2.3)
    return dNdS

#%%
f = open('franzen16.txt')
print f.readline()
tmp = f.readline()
bin_start = np.array(tmp.split(),dtype='float64')
print f.readline()
tmp = f.readline()
bin_end = np.array(tmp.split(),dtype='float64')
print f.readline()
tmp = f.readline()
bc_F16 = np.array(tmp.split(),dtype='float64')
print f.readline()
tmp = f.readline()
n_srcs = np.array(tmp.split(),dtype='float64')
print f.readline()
tmp = f.readline()
corr_fac = np.array((tmp.split())[0:len(tmp):3],dtype='float64')
corr_fac_err = np.array((tmp.split())[2:len(tmp):3],dtype='float64')
print f.readline()
tmp = f.readline()
Euc_cnts = np.array((tmp.split())[0:len(tmp):3],dtype='float64')
Euc_cnts_err = np.array((tmp.split())[2:len(tmp):3],dtype='float64')
f.close()
dNdS_F16 = Euc_cnts / np.power(bc_F16,2.5)
#%%

# convert per steradian to per square degree
sqdeg_per_ster = ((1*u.steradian).to(np.power(u.deg,2))).value
hw14file = 'hurley-walker14.fit'
t = Table.read(hw14file)

flux_hw14 = t['S180']
nsrc_hw14 = flux_hw14.size
area_hw14 = 6100./sqdeg_per_ster # square degrees
# Reasonable limits, in Jy
S_min = 0.010
S_max = 100
lb_hw14=np.logspace(np.log10(S_min), np.log10(S_max), num=100)
ps_hw14,be_hw14=np.histogram(flux_hw14, bins=lb_hw14,normed=True)
dnds_hw14 = ps_hw14 * nsrc_hw14 / area_hw14
bc_hw14 = bin_centers(be_hw14)

dNdS_6C = dNdS_models(bc_hw14,model='6C')
dNdS_VLSS = dNdS_models(bc_hw14,model='VLSS')

#%%
plt.figure(1,figsize=(8,10))
#plt.clf()
plt.subplot(311)
plt.loglog(bc_hw14,dnds_hw14,label='H-W14')
plt.loglog(bc_hw14,dNdS_6C,label='6C')
plt.loglog(bc_hw14,dNdS_VLSS,label='VLSS')
plt.loglog(bc_F16,dNdS_F16,'m',label='F16',linewidth=2)
plt.xlabel('Flux [Jy]')
plt.ylabel(r'$dN/dS~[\mathrm{Jy}^{-1} \mathrm{sr}^{-1}]$')
plt.legend()
plt.ylim([1,1e8])

plt.subplot(312)
plt.loglog(bc_hw14,np.power(bc_hw14,2)*dnds_hw14,label='H-W14')
plt.loglog(bc_hw14,np.power(bc_hw14,2)*dNdS_6C,label='6C')
plt.loglog(bc_hw14,np.power(bc_hw14,2)*dNdS_VLSS,label='VLSS')
plt.loglog(bc_F16,np.power(bc_F16,2)*dNdS_F16,'m',label='F16',linewidth=2)
plt.xlabel('Flux [Jy]')
plt.ylabel(r'$S^2 dN/dS~[\mathrm{Jy~sr}^{-1}]$')
plt.ylim([1e2,1e5])

plt.subplot(313)
plt.loglog(bc_hw14,np.power(bc_hw14,2.5)*dnds_hw14,label='H-W14')
plt.loglog(bc_hw14,np.power(bc_hw14,2.5)*dNdS_6C,label='6C')
plt.loglog(bc_hw14,np.power(bc_hw14,2.5)*dNdS_VLSS,label='VLSS')
plt.loglog(bc_F16,Euc_cnts,'m',label='F16',linewidth=2)
plt.xlabel('Flux [Jy]')
plt.ylabel(r'$S^{2.5} dN/dS}~[\mathrm{Jy}^{1.5} \mathrm{sr}^{-1}]$')
plt.ylim([1e2,1e5])

#plt.savefig('number_counts.png',dpi=300)
plt.show()




