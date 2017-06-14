import numpy as np, healpy as hp
import csv, sys, os.path,re
import matplotlib.pyplot as plt
from scipy import interpolate
import aipy as ap
import ionRIME_funcs as irf
import numba_funcs as irnf

class Parameters:
    def __init__(self, param_dict):
        for key in param_dict:
            setattr(self, key, param_dict[key])

def csvname(n,c='G',pol='X'):
    """Formats a file name to get the gain 'G' or phase 'P' of the copol 'X'
    component or the crosspol 'Y' component at the frequency n (MHz)

    n: An integer specifying the frequency in MHz.
    """

    dpath = '/data4/paper/zmart/HERA-Team/hera-cst/GP4Y2H_4900/'
    fbase = c + pol + '4Y2H_4900_'
    return dpath + fbase + str(n) + '.csv'

def ecomp(gfile,pfile):
    """Returns a healpix map of the complex valued co- or cross-pol component
    at a frequency specified in the file name.
    gfile: full path of the file containing the amplitude data.
    pfile: full path of the file containing the phase data.

    Based on github.com/HERA-Team/hera-cst/scripts/cst2hpx_C.py by (I think?) Aaron Parsons.
    """
    pwd = os.getcwd()
    # inpath = os.path.join(os.path.split(pwd)[0],'GP_paper/')
    # outpath= os.path.join(os.path.split(pwd)[0],'HP_paper/')
    inpath = os.path.join(pwd,'GP/')
    outpath = os.path.join(pwd,'hpx_HERA_Aug2016/')

    re_gain = re.compile(r"dB\(Gain[X,Y]\) \[\] - Freq='([\d.]+)GHz' Phi='([\d.]+)deg'")
    re_phas = re.compile(r"ang_deg\(rE[X,Y]\) \[deg\] - Freq='([\d.]+)GHz' Phi='([\d.]+)deg'")

    def __splitMagPhaseFilesFromString(s):
        """Splits file mag/phase pairs from string as mag0:phase0,mag1:phase1,..."""
        magfile = []
        phafile = []
        data = s.split(',')
        for d in data:
            f = d.split(':')
            magfile.append(f[0])
            phafile.append(f[1])
        return magfile,phafile
    def __procfile(s):
        """So you can use mag:phase pairs from a file"""
        fp = open(s,'r')
        magfile = []
        phafile = []
        for line in fp:
            m,p=__splitMagPhaseFilesFromString(line.strip())
            magfile.append(m[0])
            phafile.append(p[0])
        fp.close()
        return magfile,phafile

    with open(gfile) as mcsvfile:
        with open(pfile) as pcsvfile:
            pcsvread = csv.reader(pcsvfile)
            mcsvread = csv.reader(mcsvfile)
            header_m = mcsvread.next()
            header_p = pcsvread.next()
            fqs,phi    =np.array([map(float,re_gain.match(h).groups()) for h in header_m[1:]]).T
            ###Do phase just to make sure they agree
            fqs_p,phi_p=np.array([map(float,re_phas.match(h).groups()) for h in header_p[1:]]).T
            phi.shape = (1,-1)

    dm = np.loadtxt(gfile,delimiter=',',skiprows=1)
    th,dBi = dm[:,:1], dm[:,1:]
    th,phi = th * np.ones_like(phi) * ap.const.deg, phi * np.ones_like(th) * ap.const.deg
    pm = np.loadtxt(pfile,delimiter=',',skiprows=1)
    prad = pm[:,1:]*ap.const.deg
    ggg = 10**(dBi/20.)
    g = ggg*np.cos(prad) + 1j*ggg*np.sin(prad)

    h = ap.map.Map(nside=32,dtype=np.complex128)
    th_f,phi_f,g_f = th.flatten(), phi.flatten(), g.flatten()
    th_f,phi_f = np.abs(th_f), np.where(th_f < 0, phi_f + np.pi, phi_f)
    h.add((th_f,phi_f), np.ones_like(g_f), g_f)
    h.reset_wgt()
    h = h.map

    g_out = h.get_map()

## Another way to do it using a spline fit to the rectangular (theta,phi) grid.
#     g1 = g[180:,:-1]
#     g2 = g[:181,1:]
#     g2 = g2[::-1,:]

#     E_c = np.concatenate((g1,g2), axis=1)

#     Ere = E_c.real
#     Eim = E_c.imag

#     th_use = np.radians(np.linspace(0., 180.,181,endpoint=True))
#     phi_use = np.radians(np.linspace(0., 359., 360., endpoint=True))

#     gre_interpolant = interpolate.interp2d(phi_use,th_use,Ere,kind='cubic',fill_value=0)
#     gim_interpolant = interpolate.interp2d(phi_use,th_use,Eim,kind='cubic',fill_value=0)

#     npix = hp.nside2npix(nside_out)
#     hpxidx = np.arange(npix)
#     th,phi = hp.pix2ang(nside_out,hpxidx)

#     gre_hpx = np.zeros(npix)
#     gim_hpx = np.zeros(npix)
#     for pix in hpxidx:
#         gre_hpx[pix] = gre_interpolant(phi[pix],th[pix])
#         gim_hpx[pix] = gim_interpolant(phi[pix],th[pix])

#     g_out = gre_hpx + 1j*gim_hpx

    return g_out

def udgrade(x,nside_out):
    """
    Spatial interpolation of a healpix map x using spherical harmonic decomposition
    and then synthesis at resolution nside_out.
    """
    return hp.alm2map(hp.map2alm(x),nside_out, verbose=False)

def udgrade_jones(jones, nside_out):
    jones2 = np.zeros((hp.nside2npix(nside_out),2,2), dtype=np.complex128)

    parts = [np.real,np.imag]
    comp = [1., 1.j]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                z = udgrade(parts[k](jones[:,i,j]),nside_out)
                jones2[:,i,j] += z*comp[k]
    return jones2

def arm(hmap):
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

def make_jones(freq):
    if freq in range(90,221):
        pass
    else:
        raise ValueError('Frequency is not available.')

    nside = 512 # make this as large as possible to minimize singularity effects at zenith
    ## nside 512 seems to work well enough using the "neighbours of neighbours" patch
    ## of the zenith singularity in the ra/cza basis.
    npix = hp.nside2npix(nside)
    hpxidx = np.arange(npix)
    t,p = hp.pix2ang(nside,hpxidx)

    g1 = ecomp(csvname(freq,'G','X'),csvname(freq,'P','X'))
    g2 = ecomp(csvname(freq,'G','Y'),csvname(freq,'P','Y'))

    I = (abs(g1)**2. + abs(g2)**2.)

    norm = np.sqrt(np.amax(I, axis=0))

    g1 /= norm
    g2 /= norm

    rhm = irf.rotate_healpix_map

    Rb = np.array([
    [0,0,-1],
    [0,-1,0],
    [-1,0,0]
    ])

    Et_b = udgrade(g1.real, nside) + 1j * udgrade(g1.imag, nside)
    Ep_b = udgrade(g2.real, nside) + 1j * udgrade(g2.imag, nside)

    tb,pb = irf.rotate_sphr_coords(Rb, t, p)

    tb_v = irf.t_hat_cart(tb,pb)
    pb_v = irf.p_hat_cart(tb,pb)

    t_v = irf.t_hat_cart(t,p)
    p_v = irf.p_hat_cart(t,p)

    Rb_tb_v = np.einsum('ab...,b...->a...', Rb, tb_v)
    Rb_pb_v = np.einsum('ab...,b...->a...', Rb, pb_v)

    cosX = np.einsum('a...,a...', Rb_tb_v,t_v)
    sinX = np.einsum('a...,a...', Rb_pb_v,t_v)

    Et = Et_b * cosX + Ep_b * sinX
    Ep = -Et_b * sinX + Ep_b * cosX

    Ext = Et
    Exp = Ep
    ## This assumes that Et and Ep are the components of a dipole oriented
    ## along the X axis, and we want to obtain the components of the same
    ## dipole if it was oriented along Y.
    ## In the current basis, this is done by a scalar rotation of the theta and phi
    ## components by 90 degrees about the Z axis.
    Eyt = arm(Et.real) + 1j*arm(Et.imag)
    Eyp = arm(Ep.real) + 1j*arm(Ep.imag)

    jones_c = np.array([[Ext,Exp],[Eyt,Eyp]]).transpose(2,0,1)

    # jones_a = np.array([[rhm(Ext,Rb), rhm(Exp,Rb)],[rhm(Eyt,Rb),rhm(Eyp,Rb)]]).transpose(2,0,1)
    #
    # basis_rot = np.array([[cosX,-sinX],[sinX,cosX]]).transpose(2,0,1)
    #
    # jones_b = np.einsum('...ab,...bc->...ac', jones_a, basis_rot)
    #
    # Ext_b, Exp_b, Eyt_b, Eyp_b = [rhm(jones_b[:,i,j],Rb) for i in range(2) for j in range(2)]
    #
    # jones_c = np.array([[Ext_b,Exp_b],[Eyt_b,Eyp_b]]).transpose(2,0,1)

    return jones_c

def transform_basis(nside, jones, z0_cza, R_z0):

    npix = hp.nside2npix(nside)
    hpxidx = np.arange(npix)
    cza, ra = hp.pix2ang(nside, hpxidx)

    fR = R_z0

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

    # return np.einsum('...ab,...bc->...ac', jones, basis_rot)
    return irnf.M(jones, basis_rot)

def neighbors_of_neighbors(nside, th, phi):
    """
    Finds the pixel numbers of the 8 neighbors of the the point (th,phi),
    then find the 8 neighbors of each of those points. The are the 64 pixel
    indices of the "neighbors of neighbors" of the point (th,phi).
    """

    neighbors = hp.get_all_neighbours(nside, th, phi=phi)
    tn, pn = hp.pix2ang(nside, neighbors)

    nn = hp.get_all_neighbours(nside, tn, phi=pn)
    return nn.flatten()

def jones2celestial_basis(jones, z0_cza=None):
    if z0_cza is None:
        z0_cza = np.radians(120.7215)

    npix = jones.shape[0]
    nside = hp.npix2nside(npix)

    hpxidx = np.arange(npix)
    cza, ra = hp.pix2ang(nside, hpxidx)

    z0 = irf.r_hat_cart(z0_cza, 0.)

    RotAxis = np.cross(z0, np.array([0,0,1.]))
    RotAxis /= np.sqrt(np.dot(RotAxis,RotAxis))
    RotAngle = np.arccos(np.dot(z0, [0,0,1.]))

    R_z0 = irf.rotation_matrix(RotAxis, RotAngle)

    R_jones = irf.rotate_jones(jones, R_z0, multiway=True) # beams are now pointed at -31 deg latitude

    jones_out = np.zeros((npix, 2,2), dtype=np.complex128)

########
## This next bit is a routine to patch the topological hole by grabbing pixel
## data from a neighborhood of the corrupted pixels.
## It uses the crucial assumption that in the ra/cza basis the dipoles
## are orthogonal at zenith. This means that for the diagonal components,
## the zenith pixels should be a local maximum, while for the off-diagonal
## components the zenith pixels should be a local minimum (in absolute value).
## Using this assumption, we can cover the corrupted pixel(s) in the
## zenith neighborhood by the maximum pixel of the neighborhood
## for the diagonal, and the minimum of the neighborhood for the off-diagonal.
## As long as the function is relatively flat in this neighborhood, this should
## be a good fix

    jones_b = transform_basis(nside, R_jones, z0_cza, R_z0)

    cf = [np.real,np.imag]
    u = [1.,1.j]


    z0pix = hp.vec2pix(nside, z0[0],z0[1],z0[2])
    if nside < 128:
        z0_nhbrs = hp.get_all_neighbours(nside, z0_cza, phi=0.)
    else:
        z0_nhbrs = neighbors_of_neighbors(nside, z0_cza, phi=0.)

    jones_c = np.zeros((npix,2,2,2), dtype=np.float64)
    for k in range(2):
        jones_c[:,:,:,k] = cf[k](jones_b)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                z0_nbhd = jones_c[z0_nhbrs,i,j,k]

                if i == j:
                    fill_val_pix = np.argmax(abs(z0_nbhd))
                    fill_val = z0_nbhd[fill_val_pix]

                else:
                    fill_val_pix = np.argmin(abs(z0_nbhd))
                    fill_val = z0_nbhd[fill_val_pix]

                jones_c[z0_nhbrs,i,j,k] = fill_val
                jones_c[z0pix,i,j,k] = fill_val

    jones_out = jones_c[:,:,:,0] + 1j*jones_c[:,:,:,1]

    return jones_out

## Duplicate?
# def udgrade_jones(jones, nside_out):
#     jones2 = np.zeros((hp.nside2npix(nside_out),2,2), dtype=np.complex128)
#
#     parts = [np.real,np.imag]
#     comp = [1., 1.j]
#     for i in range(2):
#         for j in range(2):
#             for k in range(2):
#                 z = udgrade(parts[k](jones[:,i,j]),nside_out)
#                 jones2[:,i,j] += z*comp[k]
#     return jones2

def jones_f(nu_node, nside):
    return udgrade_jones(jones2celestial_basis(make_jones(nu_node)), nside)

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

def make_ijones_spectrum(parameters_dict, verbose=False):

    p = Parameters(parameters_dict)
    
    """
    nu_axis: frequency in Hz
    """
    fmax = int(p.nu_axis[-1]/1e6)
    fmin = int(p.nu_axis[0]/1e6)
    nfreq = len(p.nu_axis)

    nnodes = fmax - fmin + 1

    nu_nodes = np.array([fmin + x for x in range(nnodes)])

    lmax = 3 * p.nside -1
    nlm = hp.Alm.getsize(lmax)
    joneslm = np.zeros((nnodes, nlm, 2,2,2), dtype=np.complex128)

    sht = lambda x: hp.map2alm(x, lmax=lmax)

    comp = [np.real, np.imag]
    u = [1,1j]

    if verbose == True:
        print "Freq. min/max:", fmin, fmax
        print "nnodes: ", nnodes
        print "len(nu_nodes): ", len(nu_nodes)

    # synthesize maps at the nside to be used in the simulation for each frequency node
    # This is necessary because the basis transformation is done at nside 1024 to minimize
    # the topological error at the center of the beam. But for the frequency interpolation
    # this resolution would probably use too much memory.
    # note that the output of jones_f() is a beam with zenith at -31 deg latitude
    for n in range(nnodes):
        if verbose == True:
            print "Loading jones node ", n
        jones_node = jones_f(nu_nodes[n], p.nside)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    joneslm[n,:,i,j,k] = sht(comp[k](jones_node[:,i,j]))

    joneslm_re = joneslm.real
    joneslm_im = joneslm.imag

    if verbose == True:
        print joneslm_re.shape
        print joneslm_im.shape
        print nu_nodes.shape

    interpolant_re = interpolate.interp1d(nu_nodes,joneslm_re,kind='cubic',axis=0)
    interpolant_im = interpolate.interp1d(nu_nodes,joneslm_im,kind='cubic',axis=0)

    freqs_out = p.nu_axis/1e6

    joneslm_re_int = interpolant_re(freqs_out)
    joneslm_im_int = interpolant_im(freqs_out)

    joneslm_int = joneslm_re_int + 1j*joneslm_im_int

    # now we just need to resynthesize at each frequency and we're done
    isht = lambda x: hp.alm2map(np.ascontiguousarray(x), p.nside,verbose=False)

    z0_cza = np.radians(120.7215)

    ijones = np.zeros((p.nfreq,p.npix,2,2), dtype=np.complex128)
    for n in range(p.nfreq):
        for i in range(2):
            for j in range(2):
                ijones[n,:,i,j] = isht(joneslm_int[n,:,i,j,0]) + 1j*isht(joneslm_int[n,:,i,j,1])

        If = abs(ijones[n,:,0,0])**2. + abs(ijones[n,:,0,1])**2. + abs(ijones[n,:,0,1])**2. + abs(ijones[n,:,1,0])**2
        norm = np.sqrt(np.amax(If))
        ijones[n] /= norm

        ijones[n] *= horizon_mask(ijones[n].squeeze(), z0_cza)

        if verbose == True:
            print "norm is:", norm

    return ijones
