import numpy as np, healpy as hp
import os

from scipy import interpolate
import ionRIME_funcs as irf
import numba_funcs as irnf

def txtname(n):
    # dpath = '/data4/paper/zionos/polskysim/IonRIME/InstrumentSimData/NicCST_Old/'
    # fname = 'Directivity ' + str(n) + ' MHz.txt'

    dpath = '/data4/paper/zionos/polskysim/IonRIME/InstrumentSimData/NicCST/'
    fname = 'HERA - E-pattern - ' + str(n) + 'MHz.txt'
    return dpath + fname

def linear2Dbi(E): return  2. * 10. * np.log10(E)

def Dbi2linear(Dbi): return np.power(10., Dbi/20.)

def AzimuthalRotation(hmap):
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

def udgrade(x,nside_out):
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

def make_jones(freq):

    # if freq not in range(50,250):
    #     raise Exception('The input must be an integer in the range [50,250]')

    data1 = np.loadtxt(txtname(freq),skiprows=2)

    th_data = np.radians(data1[:,0])
    phi_data = np.radians(data1[:,1])

    Et = data1[:,3] * np.exp(-1j * np.radians(data1[:,4]))
    Ep = data1[:,5] * np.exp(-1j * np.radians(data1[:,6]))

    cosp = np.cos(phi_data)
    sinp = np.sin(phi_data)

    rEt = cosp * Et - sinp * Ep
    rEp = sinp * Et + cosp * Ep

    th_f,phi_f = np.abs(th_data), np.where(th_data < 0, phi_data + np.pi, phi_data)
    # th_f, phi_f = np.abs(thM), np.where(thM < 0, phiM + np.pi, phiM)

    hpxiz = lambda m: irf.healpixellize(m,th_f,phi_f,32,fancy=False)

    nside = 512
    npix = hp.nside2npix(nside)
    hpxidx = np.arange(npix)
    th,phi = hp.pix2ang(nside, hpxidx)
    phi = np.where(phi >= np.pi, phi - np.amax(phi), phi)

    EXt, EXp = [udgrade(hpxiz(X.real),nside) + 1j * udgrade(hpxiz(X.imag),nside) for X in [rEt,rEp]]

    cosP = np.cos(phi)
    sinP = np.sin(phi)

    nEXt = cosP * EXt + sinP * EXp
    nEXp = -sinP * EXt + cosP * EXp

    nEYt = AzimuthalRotation(nEXt)
    nEYp = AzimuthalRotation(nEXp)

    jones_out = np.array([[nEXt,nEXp],[nEYt,nEYp]]).transpose(2,0,1)
    joens_out = np.ascontiguousarray(jones_out)

    return jones_out

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

def make_ijones_spectrum(p, verbose=False):
    """
    nu_axis: frequency in Hz
    """
    fmax = int(p.nu_axis[-1]/1e6)
    fmin = int(p.nu_axis[0]/1e6)

    # fmax = 250
    # fmin = 80
    nfreq = len(p.nu_axis)

    nnodes = fmax - fmin + 1

    nu_nodes = np.array([fmin + x for x in range(nnodes)])
    # nu_nodes = np.array(range(80,260,10))
    # nnodes = len(nu_nodes)

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
            print "Loading jones node ", n, ", freq", nu_nodes[n]
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
