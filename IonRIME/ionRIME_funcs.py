import numpy as np
import math
import healpy as hp

def rotate_sphr_coords(R, theta, phi):
    """
    Returns the spherical coordinates of the point specified by vp = R . v,
    where v is the 3D position vector of the point specified by (theta,phi) and
    R is the 3D rotation matrix that relates two coordinate charts.
    """
    rhx = np.cos(phi) * np.sin(theta)
    rhy = np.sin(phi) * np.sin(theta)
    rhz = np.cos(theta)
    r = np.stack((rhx,rhy,rhz))
    rP = np.einsum('ab...,b...->a...',R ,r)
    thetaP = np.arccos(rP[-1,:])
    phiP = np.arctan2(rP[1,:],rP[0,:])
    phiP[phiP < 0] += 2. * np.pi
    return (thetaP,phiP)

def t_hat_cart(t,p):
    """ Calculate the theta_hat vector at a given point (t,p) """
    thx = np.cos(t)*np.cos(p)
    thy = np.cos(t)*np.sin(p)
    thz = -np.sin(t)
    return np.stack((thx,thy,thz))

def p_hat_cart(t,p):
    """ Calculate the phi_hat vector at a given point (t,p) """
    phx = -np.sin(p)
    phy = np.cos(p)
    phz = np.zeros_like(p)
    return np.stack((phx,phy,phz))

def r_hat_cart(t,p):
    """ Calculate the r_hat vector at the given point (t,p)"""
    rhx = np.cos(p) * np.sin(t)
    rhy = np.sin(p) * np.sin(t)
    rhz = np.cos(t)
    return np.stack((rhx,rhy,rhz))

def rotation_matrix(axis, theta):

    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    # Taken from the internet:
    # http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector

    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def flatten_jones(j):
    """
    Returns an (npix,8)-shaped real-valued array from the input (npix,2,2)-shaped
    complex-valued array.
    """
    npix = len(j[:,0,0])
    j = j.reshape(npix,4)
    j_re = np.real(j)
    j_im = np.imag(j)

    j_flat = np.concatenate((j_re, j_im), axis=1)
    return j_flat

def inverse_flatten_jones(j_flat):
    """
    Inverts flatten_jones() to return an (npix,2,2)-shaped complex-valued array.
    """
    npix = len(j_flat[:,0])
    j_re = (j_flat[:,0:4]).reshape(npix,2,2)
    j_im = (j_flat[:,4:8]).reshape(npix,2,2)
    j = j_re + 1j * j_im

    return j

def rotate_jones(j, rotmat, multiway=True):
    """
    Rotates the scalar components of a complex-valued 2x2 matrix field, relative
    to the Healpix coordinate frame.
    """
    if multiway == True:
        j = flatten_jones(j)

    npix = len(j[:,0])
    nside = hp.npix2nside(npix)
    hpxidx = np.arange(npix)
    c, a = hp.pix2ang(nside, hpxidx)

    t, p = rotate_sphr_coords(rotmat, c, a)

    intp = lambda m: hp.get_interp_val(m, t, p)

    # This is the fastest by ~2%. 99.1ms vs 101ms for the loop. (at nside=2**6)
    # ...totally worth it!
    jones = (np.asarray(map(intp,j.T))).T

    # 101ms
    # jR = np.empty_like(j)
    # for i in range(8):
    #     jR[:,i] = intp(j[:,i])

    # 102ms SO SLOW WTF NUMPY?
    #jR = np.apply_along_axis(intp, 0, j)
    # ahem. apparently this is just synactic sugar

    if multiway == True:
        jones = inverse_flatten_jones(jones)

    return jones

def harmonic_ud_grade(m, nside_in, nside_out):
    """
    Decompose a map at a resolution nside_in into spherical harmonic components
    and then resynthesize the map at nside_out.
    """
    lmax = 3 * nside_in - 1
    alm = hp.map2alm(m, lmax=lmax)
    return hp.alm2map(alm, nside_out, lmax=lmax, verbose=False)

def rotate_healpix_map(m, R):
    """
    Performs a scalar rotation of the map relative to the Healpix coordinate
    frame by interpolating the map at the coordinates of new coordinate frame.
    """
    npix = len(m)
    nside = hp.npix2nside(npix)
    hpxidx = np.arange(npix)
    c, a = hp.pix2ang(nside, hpxidx)
    t, p = rotate_sphr_coords(R, c, a)
    return hp.get_interp_val(m, t, p)

def rotate_healpix_mapHPX(m, rot):
    """
    This one uses the Healpix Rotator object, thus the rotation must be
    specified by an Euler angle sequence.

    Otherwise, same thing as rotate_healpix_map.
    """
    npix = len(m)
    nside = hp.npix2nside(npix)
    hpxidx = np.arange(npix)
    c, a = hp.pix2ang(nside, hpxidx)
    R = hp.Rotator(rot=rot)
    t, p = R(c,a)
    return hp.get_interp_val(m, t, p)

def compose_healpix_map_rotations(m,RL):
    """
    Rotates a healpix map by the sequence of rotation matrices in the list RL.
    Note that this means each rotation is performed relative to the Healpix index
    coordinate frame, so this sequence does NOT correspond to a sequence of Euler angle rotations.
    """
    if not RL:
        return m
    arg = [m]
    arg.extend(RL)
    return reduce(lambda m, R: rotate_healpix_map(m, R), arg)
