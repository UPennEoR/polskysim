import numpy as np
import math
import healpy as hp

from radiono import std_hour
from radiono import physics as phys, interp as itp, ionex_file as inx

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

def rotate_healpix_mindex(m, R):
    npix = len(m)
    nside = hp.npix2nside(npix)
    hpxidx = np.arange(npix)
    c, a = hp.pix2ang(nside, hpxidx)

    t, p = rotate_sphr_coords(R, c, a)
    rotidx = hp.ang2pix(nside, t, p)

    m_R = m[rotidx]

    return m_R

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

def local_rot(z0_cza):
    """
    Convinience function for the frequently used rotation matrix from cza/ra to
    local za/az coordinate frame.

    z0_cza should have units of radians.
    """

    z0 = r_hat_cart(z0_cza, 0.)

    RotAxis = np.cross(z0, np.array([0,0,1.]))
    RotAxis /= np.sqrt(np.dot(RotAxis,RotAxis))
    RotAngle = np.arccos(np.dot(z0, [0,0,1.]))

    R_z0 = rotation_matrix(RotAxis, RotAngle)

    return R_z0

def transform_baselines(baselines_list):
    """
    Transforms
    """
    # Compute coordinate rotation matrix
    z0_cza = np.radians(120.7215) # Hardcoded for HERA/PAPER latitude

    R_z0 = local_rot(z0_cza)

    # Rb = np.array([
    # [0,0,-1],
    # [0,-1,0],
    # [-1,0,0]
    # ])

    # fR = np.einsum('ab,bc->ac', Rb, R_z0) # matrix product of two rotations

    b = np.array(baselines_list)
    bl_eq = np.einsum('...ab,...b->...a', R_z0.T, b) # this give the right fringes. See fringe_rotate.ipynb

    return bl_eq

def ion_RM(B_para, TEC_path):
    IFR = 2.6e-17 * B_para * TEC_path
    return IFR

def _test_ionosphere_map(date_str='2004-05-19T00:00:00'):
    # date_str = '2004-05-19T00:00:00'
    lat_str = '30d43m17.5ss'
    lon_str = '21d25m41.9se'



    year, month, day = date_str.split('T')[0].split('-')

    tec_hp, rms_hp, ion_height = inx.IONEX_data(year, month, day, verbose=False)

    nside_in = 2**4
    npix_in = hp.nside2npix(nside_in)
    hpxidx = np.arange(npix_in)
    za, az = hp.pix2ang(nside_in, hpxidx)

    lat, lon, az_p, za_p = phys.ipp(lat_str, lon_str,
                                np.degrees(az), np.degrees(za),
                                ion_height)

    B_para = phys.B_IGRF(year, month, day,
                    lat, lon,
                    ion_height,
                    az_p, za_p)

    TEC_path = np.zeros((24,npix))
    for t in range(0,24):
        hour = rad.std_hour(t, verbose=False)

        TEC_path[t], _ = itp.interp_space(tec_hp[t], rms_hp[t],
                                                 lat, lon,
                                                 za_p)

    RM_maps = ion_RM(B_para, TEC_path)

    for t in range(24):
        RM_maps[t] = rotate_healpix_map(RM_maps[t], R_z0.T)

def std_day_str(n):
    if n <= 10:
        day_str = '0' + str(n)
    else:
        day_str = str(n)

def date_range(day0, ndays):
    """
    Gets the dates at which to compute ionosphere maps over a range of days.
    """

    year='2010'
    month='06'
    day0=str(day0)

    # date_strs = ["-".join((year,month,std_day_str(n) for n in range(ndays) ]
