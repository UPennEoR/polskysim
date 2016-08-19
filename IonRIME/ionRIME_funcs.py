import numpy as np
import math
import healpy as hp
from datetime import datetime

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

def healpixellize(f_in,theta_in,phi_in,nside,fancy=True):
    """ A dumb method for converting data f sampled at points theta and phi (not on a healpix grid) into a healpix at resolution nside """

    # Input arrays are likely to be rectangular, but this is inconvenient
    f = f_in.flatten()
    theta = theta_in.flatten()
    phi = phi_in.flatten()

    pix = hp.ang2pix(nside,theta,phi)

    map = np.zeros(hp.nside2npix(nside))
    hits = np.zeros(hp.nside2npix(nside))

    # Simplest gridding is map[pix] = val. This tries to do some
    #averaging Better would be to do some weighting by distance from
    #pixel center or something ...
    if (fancy):
        for i,v in enumerate(f):
            # Find the nearest pixels to the pixel in question
            neighbours,weights = hp.get_interp_weights(nside,theta[i],phi[i])
            # Add weighted values to map
            map[neighbours] += v*weights
            # Keep track of weights
            hits[neighbours] += weights
        map = map/hits
        wh_no_hits = np.where(hits == 0)
        print 'pixels with no hits',wh_no_hits[0].shape
#         map[wh_no_hits[0]] = hp.UNSEEN
        map[wh_no_hits[0]] = 0 # effectively a horizon mask in this case,
        # since there is no FEKO data below the local horizaon.
    else:
        for i,v in enumerate(f):
            map[pix[i]] += v
            hits[pix[i]] +=1
        map = map/hits
    print 'Healpixellization successful.'
    return map

def get_time_string(d, day0):
    date0 = datetime(*day0)
    one_day = datetime(1,1,2) - datetime(1,1,1)

    # the trailing time string was removed from radionopy so prob. don't need this anymore
    # date = str(date0 + d * one_day).split(' ')[0]
    # time_str = date + 'T00:00:00'

    time_str = str(date0 + d * one_day).split(' ')[0]
    return time_str

def PAPER_instrument_setup(z0_cza):
    # hack hack hack
    import sys
    sys.path.append('PAPER_beams/') # make this whatever it needs to be so that fmt can be imported

    import fmt
    nu0 = str(int(p.nu_axis[0] / 1e6))
    nuf = str(int(p.nu_axis[-1] / 1e6))
    band_str = nu0 + "-" + nuf

    local_jones0_file = 'local_jones0/PAPER/nside' + str(p.nside) + '_band' + band_str + '_Jdata.npy'
    if os.path.exists(local_jones0_file) == True:
        return np.load(local_jones0_file)

    fekoX = fmt.FEKO('PAPER_beams/PAPER_FF_X.ffe')
    fekoY = fmt.FEKO('PAPER_beams/PAPER_FF_Y.ffe')

    thetaF = np.radians(fekoX.fields[0].theta)
    phiF = np.radians(fekoX.fields[0].phi)

    nfreq = 11
    npixF = thetaF.shape[0]
    nthetaF = 91 # don't think these are used
    nphiF = 73

    jonesFnodes_ludwig = np.zeros((nfreq,npixF,2,2), dtype='complex128')
    for f in range(nfreq):
        jonesFnodes_ludwig[f,:,0,0] = fekoX.fields[f].etheta
        jonesFnodes_ludwig[f,:,0,1] = fekoX.fields[f].ephi
        jonesFnodes_ludwig[f,:,1,0] = fekoY.fields[f].etheta
        jonesFnodes_ludwig[f,:,1,1] = fekoY.fields[f].ephi

    # getting out of the Ludwig-3 basis. Seriously, wtf?
    # Copied Chuneeta/PolSims/genHealpyBeam.
    R_phi = np.array([[np.cos(phiF), np.sin(phiF)],[-np.sin(phiF), np.cos(phiF)]]).transpose(2,0,1)

    jonesFnodes = np.einsum('...ab,...bc->...ac', jonesFnodes_ludwig, R_phi)

    Rb = np.array([
    [0,0,-1],
    [0,-1,0],
    [-1,0,0]
    ])

    tb, pb = rotate_sphr_coords(Rb, thetaF, phiF)

    tF_v = t_hat_cart(thetaF, phiF)
    pF_v = p_hat_cart(thetaF, phiF)

    tb_v = t_hat_cart(tb, pb)

    fRtF_v = np.einsum('ab...,b...->a...', Rb, tF_v)
    fRpF_v = np.einsum('ab...,b...->a...', Rb, pF_v)

    cosX = np.einsum('a...,a...', fRtF_v, tb_v)
    sinX = np.einsum('a...,a...', fRpF_v, tb_v)

    basis_rot = np.array([[cosX, sinX],[-sinX, cosX]])
    basis_rot = np.transpose(basis_rot,(2,0,1))

    jonesFnodes_b = np.einsum('...ab,...bc->...ac', jonesFnodes, basis_rot)

    nside_F = 2**5
    npix_F = hp.nside2npix(nside_F)

    h = lambda m: healpixellize(m, tb, pb, nside_F)

    jones_hpx_b = np.zeros((nfreq,npix_F,2,2), dtype='complex128')

    for f in range(nfreq):
        for i in range(2):
            for j in range(2):
                Re = h((jonesFnodes_b.real)[f,:,i,j])
                Im = h((jonesFnodes_b.imag)[f,:,i,j])
                jones_hpx_b[f,:,i,j] = Re + 1j*Im

    # note that Rb is an involution, Rb = Rb^-1
    jones = np.zeros_like(jones_hpx_b)
    for i in range(11):
        jones[i] = rotate_jones(jones_hpx_b[i], Rb, multiway=True) # rotate scalar components so instrument is pointed to northpole of healpix coordinate frame

    npix = hp.nside2npix(nside_F)
    hpxidx = np.arange(npix)
    cza, ra = hp.pix2ang(nside_F, hpxidx)

    z0 = r_hat_cart(z0_cza, 0.)

    RotAxis = np.cross(z0, np.array([0,0,1.]))
    RotAxis /= np.sqrt(np.dot(RotAxis,RotAxis))
    RotAngle = np.arccos(np.dot(z0, [0,0,1.]))

    R_z0 = rotation_matrix(RotAxis, RotAngle)

    t0, p0 = rotate_sphr_coords(R_z0, cza, ra)

    hm = np.zeros(npix)
    hm[np.where(cza < (np.pi / 2. + np.pi / 20.))] = 1 # Horizon mask; is 0 below the local horizon.
    # added some padding. Idea being to allow for some interpolation near the horizon. Questionable.
    npix_out = hp.nside2npix(p.nside)

    Jdata = np.zeros((11,npix_out,2,2),dtype='complex128')
    for i in range(11):
        J_f = flatten_jones(jones[i]) # J_f.shape = (npix_in, 8)

        J_f = J_f * np.tile(hm, 8).reshape(8, npix).transpose(1,0) # Apply horizon mask

        # Could future "rotation" of these zeroed-maps have small errors at the
        # edges of the horizon? due to the way healpy interpolates.
        # Unlikely to be important.
        # Comment update: Yep, it turns out this happens, BUT it is approximately
        # power-preserving. The pixels at the edges of the rotated mask are not
        # identically 1, but the sum over the mask is maintained to about a part
        # in 1e-5

        # Perform a scalar rotation of each component so that the instrument's boresight
        # is pointed toward (z0_cza, 0), the location of the instrument on the
        # earth in the Current-Epoch-RA/Dec coordinate frame.
        J_f = rotate_jones(J_f, R_z0, multiway=False)

        if p.nside != nside_F:
            # Change the map resolution as needed.

            #d = lambda m: hp.ud_grade(m, nside=p.nside, power=-2.)
                # I think these two ended up being (roughly) the same?
                # The apparent normalization problem was really becuase of an freq. interpolation problem.
                # irf.harmonic_ud_grade is probably better for increasing resolution, but hp.ud_grade is
                # faster because it's just averaging/tiling instead of doing SHT's
            d = lambda m: harmonic_ud_grade(m, nside_F, p.nside)
            J_f = (np.asarray(map(d, J_f.T))).T
                # The inner transpose is so that correct dimension is map()'ed over,
                # and then the outer transpose returns the array to its original shape.

        J_f = inverse_flatten_jones(J_f) # Change shape to (nfreq,npix,2,2), complex-valued
        J_f = transform_basis(p.nside, J_f, z0_cza, R_z0) # right-multiply by the basis transformation matrix from RA/Dec to the Local CST basis.
        Jdata[i,:,:,:] = J_f

    if os.path.exists(local_jones0_file) == False:
        np.save(local_jones0_file, Jdata)

    return Jdata
