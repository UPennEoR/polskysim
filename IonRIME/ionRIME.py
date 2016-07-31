import numpy as np
import healpy as hp
import os
from scipy import interpolate
import ionRIME_funcs as irf

import astropy.coordinates as coord
import astropy.units as units
import gsm2016
import time
from numba import jit

@jit
def M(m1,m2):
    """
    Computes the matrix multiplication of two arrays of matricies m1 and m2.
    m1.shape = m2.shape = (N,2,2)
    For each n < N, m_out is the product of the 2x2 matricies m1[n,:,:].m2[n,:,:],
    where the first index of the matrix corresponds to a row, and the second
    corresponds to a column.

    Made double-plus-gooder by the @jit decorator from the numba package.
    """
    m_out = np.zeros_like(m1)
    for n in range(len(m1[:,0,0])):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    m_out[n,i,k] += m1[n,i,j] * m2[n,j,k]
    return m_out

@jit
def RIME_integral(C, K, V):
    """
    C.shape = (npix, 2, 2)
    K.shape = (npix,)

    For each component of the 2x2 coherency tensor field C, sum the product
    C(p)_ij * exp(-2 * pi * i * b.s(p) ) to produce a model visibility V(b)_ij.
    """
    npix = np.size(K)
    for i in range(2):
        for j in range(2):
            for pi in range(npix):
                V[i, j] += C[pi,i,j]*K[pi]
    return V / np.float(npix)

def Hz2GHz(freq):
    return freq / 1e9

def get_gsm_cube():
    nside_in = 64
    npix_in = hp.nside2npix(nside_in)
    I_gal = np.zeros((p.nfreq, npix_in))
    for fi, f in enumerate(p.nu_axis):
        I_gal[fi] = gsm2016.get_gsm_map_lowres(Hz2GHz(f))

    x_c = np.array([1.,0,0])
    y_c = np.array([0,1.,0])
    z_c = np.array([0,0,1.])

    # The GSM is given in galactic coordinates. We will rotate it to J2000 equatorial coordinates.
    axes_icrs = coord.SkyCoord(x=x_c, y=y_c, z=z_c, frame='icrs', representation='cartesian')
    axes_gal = axes_icrs.transform_to('galactic')
    axes_gal.representation = 'cartesian'

    R = np.array(axes_gal.cartesian.xyz) # The 3D rotation matrix that defines the coordinate transformation.

    npix_out = hp.nside2npix(p.nside)
    I = np.zeros((p.nfreq, npix_out))

    for i in range(p.nfreq):
        I[i] = irf.harmonic_ud_grade(I_gal[i], nside_in, p.nside)
        I[i] = irf.rotate_healpix_map(I[i], R)

    return I

def transform_basis(nside, jones, z0_cza, R_z0):
    """
    At zenith in the local frame the 'x' feed is aligned with 'theta' and
    the 'y' feed is aligned with 'phi'
    """
    npix = hp.nside2npix(nside)
    hpxidx = np.arange(npix)
    cza, ra = hp.pix2ang(nside, hpxidx)

    # Rb is the rotation relating the E-field basis coordinate frame to the local horizontal zenith.
    # (specific to this instrument response simulation data)
    Rb = np.array([
    [0,0,-1],
    [0,-1,0],
    [-1,0,0]
    ])

    fR = np.einsum('ab,bc->ac', Rb, R_z0) # matrix product of two rotations

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

    return M(jones, basis_rot)

def instrument_setup(z0_cza, freqs, restore=False):
    """
    This is the CST simulation using the efield basis of z' = -x, y' = -y, x' = -z
    frequencies are every 10MHz, from 100-200
    Each file contains 8 columns which are ordered as:
          (Re(xt),Re(xp),Re(yt),Re(yp),Im(xt),Im(xp),Im(yt),Im(yp)).
    Each column is a healpix map with resolution nside = 2**8
    """

    nu0 = str(int(p.nu_axis[0] / 1e6))
    nuf = str(int(p.nu_axis[-1] / 1e6))
    band_str = nu0 + "-" + nuf

    restore_name = p.interp_type + "_" + "band_" + band_str + "mhz_nfreq" + str(p.nfreq)+ "_nside" + str(p.nside) + ".npz"

    if os.path.exists('jones_save/' + restore_name) == True:
        return (np.load('jones_save/' + restore_name))['J_out']

    local_jones0_file = 'local_jones0/nside' + str(p.nside) + '_band' + band_str + '_Jdata.npz'

    if os.path.exists(local_jones0_file) == True:
        return np.load(local_jones0_file)['Jdata']

    fdir_base = os.path.dirname(__file__)
    fbase = os.path.join(fdir_base,'/HERA_jones_data/HERA_Jones_healpix_')

    nside_in = 2**8
    fnames = [fbase + str(int(f / 1e6)) + 'MHz.txt' for f in freqs]

    npix = hp.nside2npix(nside_in)
    hpxidx = np.arange(npix)
    cza, ra = hp.pix2ang(nside_in, hpxidx)

    z0 = irf.r_hat_cart(z0_cza, 0.)

    RotAxis = np.cross(z0, np.array([0,0,1.]))
    RotAxis /= np.sqrt(np.dot(RotAxis,RotAxis))
    RotAngle = np.arccos(np.dot(z0, [0,0,1.]))

    R_z0 = irf.rotation_matrix(RotAxis, RotAngle)

    t0, p0 = irf.rotate_sphr_coords(R_z0, cza, ra)

    hm = np.zeros(npix)
    hm[np.where(cza < (np.pi / 2. + np.pi / 20.))] = 1 # Horizon mask; is 0 below the local horizon.
    # added some padding. Idea being to allow for some interpolation near the horizon. Questionable.
    npix_out = hp.nside2npix(p.nside)

    Jdata = np.zeros((p.nfreq,npix_out,2,2),dtype='complex128')
    for i,f in enumerate(fnames):
        J_f = np.loadtxt(f) # J_f.shape = (npix_in, 8)

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
        J_f = irf.rotate_jones(J_f, R_z0, multiway=False)

        if p.nside != nside_in:
            # Change the map resolution as needed.

            #d = lambda m: hp.ud_grade(m, nside=p.nside, power=-2.)
                # I think these two ended up being (roughly) the same?
                # The apparent normalization problem was really becuase of an freq. interpolation problem.
                # irf.harmonic_ud_grade is probably better for increasing resolution, but hp.ud_grade is
                # faster because it's just averaging/tiling instead of doing SHT's
            d = lambda m: irf.harmonic_ud_grade(m, nside_in, p.nside)
            J_f = (np.asarray(map(d, J_f.T))).T
                # The inner transpose is so that correct dimension is map()'ed over,
                # and then the outer transpose returns the array to its original shape.

        J_f = irf.inverse_flatten_jones(J_f) # Change shape to (nfreq,npix,2,2), complex-valued
        J_f = transform_basis(p.nside, J_f, z0_cza, R_z0) # right-multiply by the basis transformation matrix from RA/Dec to the Local CST basis.
        Jdata[i,:,:,:] = J_f

    # If the model at the current nside hasn't been generated before, save it for future reuse.
    if os.path.exists(local_jones0_file) == False:
        np.savez(local_jones0_file, Jdata=Jdata)

    return Jdata

def _interpolate_jones_freq(J_in, freqs, multiway=True, interp_type='spline'):
    """
    A scheme to interpolate the spherical harmonic components of jones matrix elements.
    Does not seem to work well, and is unused.
    """
    nfreq_in = len(freqs)

    if multiway == True:
        J_flat = np.zeros((nfreq_in, npix, 8), dtype='float64')
        for i in range(nfreq_in):
            J_flat[i] = irf.flatten_jones(J_in[i])
        J_in = J_flat

    lmax = 3 * nside -1
    nlm = hp.Alm.getsize(lmax)
    Jlm_in = np.zeros(nfreq_in, nlm, 8)
    for i in range(nfreq_in):
        sht = lambda m: hp.map2alm(m, lmax=lmax)
        Jlm_in[i,:,:] = (np.asarray(map(sht, J_in.T))).T

    Jlm_out = np.zeros(p.nfreq, nlm, 8)
    for lm in range(nlm):
        for j in range(8):
            Jlmj_re = np.real(Jlm_in[:,lm,j])
            Jlmj_im = np.imag(Jlm_in[:,lm,j])

            a = interpolate_pixel(Jlmj_re, freqs, p.nu_axis, interp_type=p.interp_type)
            b = interpolate_pixel(Jlmj_im, freqs, p.nu_axis, interp_type=p.interp_type)
            Jlm_out[:, lm, j] = a + 1j*b

    # J_in.shape = (p.nfreq_in, ??, 8)

    # Now, return alm's? or spatial maps?

def interpolate_pixel(pi, freqs, interp_type='linear'):
    """
    Interpolates pixel along the frequency axis.
    """
    # looks like 'hermite' interpolation will take ~6 times as long as linear
    if interp_type == 'hermite':
        interpolant = interpolate.PchipInterpolator(freqs, pi)

    elif interp_type == 'spline':
        interpolant = interpolate.InterpolatedUnivariateSpline(freqs, pi)

    elif interp_type == 'fitspline':
        interpolant = interpolate.UnivariateSpline(freqs, pi)

    else:
        interpolant = interpolate.interp1d(freqs, pi) # linear interpolation

    p_out = interpolant(p.nu_axis)
    return p_out

def interpolate_jones_freq(J_in, freqs, multiway=True, interp_type='cubic', save=False):
    #nfreq_out = len(nu_axis)
    nfreq_in = len(freqs)
    npix = len(J_in[0,:,0])
    #nside = hp.npix2nside(npix)

    if multiway == True:
        J_flat = np.zeros((nfreq_in, npix, 8), dtype='float64')
        for i in range(nfreq_in):
            J_flat[i] = irf.flatten_jones(J_in[i])
        J_in = J_flat

    # J_in.shape = (nfreq_in,npix, 8)

    # J_out = np.zeros((p.nfreq,npix,8))
    # for pi in xrange(npix):
    #     for i in xrange(8):
    #         J_out[:,pi,i] = interpolate_pixel(J_in[:, pi, i], freqs, interp_type=interp_type)

    interpolant = interpolate.interp1d(freqs, J_in, kind=interp_type,axis=0)
    J_out = interpolant(p.nu_axis)

    if multiway == True:
        J_m = np.zeros((p.nfreq, npix, 2,2), dtype='complex128')
        for i in range(p.nfreq):
            J_m[i] = irf.inverse_flatten_jones(J_out[i])
        J_out = J_m

    for i in range(p.nfreq):
        Bx_max = np.amax(np.absolute(J_out[i,:,0,0])**2. + np.absolute(J_out[i,:,0,1])**2.)
        By_max = np.amax(np.absolute(J_out[i,:,1,0])**2. + np.absolute(J_out[i,:,1,1])**2.)
        J_out[i,:,0,0] /= np.sqrt(Bx_max)
        J_out[i,:,0,1] /= np.sqrt(Bx_max)
        J_out[i,:,1,0] /= np.sqrt(By_max)
        J_out[i,:,1,1] /= np.sqrt(By_max)

    # Bah, figure it out later
    # Bx_max = np.amax(
    #     np.absolute(J_out[:,:,0,0])**2. + np.absolute(J_out[:,:,0,1])**2.,
    #     axis=1)
    # By_max = np.amax(
    #     np.absolute(J_out[:,:,1,0])**2. + np.absolute(J_out[:,:,1,1])**2.,
    #     axis=1)
    # # Bx_max.shape = By_max = (nfreq,)
    #
    # J_out[:,:,0,:] /= Bx_max[:,None,None]
    #
    # J_out[:,:,1,:] /= By_max[:,None,None]

    if save == True:
        nu0 = str(int(p.nu_axis[0] / 1e6))
        nuf = str(int(p.nu_axis[-1] / 1e6))
        fname = p.interp_type + "_" + "band_" + nu0 + "-" + nuf + "mhz_nfreq" + str(p.nfreq)+ "_nside" + str(p.nside) + ".npz"
        np.savez('jones_save/' + fname, J_out=J_out)
    return J_out

def main(p, restore=False, save=False):

    npix = hp.nside2npix(p.nside)
    hpxidx = np.arange(npix)
    cza, ra = hp.pix2ang(p.nside, hpxidx)

    z0_cza = np.radians(120.7215)
    z0_ra = np.radians(0.)

    ## sky
    """
    sky.shape = (p.nfreq, npix, 2,2)
    """
    #I,Q,U,V = [np.random.rand(p.nfreq,npix) for x in range(4)]

    if True:
        I = get_gsm_cube()
        Q,U,V = [np.zeros((p.nfreq, npix)) for x in range(3)]

    if False:
        I = np.ones((p.nfreq,npix))
        Q,U,V = [np.zeros((p.nfreq, npix)) for x in range(3)]

    if False:
        I = np.zeros((p.nfreq, npix))
        sc, sa = np.radians([122., 123.]), np.radians([345.,255.])
        pidx = hp.ang2pix(p.nside, sc, sa)
        I[:,pidx] = 10. # Jy? meh
        Q,U,V = [np.zeros((p.nfreq, npix)) for x in range(3)]

        I += ( 2. * np.random.random_sample((p.nfreq,npix)) - 1.) * 1e-4
        I += 1e-2

    if False:
        I = np.zeros((p.nfreq, npix))
        sc, sa = np.radians([122.]), np.radians([355.])
        pidx = hp.ang2pix(p.nside, sc, sa)
        I[:,pidx] = 10. # Jy? meh
        Q,U,V = [np.zeros((p.nfreq, npix)) for x in range(3)]

        I += ( 2. * np.random.random_sample((p.nfreq,npix)) - 1.) * 1e-4
        I += 1e-2


    ## Instrument
    """
    Jdata = (nfreq, npix, )
    ijones.shape = (nfreq, npix, 2,2)
    ijones_init.shape = (nfreq_in, npix, 2,2)
    nfreq > nfreq_in
    """
    freqs = [x * 1e6 for x in range(150,171)] # Hz
    # freqs = [(100 + 10 * x) * 1e6 for x in range(11)] # Hz. Must be converted to MHz for file list.
    #freqs = [140, 150, 160]
    tmark0 = time.clock()

    if restore == False:
        Jdata = instrument_setup(z0_cza, freqs, restore=restore)
        #np.savez('var_test/Jdata_test.npz', Jdata=Jdata)

        tmark_inst = time.clock()
        print "Completed instrument_setup(), in " + str(tmark_inst - tmark0)

        ijones = interpolate_jones_freq(Jdata, freqs, interp_type=p.interp_type, save=save)

        tmark_interp = time.clock()
        print "Completed interpolate_jones_freq(), in " + str(tmark_interp - tmark_inst)

    else:
        nu0 = str(int(p.nu_axis[0] / 1e6))
        nuf = str(int(p.nu_axis[-1] / 1e6))
        fname = "band_" + nu0 + "-" + nuf + "mhz_nfreq" + str(p.nfreq)+ "_nside" + str(p.nside) + ".npz"
        ijones = (np.load('jones_save/' + fname))['J_out']
        print "Restored Jones model"

    ijonesH = np.transpose(ijones.conj(),(0,1,3,2))

        # ## Basis rotation
        # """
        # Rb.shape = (npix, 2, 2)
        # """
        # cosX, sinX = sbr.spherical_CSTbasis_rotation_components(nside, z0_cza * d2r, z0_ra * d2r)
        # Rb = np.array([
        #     [cosX, sinX],
        #     [-sinX, cosX]])
        # Rb = np.transpose(Rb,(2,0,1))
        # RbT = np.transpose(Rb,(0,2,1))

    ## Baselines
    bl_eq = irf.transform_baselines(p.baselines) # get baseline vectors in equatorial coordinates

    ## For each (t,f):
    # V[t,f,0,0] == V_xx[t,f]
    # V[t,f,0,1] == V_xy[t,f]
    # V[t,f,1,0] == V_yx[t,f]
    # V[t,f,1,1] == V_yy[t,f]
    Vis = np.zeros(p.nbaseline * p.ntime * p.nfreq * 2 * 2, dtype='complex128')
    Vis = Vis.reshape(p.nbaseline, p.ntime, p.nfreq, 2, 2)

    tmark_loopstart = time.clock()

    if debug == True:
        source_index = np.zeros(p.ntime)
        beam_track = np.zeros(npix)

    for b_i in range(bl_eq.shape[0]):
        for t in range(p.ntime):
            zl_cza = z0_cza
            total_angle = 10. # degrees
            zl_ra = (float(t) / float(p.ntime)) * np.radians(total_angle)

            RotAxis = np.array([0.,0.,1.])
            RotAngle = -zl_ra
            R_t = irf.rotation_matrix(RotAxis, RotAngle)

            # It = I
            # Qt = Q
            # Ut = U
            # Vt = V

            It = np.zeros_like(I)
            Qt = np.zeros_like(Q)
            Ut = np.zeros_like(U)
            Vt = np.zeros_like(V)

            for i in range(p.nfreq):
                It[i] = irf.rotate_healpix_mindex(I[i], R_t)
                Qt[i] = irf.rotate_healpix_mindex(Q[i], R_t)
                Ut[i] = irf.rotate_healpix_mindex(U[i], R_t)
                Vt[i] = irf.rotate_healpix_mindex(V[i], R_t)

            sky_t = np.array([
                [It + Qt, Ut - 1j*Vt],
                [Ut + 1j*Vt, It - Qt]]).transpose(2,3,0,1)
                # Could do this iteratively! Define the differential rotation
                # and apply it in-place to the same sky tensor at each step of the time loop.
            #ijones_t = irf.rotate_jones(ijones, R_t, multiway=True)

            if debug == True:
                source_index[t] = np.argmax(It[int(p.nfreq / 2),:])
                Bt = irf.rotate_healpix_map(abs(ijones[int(p.nfreq/2),:,0,0])**2. + abs(ijones[int(p.nfreq/2),:,0,1])**2., R_t.T)
                beam_track += Bt / float(p.ntime)


            for nu_i, nu in enumerate(p.nu_axis):
                #print "t is " + str(t) + ", nu_i is " + str(nu_i)

                ## Ionosphere
                """
                ionrot.shape = (p.nfreq,npix 2,2)
                """

                # RMangle = 2. * np.pi * np.random.rand(p.nfreq,npix)
                # ion_cos = np.cos(RMangle)
                # ion_sin = np.sin(RMangle)
                ion_cos = np.ones((p.nfreq, npix))
                ion_sin = np.zeros((p.nfreq, npix))
                ion_rot = np.array([[ion_cos, ion_sin],[-ion_sin,ion_cos]])
                ion_rot = np.transpose(ion_rot,(2,3,0,1))
                ion_rotT = np.transpose(ion_rot,(0,1,3,2))
                # worried abou this...is the last line producing the right ordering,
                # or is ion_rot unchanged

                ## Fringe
                """K.shape = (npix)"""

                c = 299792458. # meters / sec
                b = bl_eq[b_i]# meters, in the Equatorial basis
                s = hp.pix2vec(p.nside, hpxidx)
                b_dot_s = np.einsum('a...,a...',b,s)
                tau = b_dot_s / c
                K = np.exp(-2. * np.pi * 1j * tau * nu)


                # Oleg, eat your heart out
                compose_4M = lambda a,b,c,d,e: M(M(M(M(a,b),c),d),e)

                C = compose_4M(
                    ijones[nu_i],
                    ion_rot[nu_i],
                    sky_t[nu_i],
                    ion_rotT[nu_i],
                    ijonesH[nu_i])

                # could also be:
                # reduce(M, [ijones[nu_i],ion_rot[nu_i]])

                Vis[b_i,t,nu_i,:,:] = RIME_integral(C, K, Vis[b_i,t,nu_i,:,:].squeeze())

    tmark_loopstop = time.clock()
    print "Visibility loop completed in " + str(tmark_loopstop - tmark_loopstart)
    print "Full run in " + str(tmark_loopstop -tmark0) + " seconds."

    out_name = "Vis_" + p.interp_type + "_band_" + str(int(p.nu_0 / 1e6)) + "-" + str(int(p.nu_f /1e6)) + "MHz_nfreq" + str(p.nfreq)+ "_ntime" + str(p.ntime) + "_nside" + str(p.nside) + ".npz"

    #if os.path.exists(out_name) == False:
    np.savez('output_vis/' + out_name, Vis=Vis)
    if debug == True:
        np.savez('debug/source_index.npz', source_index=source_index)
        np.savez('debug/beam_track.npz', beam_track=beam_track)


class Parameters:
    pass

if __name__ == '__main__':
    print "Note! Horizon mask is off!"
    print "Note! Ionosphere set to Identity!"
    #print "Note: Horizon mask turned off!"
    #print "Note! Sky rotation turned off"
    print "Note! time rotation angle is not 360deg"

    #########
    # Dimensions and Boundaries

    p = Parameters()

    p.nside = 2**6 # sets the spatial resolution of the simulation, for a given baseline

    p.nfreq = 41 # the number of frequency channels at which visibilities will be computed.

    p.ntime = 10  # the number of time samples in one rotation of the earch that will be computed

    p.ndays = 1 # The number of days that will be simulated.

    p.nu_0 = 1.5e8 # Hz. The high end of the simulated frequency band.

    p.nu_f = 1.7e8 # Hz. The low end of the simulated frequency band.

    p.nu_axis = np.linspace(p.nu_0,p.nu_f,num=p.nfreq,endpoint=True)

    p.baselines = [[15.,0,0],[15.,0,0]]

    p.nbaseline = len(p.baselines)

    p.interp_type = 'cubic'
    # options for interpolation are:
    # 'linear' and 'cubic', both via scipy.interpolate.interp1d()

    ## OLD OPTIONS
    #   'linear' : linear interpolation between nodes
    #   'hermite': Piecewise Cubic Hermite Interpolating Polynomials between each
    #       pair of nodes. This produces a monotonic interpolant between each pair
    #       of nodes, but the derivative is not continuous at the nodes i.e there
    #       are corners in the interpolant. This one takes the longest to compute, ~6.5x 'linear'.
    #   'fitspline': cubic spline fit to the nodes. Does NOT interpolate the nodes.
    #   'spline': interpolating cubic spline

    global debug
    debug = True

    main(p, restore=False,save=True)
    print "Compiled successfully"
