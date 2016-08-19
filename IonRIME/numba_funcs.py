from numba import jit
from numba import guvectorize
import numpy as np

@guvectorize('complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:],complex128[:,:]',
 '(n,a,b),(n,b,c),(n,c,d),(n,d,e),(n,e,f),(n)->(a,f)',nopython=True, target='parallel')
def RIME_integral(m1,m2,m3,m4,m5,K,V):
    for n in range(K.shape[0]):
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        for e in range(2):
                            for f in range(2):
                                V[a,f] += m1[n,a,b] * m2[n,b,c] * m3[n,c,d] * m4[n,d,e] * m5[n,e,f] * K[n]

@guvectorize('float64[:],float64[:],float64[:],float64[:],complex128[:]',
 '(n),(n),(n),(n)->(n)', nopython=True, target='parallel')
def spinor_rotation(q,u,cos,sin,qun):
    qun = (q + 1j*u) * (cos + 1j*sin)


@guvectorize('complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:],complex128[:,:]',
 '(n,a,b),(n,b,c),(n,c,d),(n)->(a,d)',nopython=True, target='parallel')
def instrRIME_integral(m1,m2,m3,m4,m5,K,V):
    for n in range(K.shape[0]):
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        V[a,f] += m1[n,a,b] * m2[n,b,c] * m3[n,c,d] * K[n]


@jit(nopython=True)
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
    for n in range(m1.shape[0]):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    m_out[n,i,k] += m1[n,i,j] * m2[n,j,k]
    return m_out

@guvectorize('complex128[:,:],complex128[:,:],complex128[:,:]', '(a,b),(b,c)->(a,c)')
def _M(m1,m2,m_out):
    for i in range(2):
        for j in range(2):
            for k in range(2):
                m_out[i,k] += m1[i,j] * m2[j,k]

# @guvectorize('complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:]',
#  '(n,a,b),(n,b,c),(n,c,d),(n,d,e),(n,e,f)->(n,a,f)', nopython=True, target='parallel')
# def jones_chain(m1,m2,m3,m4,m5,C):
#     for a in range(2):
#         for b in range(2):
#             for c in range(2):
#                 for d in range(2):
#                     for e in range(2):
#                         for f in range(2):
#                             C[a,f] += m1[a,b] * m2[b,c] * m3[c,d] * m4[d,e] * m5[e,f]
# Need this version to loop over pixels? might have a bunch of zeros in C
@guvectorize('complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:]',
 '(n,a,b),(n,b,c),(n,c,d),(n,d,e),(n,e,f)->(n,a,f)', nopython=True, target='parallel')
def jones_chain(m1,m2,m3,m4,m5,C):
    for n in range(C.shape[0]):
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        for e in range(2):
                            for f in range(2):
                                C[n,a,f] += m1[n,a,b] * m2[n,b,c] * m3[n,c,d] * m4[n,d,e] * m5[n,e,f]

@jit(nopython=True)
def compose_4M(m1,m2,m3,m4,m5,C):
    for nu_i in range(C.shape[0]):
        C[nu_i] = M(M(M(M(m1[nu_i],m2[nu_i]),m3[nu_i]),m4[nu_i,]),m5[nu_i])

@guvectorize('complex128[:,:,:],complex128[:], complex128[:,:]', '(n, i, j),(n)->(i,j)',
 nopython=True, target='parallel')
def _RIME_integral(C, K, V):
    """
    C.shape = (npix, 2, 2)
    K.shape = (npix,)

    For each component of the 2x2 coherency tensor field C, sum the product
    C(p)_ij * exp(-2 * pi * i * b.s(p) ) to produce a model visibility V(b)_ij.
    """
    for pi in range(C.shape[0]):
        for i in range(2):
            for j in range(2):
                V[i, j] += C[pi,i,j]*K[pi]
#    V /= np.float(np.size(K))

## Old versions
# @jit(nopython=True)
# def M(m1,m2):
#     """
#     Computes the matrix multiplication of two arrays of matricies m1 and m2.
#     m1.shape = m2.shape = (N,2,2)
#     For each n < N, m_out is the product of the 2x2 matricies m1[n,:,:].m2[n,:,:],
#     where the first index of the matrix corresponds to a row, and the second
#     corresponds to a column.
#
#     Made double-plus-gooder by the @jit decorator from the numba package.
#     """
#     m_out = np.zeros_like(m1)
#     for n in range(len(m1[:,0,0])):
#         for i in range(2):
#             for j in range(2):
#                 for k in range(2):
#                     m_out[n,i,k] += m1[n,i,j] * m2[n,j,k]
#     return m_out
#
# @jit(nopython=True)
# def RIME_integral(C, K, V):
#     """
#     C.shape = (npix, 2, 2)
#     K.shape = (npix,)
#
#     For each component of the 2x2 coherency tensor field C, sum the product
#     C(p)_ij * exp(-2 * pi * i * b.s(p) ) to produce a model visibility V(b)_ij.
#     """
#     npix = np.size(K)
#     for i in range(2):
#         for j in range(2):
#             for pi in range(npix):
#                 V[i, j] += C[pi,i,j]*K[pi]
#     return V / np.float(npix)
