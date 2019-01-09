from __future__ import print_function, absolute_import, division

import numpy as np
from ..util import memoize
from ..fock import lossy_basis_size, lossy_basis_lookup
from ..fock import lossy_basis as lossy_fock_basis
from ..nonlinear import build_fock_nonlinear_layer
from autograd.extend import primitive, defvjp

# Needed for compile-time information about numpy
cimport numpy as np
cimport cython


@memoize
def get_expansion_modes(n, m, newPhotons):
    basis = lossy_fock_basis(n, m)
    expandedBasis = lossy_fock_basis(n+newPhotons, m+1)
    expansionModes = []
    for i, state in enumerate(basis):
        extendedState = state[:]
        extendedState.append(newPhotons)
        for j, eState in enumerate(expandedBasis):
            if tuple(extendedState) == tuple(eState):
                expansionModes.append(j)
    assert len(expansionModes) == len(basis)
    return expansionModes


def add_mode(rho, n, m, newPhotons=0):
    expansionModes = get_expansion_modes(n, m, newPhotons)
    N = lossy_basis_size(n + newPhotons, m + 1)
    sigma = np.zeros((N, N), dtype=complex)
    for i, l in enumerate(expansionModes):
        for j, m in enumerate(expansionModes):
            sigma[l, m] = rho[i, j]
    return sigma


@memoize
def get_deletion_mapping(n, m, d):
    inputBasis = lossy_fock_basis(n, m)
    outputBasisLookup = lossy_basis_lookup(n, m-1)
    N = len(inputBasis)
    mapping = []
    for k in xrange(N):
        for l in xrange(N):
            if inputBasis[k][d] != inputBasis[l][d]:
                continue
            # Find index for a_k
            akState = list(inputBasis[k])
            del(akState[d])
            ak = outputBasisLookup[tuple(akState)]

            # Find index for a_l
            alState = list(inputBasis[l])
            del(alState[d])
            al = outputBasisLookup[tuple(alState)]

            mapping.append((ak, al, k, l))
    return mapping


@primitive
@cython.boundscheck(False)
def delete_mode(np.ndarray[np.complex128_t, ndim=2] rho,
                size_t n, size_t m, int d):
    cdef size_t NOut = lossy_basis_size(n, m-1)
    cdef np.ndarray[np.complex128_t, ndim= 2] sigma = np.zeros((NOut, NOut), dtype=complex)
    cdef size_t ak, al, k, l
    for ak, al, k, l in get_deletion_mapping(n, m, d):
        sigma[ak, al] = sigma[ak, al] + rho[k, l]
    return sigma


@cython.boundscheck(False)
def delete_mode_vjp(np.ndarray[np.complex128_t, ndim=2] ans,
                    np.ndarray[np.complex128_t, ndim=2] rho,
                    size_t n, size_t m, int d):
    cdef size_t N = rho.shape[0]

    @cython.boundscheck(False)
    def vjp(np.ndarray[np.complex128_t, ndim=2] g):
        cdef np.ndarray[np.complex128_t, ndim= 2] drho = np.zeros((N, N), dtype=complex)
        cdef size_t ak, al, kk, ll
        for ak, al, kk, ll in get_deletion_mapping(n, m, d):
            drho[kk, ll] = drho[kk, ll] + g[ak, al]
        return drho
    return vjp


defvjp(delete_mode, delete_mode_vjp, None, None, None)


def apply_nonlinear(rho, theta, numPhotons, numModes):
    U = build_fock_nonlinear_layer(
        numPhotons, numModes, theta, includeZero=True)
    return U.dot(rho).dot(np.conj(U.T))


def apply_U(rho, U):
    return U.dot(rho).dot(np.conj(U.T))
