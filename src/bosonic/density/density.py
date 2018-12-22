from __future__ import print_function, absolute_import, division

import numpy as np
from .density_loss import lossy_fock_basis
from ..util import memoize
from ..fock import lossy_basis_size
from ..nonlinear import build_fock_nonlinear_layer


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
def get_reverse_basis_lookup(n, m):
    lookup = dict()
    outputBasis = lossy_fock_basis(n, m)
    for i, state in enumerate(outputBasis):
        lookup[tuple(state)] = i
    return lookup


@memoize
def get_deletion_mapping(n, m, d):
    inputBasis = lossy_fock_basis(n, m)
    outputBasisLookup = get_reverse_basis_lookup(n, m-1)
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


def delete_mode(rho, n, m, d):
    NOut = lossy_basis_size(n, m-1)
    sigma = np.zeros((NOut, NOut), dtype=complex)
    for ak, al, k, l in get_deletion_mapping(n, m, d):
        sigma[ak, al] += rho[k, l]
    return sigma


def apply_nonlinear(rho, theta, numPhotons, numModes):
    U = build_fock_nonlinear_layer(
        numPhotons, numModes, theta, includeZero=True)
    return U.dot(rho).dot(np.conj(U.T))


def apply_U(rho, U):
    return U.dot(rho).dot(np.conj(U.T))
