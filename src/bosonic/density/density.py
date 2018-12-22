import numpy as np
try:
    from numba import jit, complex128, int64
except ImportError:
    def jit(x):
        return x
from .density_loss import lossy_fock_basis
from ..bosonic_util import memoize
from ..aa_phi import lossy_basis_size

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
    assert len(expansionLookup) == len(basis)
    return expansionModes

def add_mode(rho, n, m, newPhotons=0):
    expansionModes = get_expansion(n, m, newPhotons)
    N = len(lossy_fock_basis(n+newPhotons, m+1))
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
    outputBasis = lossy_fock_basis(n, m-1)
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
    rhoOut = np.zeros((NOut, NOut), dtype=complex)
    for ak, al, k, l in get_deletion_mapping(n, m, d):
        rhoOut[ak, al] += rho[k, l]
    return rhoOut



