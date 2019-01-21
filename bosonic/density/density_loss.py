from __future__ import print_function, absolute_import, division

import numpy as np
try:
    from numba import jit
except ImportError:
    def jit(x):
        return x
from ..aa_phi import aa_phi
from ..fock import lossy_basis, lossy_basis_size
from ..util import memoize


# Step 1 functions
@memoize
def get_pair_table(n, m):
    inputBasis = lossy_basis(n, m)
    expandedBasis = lossy_basis(n, 2*m)
    pairLookupTable = np.zeros((len(expandedBasis), 2), dtype=int)
    d = len(inputBasis[0])

    # Iterate over the basis
    for i, state in enumerate(expandedBasis):
        # Break the state in half
        a = state[:d]
        b = state[d:]
        # Check which input state each half is equal to
        for j, inputState in enumerate(inputBasis):
            if tuple(a) == tuple(inputState):
                pairLookupTable[i, 0] = j
            if tuple(b) == tuple(inputState):
                pairLookupTable[i, 1] = j
    return pairLookupTable


@memoize
def get_expansion_map(n, m):
    inputBasis = lossy_basis(n, m)
    pairLookupTable = get_pair_table(n, m)
    zeroStateIdx = len(inputBasis)-1

    expansionMap = np.zeros((len(inputBasis),), dtype=int)
    for i in xrange(len(inputBasis)):
        for j in xrange(pairLookupTable.shape[0]):
            if pairLookupTable[j, 0] == i:
                if pairLookupTable[j, 1] == zeroStateIdx:
                    expansionMap[i] = j
    return expansionMap


def expand_density_for_loss(rho, n, m):
    """Expand a density matrix (assumed over a lossy fock basis) to
    twice as many modes
    """
    expansionMap = get_expansion_map(n, m)
    # N is size of expanded basis. Since zeros always get mapped to zeros,
    # this avoids computing the basis here
    N = expansionMap[-1] + 1
    sigma = np.zeros((N, N), dtype=complex)
    for i, l in enumerate(expansionMap):
        for j, m in enumerate(expansionMap):
            sigma[l, m] = rho[i, j]
    return sigma


# Step 2 functions
@memoize
def single_loss_matrix(eta, i, j, m):
    # gives a loss matrix between modes i,j for loss eta (i.e. T=1-eta)
    M = np.eye(2*m, dtype=complex)
    M[i, i] = np.sqrt(1-eta)
    M[j, j] = -np.sqrt(1-eta)
    M[i, j] = np.sqrt(eta)
    M[j, i] = np.sqrt(eta)
    return M


@memoize
def full_loss_matrix(eta, m):
    # multiply single loss matrices:
    ls = []
    for i in range(m):
        ls.append(single_loss_matrix(eta, i, m+i, m))
    return reduce(np.dot, ls)


@memoize
def get_loss_matrix(eta, n, m):
    N = lossy_basis_size(n, 2*m)
    U = full_loss_matrix(eta, m)
    UFock = np.eye(N, dtype=complex)
    count = 0
    for numPhotons in range(n, 0, -1):
        UHere = aa_phi(U, numPhotons)
        NHere = UHere.shape[0]
        UFock[count:count+NHere, count:count+NHere] = UHere
        count += NHere
    return UFock


@jit
def apply_loss(rho, eta, n, m):
    sigma = expand_density_for_loss(rho, n, m)
    U = get_loss_matrix(eta, n, m)
    return U.dot(sigma).dot(np.conj(U.T))


@memoize
def get_trace_table(n, m):
    pairTable = get_pair_table(n, m)
    N = pairTable.shape[0]
    A = pairTable[:, 0]
    B = pairTable[:, 1]

    # N = dim(sigma)
    N = pairTable.shape[0]

    klPairs = []
    aPairs = []
    for k in xrange(N):
        for l in xrange(N):
            if B[k] == B[l]:
                klPairs.append((k, l))
                aPairs.append((A[k], A[l]))
    traceTable = np.zeros((len(klPairs), 4), dtype=int)
    for i in xrange(len(klPairs)):
        traceTable[i, :2] = klPairs[i]
        traceTable[i, 2:] = aPairs[i]
    return traceTable


@jit
def apply_density_loss(rho, n, m, eta):
    """Applies loss to an input density matrix and traces out loss modes
    rho: density matrix over lossy basis (including zero state)
    n: number of photons
    m: number of modes
    eta: power loss (i.e. transmission = 1-eta; t=sqrt(1-eta))
    Returns density matrix of same shape as rho
    """
    sigma = apply_loss(rho, eta, n, m)
    rhoOut = np.zeros(rho.shape, dtype=complex)
    traceTable = get_trace_table(n, m)
    for i in xrange(traceTable.shape[0]):
        k, l, ak, al = traceTable[i, :]
        rhoOut[ak, al] += sigma[k, l]
    return rhoOut


@memoize
def get_qd_mode_pairs(n, m):
    basis = lossy_basis(n, 2*m)
    pairs = np.zeros((m, 2), dtype=int)
    for i in xrange(m):
        for j, state in enumerate(basis):
            if state[i] == 2 and state[i+m] == 0:
                pairs[i, 0] = j
            if state[i] == 0 and state[i+m] == 2:
                pairs[i, 1] = j
    return pairs


# Quantum dot nonlinearity functions below this
@memoize
def get_qd_loss_unitary(n, m, etas):
    etas = np.array(etas)
    if etas.size == 1:
        etas = etas * np.ones(m)
    if etas.size == m:
        pass
    else:
        raise ValueError(
            "etas must be a single number or have length equal to m")
    pairs = get_qd_mode_pairs(n, m)
    N = len(lossy_basis(n, 2*m))
    U = np.eye(N, dtype=complex)
    for i in xrange(m):
        eta = etas[i]
        j, k = pairs[i, :]
        U[j, j] = 1 - 2*eta
        U[k, k] = -(1 - 2*eta)
        U[j, k] = 2 * np.sqrt(eta * (1-eta))
        U[k, j] = 2 * np.sqrt(eta * (1-eta))
    return U


def apply_qd(rho, n, m, etas):
    sigma = expand_density_for_loss(rho, n, m)
    U = get_qd_loss_unitary(n, m, etas)
    sigma = U.dot(sigma).dot(np.conj(U.T))
    rhoOut = np.zeros(rho.shape, dtype=complex)
    traceTable = get_trace_table(n, m)
    for i in xrange(traceTable.shape[0]):
        k, l, ak, al = traceTable[i, :]
        rhoOut[ak, al] += sigma[k, l]
    return rhoOut
