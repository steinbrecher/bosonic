import numpy as np
try:
    from numba import jit
except ImportError:
    def jit(x):
        return x

expansionModesLookup = {}
def get_expansion_modes(n, m, newPhotons):
    try:
        return expansionModesLookup[(n,m,newPhotons)]
    except KeyError:
        pass
    basis = q.lossy_fock_basis(n, m, includeZero=True)
    expandedBasis = q.lossy_fock_basis(n+newPhotons, m+1, includeZero=True)
    expansionModes = []
    for i,state in enumerate(basis):
        extendedState = state[:]
        extendedState.append(newPhotons)
        for j,eState in enumerate(expandedBasis):
            if tuple(extendedState) == tuple(eState):
                expansionModes.append(j)
    assert len(expansionLookup) == len(basis)
    expansionModesLookup[(n,m,newPhotons)] = expansionModes
    return expansionModes

def add_mode(rho, n, m, newPhotons=0):
    expansionModes = get_expansion(n,m,newPhotons)
    N = len(q.lossy_fock_basis(n+newPhotons, m+1, includeZero=True))
    sigma = np.zeros((N,N), dtype=complex)
    for i,l in enumerate(expansionModes):
        for j,m in enumerate(expansionModes):
            sigma[l,m] = rho[i,j]
    return sigma
