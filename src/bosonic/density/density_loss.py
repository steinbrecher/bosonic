import numpy as np
import itertools as it
from numba import jit
from ..aa_phi import aa_phi

def mode_basis(numPhotons, numModes):
    """Generates the mode basis as a list of tuples of length numPhotons
    Each tuple is of the form (mode that photon 1 is in, mode that photon
    2 is in, ..., mode that photon numPhotons is in)
    """
    return list(it.combinations_with_replacement(range(numModes), numPhotons))

fock_basis_lookup = {}
def fock_basis(numPhotons, numModes):
    """Generates the fock basis as a list of lists of length numModes
    The list is choose(numPhotons+numModes-1, numPhotons) long and in the
    order [(numPhotons, 0,...,0), (numPhotons-1, 1, 0, ..., 0), ...].
    """
    try:
        return fock_basis_lookup[(numPhotons, numModes)]
    except KeyError:
        pass
    modeBasisList = mode_basis(numPhotons, numModes)
    ls = []
    for element in modeBasisList:
        ks = []
        for i in range(numModes):
            loc_count = element.count(i)
            ks.append(loc_count)
        ls.append(ks)
    fock_basis_lookup[(numPhotons, numModes)] = ls
    return ls

def lossy_fock_basis(numPhotons, numModes,includeZero=False):
    basis = []
    for j in range(numPhotons, 0, -1):
        basis.extend(fock_basis(j, numModes))
    if includeZero:
        basis.extend([numModes*[0],])
    return basis

# Step 1 functions
def get_pair_table(n, m):
    inputBasis = lossy_fock_basis(n, m, includeZero=True)
    expandedBasis = lossy_fock_basis(n, 2*m, includeZero=True)
    pairLookupTable = np.zeros((len(expandedBasis), 2), dtype=int)
    d = len(inputBasis[0])

    # Iterate over the basis
    for i,state in enumerate(expandedBasis):
        # Break the state in half
        a = state[:d]
        b = state[d:]
        # Check which input state each half is equal to
        for j,inputState in enumerate(inputBasis):
            if tuple(a) == tuple(inputState):
                pairLookupTable[i,0] = j
            if tuple(b) == tuple(inputState):
                pairLookupTable[i,1] = j
    return pairLookupTable
    
expansionMapLookup = {}
def get_expansion_map(n, m):
    # Memoize this function
    try:
        return expansionMapLookup[(n,m)]
    except KeyError:
        pass
    inputBasis = lossy_fock_basis(n, m, includeZero=True)
    expandedBasis = lossy_fock_basis(n, 2*m, includeZero=True)
    pairLookupTable = get_pair_table(n, m)
    zeroStateIdx = len(inputBasis)-1
    
    expansionMap = np.zeros((len(inputBasis),), dtype=int)
    for i in xrange(len(inputBasis)):
        for j in xrange(pairLookupTable.shape[0]):
            if pairLookupTable[j,0] == i and pairLookupTable[j,1] == zeroStateIdx:
                expansionMap[i] = j
    expansionMapLookup[(n,m)] = expansionMap
    return expansionMap

#@jit
def expand_density(rho, n, m):
    """Expand a density matrix (assumed over a lossy fock basis) to twice as many modes
    """
    expansionMap = get_expansion_map(n, m)
    # N is size of expanded basis. Since zeros always get mapped to zeros, this avoids computing the basis here
    N = expansionMap[-1] + 1 
    sigma = np.zeros((N, N), dtype=complex)
    for i,l in enumerate(expansionMap):
        for j,m in enumerate(expansionMap):
            sigma[l,m] = rho[i,j]
    return sigma

# Step 2 functions
@jit
def single_loss_matrix(eta, i, j, m):
    #gives a loss matrix between modes i,j for loss eta (i.e. T=1-eta)
    M = np.eye(2*m, dtype=complex)
    M[i,i]=np.sqrt(1-eta)
    M[j,j]=-np.sqrt(1-eta)
    M[i,j]=np.sqrt(eta)
    M[j,i]=np.sqrt(eta)
    return M

@jit
def full_loss_matrix(eta, m):
    #multiply single loss matrices:
    ls = []
    for i in range(m):
        ls.append(single_loss_matrix(eta, i, m+i, m))
    return reduce(np.dot, ls)

lossMatrixLookup = {}
def get_loss_matrix(eta,n,m):
    try:
        return lossMatrixLookup[(eta,n,m)]
    except KeyError:
        pass
    
    N = len(lossy_fock_basis(n,2*m,includeZero=True))
    U = full_loss_matrix(eta, m)
    UFock = np.eye(N, dtype=complex)
    count = 0
    for numPhotons in range(n,0,-1):
        UHere = aa_phi(U, numPhotons)
        NHere = UHere.shape[0]
        UFock[count:count+NHere, count:count+NHere] = UHere
        count += NHere
    lossMatrixLookup[(eta,n,m)] = UFock
    return UFock

#@jit
def apply_loss(rho, eta, n, m):
    sigma = expand_density(rho, n, m)
    U = get_loss_matrix(eta, n, m)
    return U.dot(sigma).dot(np.conj(U.T))

traceTableLookup = {}
#@jit
def get_trace_table(n,m):
    try:
        return traceTableLookup[(n,m)]
    except KeyError:
        pass
    
    pairTable = get_pair_table(n,m)
    N = pairTable.shape[0]
    A = pairTable[:,0]
    B = pairTable[:,1]
    
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
        traceTable[i,:2] = klPairs[i]
        traceTable[i,2:] = aPairs[i]
    traceTableLookup[(n,m)] = traceTable
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
    traceTable = get_trace_table(n,m)
    for i in xrange(traceTable.shape[0]):
        k, l, ak, al = traceTable[i,:]
        rhoOut[ak,al] += sigma[k,l]
    return rhoOut

