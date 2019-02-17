from __future__ import print_function, absolute_import, division

import numpy as np
import itertools as it
from .util import memoize

cimport numpy as np


@memoize
def factorial(int n):
    """Optimized factorial implementation for integers"""
    cdef long long answer = 1
    for x in range(2, n+1):
        answer = answer * x
    return answer


@memoize
def binom(int n, int m):
    """Optimized binomial implementation for integers"""
    cdef unsigned long long numer = 1
    cdef unsigned long long denom = factorial(n-m)
    for x in range(m+1, n+1):
        numer = numer * x
    return numer // denom


@memoize
def mode_basis(numPhotons, numModes):
    """Generates the mode basis as a list of tuples of length numPhotons
    Each tuple is of the form (mode that photon 1 is in, mode that photon
    2 is in, ..., mode that photon numPhotons is in)
    """
    return list(it.combinations_with_replacement(range(numModes), numPhotons))


# Fock basis generation for number of photons & modes
@memoize
def basis(numPhotons, numModes):
    """Generates the fock basis as a list of lists of length numModes
    The list is choose(numPhotons+numModes-1, numPhotons) long and in the
    order [(numPhotons, 0,...,0), (numPhotons-1, 1, 0, ..., 0), ...].
    """
    modeBasisList = mode_basis(numPhotons, numModes)
    ls = []
    for element in modeBasisList:
        ks = []
        for i in range(numModes):
            loc_count = element.count(i)
            ks.append(loc_count)
        ls.append(ks)
    return ls


@memoize
def basis_lookup(n, m):
    lookup = dict()
    outputBasis = basis(n, m)
    for i, state in enumerate(outputBasis):
        lookup[tuple(state)] = i
    return lookup


# Memoized function to build the basis efficiently
# Note: basis is a numpy array here, not a list of lists as fock_basis
# returns
@memoize
def basis_array(int n, int m):
    cdef np.ndarray[np.int_t, ndim= 2] basis_array = np.array(
        basis(n, m), dtype=np.int)
    return basis_array


@memoize
def lossy_basis(numPhotons, numModes):
    lb = []
    for j in range(numPhotons, 0, -1):
        lb.extend(basis(j, numModes))
    lb.extend([numModes*[0], ])
    return lb


@memoize
def lossy_basis_array(int n, int m):
    cdef np.ndarray[np.int_t, ndim= 2] basis_array = np.array(
        lossy_basis(n, m), dtype=np.int)
    return basis_array


@memoize
def lossy_basis_lookup(n, m):
    lookup = dict()
    outputBasis = lossy_basis(n, m)
    for i, state in enumerate(outputBasis):
        lookup[tuple(state)] = i
    return lookup

# Computes binomial(m+n-1, n)
# This is ~5x faster than using scipy.special.binom oddly enough


@memoize
def basis_size(int n, int m):
    cdef int top = n + m - 1
    cdef int numer = 1
    cdef int denom = 1
    for i in range(1, n+1):
        numer = numer * (top + 1 - i)
        denom = denom * i
    return numer // denom


@memoize
def lossy_basis_size(int n, int m):
    cdef size_t i
    cdef size_t basisSize = 0
    for i in range(n+1):
        basisSize += basis_size(i, m)
    return basisSize
