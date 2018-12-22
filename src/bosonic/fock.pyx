from __future__ import print_function, absolute_import, division

import numpy as np
import itertools as it
from .util import memoize

cimport numpy as np


# Fast implementations of factorial and binomial assuming integer inputs
@memoize
def factorial(int n):
    cdef long long answer = 1
    for x in range(2, n+1):
        answer = answer * x
    return answer


@memoize
def binom(int n, int m):
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
def lossy_basis(numPhotons, numModes):
    basis = []
    for j in range(numPhotons, 0, -1):
        basis.extend(fock_basis(j, numModes))
    basis.extend([numModes*[0], ])
    return basis


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
