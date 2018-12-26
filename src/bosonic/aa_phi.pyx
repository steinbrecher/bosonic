from __future__ import print_function, absolute_import, division
from cython.parallel cimport prange, parallel

import numpy as np
from libc.stdlib cimport abort, malloc, free
from libc.string cimport memset
from .util import memoize
from .fock import basis as fock_basis
from .fock import basis_array, basis_size, lossy_basis_size, factorial

# Needed for compile-time information about numpy
cimport numpy as np


def fock_to_idx(np.ndarray[np.int_t, ndim=1] S, int n):
    """Converts fock state S to list with s_i copies of the number i
    i.e. state [0,2,1,0]->[1,1,2]
    """
    cdef np.ndarray idx = np.zeros([n, ], dtype=np.int)
    cdef int s
    cdef int count = 0
    for i in range(S.shape[0]):
        s = S[i]
        if s == 0:
            continue
        for j in range(s):
            idx[count] = i
            count += 1
    return idx

# This stuff only gets run once per basis (dim of U and # of photons)
# so may as well cache all of it


@memoize
def build_norm_and_idxs(int n, int m):
    cdef np.ndarray basis = basis_array(n, m)
    cdef int N = basis_size(n, m)
    cdef np.ndarray[np.double_t, ndim = 1] factProducts = np.zeros([N, ], dtype=np.double)
    cdef np.ndarray[np.double_t, ndim= 2] normalization = np.zeros([N, N], dtype=np.double)
    cdef np.ndarray[np.int_t, ndim = 2] idxs = np.zeros([N, n], dtype=np.int)

    for i in range(basis.shape[0]):
        S = basis[i]
        # Generate factorial product for state
        product = 1
        for x in S:
            product *= factorial(x)
        factProducts[i] = np.sqrt(product)

        idxs[i] = fock_to_idx(S, n)
    normalization = np.outer(factProducts, factProducts)
    return (normalization, idxs)


cimport cython


@cython.boundscheck(False)  # turn off bounds-checking for entire function
# turn off negative index wrapping for entire function
@cython.wraparound(False)
def aa_phi3(np.ndarray[np.complex128_t, ndim=2] U, int n):
    assert U.dtype == np.complex128
    cdef int m = U.shape[0]
    cdef int N = basis_size(n, m)
    cdef int i
    cdef int j
    cdef int I
    cdef int J
    cdef np.ndarray[np.complex128_t, ndim= 2] phiU = np.empty([N, N], dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim= 2] U_T = np.empty([m, n], dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim= 2] U_ST = np.empty([n, n], dtype=np.complex128)
    cdef np.ndarray[np.int_t, ndim= 2] idxs
    cdef np.ndarray[np.double_t, ndim= 2] normalization

    normalization, idxs = build_norm_and_idxs(n, m)

    for col in range(N):
        for j in range(n):
            J = idxs[col, j]
            for i in range(m):
                U_T[i, j] = U[i, J]

        for row in range(N):
            for i in range(n):
                I = idxs[row, i]
                for j in range(n):
                    U_ST[i, j] = U_T[I, j]

            phiU[row, col] = permanent(U_ST)
    return phiU / normalization


# New, optimized version of aa_phi that uses OpenMP threads to speed calculation


@cython.boundscheck(False)  # turn off bounds-checking for entire function
# turn off negative index wrapping for entire function
@cython.wraparound(False)
def aa_phi2(np.ndarray[np.complex128_t, ndim=2] U, size_t n):
    assert U.dtype == np.complex128
    cdef size_t m = U.shape[0]
    cdef size_t N = basis_size(n, m)

    cdef size_t row, col

    cdef size_t i, j, I, J

    cdef int gray
    cdef int k  # k needs to be int, not size_t for how we calculate gray
    cdef complex rowsum = 0
    cdef complex rowsumprod
    cdef complex p = 1
    cdef int sgn = 1

    cdef complex * U_T
    cdef complex * U_ST

    cdef np.ndarray[np.complex128_t, ndim= 2] phiU = np.empty([N, N], dtype=np.complex128)

    cdef np.ndarray[np.double_t, ndim= 2] normalization
    cdef np.ndarray[np.int_t, ndim= 2] idxs

    normalization, idxs = build_norm_and_idxs(n, m)
    # If n is odd, we flip the sign of the permanents. More efficient
    # to flip the sign of the normalizations since we have to divide by
    # it later anyway
    if (n % 2) == 1:
        sgn = -1

    with nogil, parallel(num_threads=12):
        U_T = <complex * >malloc(sizeof(complex) * m * n)
        if U_T == NULL:
            abort()
        U_ST = <complex * >malloc(sizeof(complex) * n * n)
        if U_ST == NULL:
            abort()
        for col in prange(N, schedule='dynamic'):
            # Populate U_T once per column
            for j in range(n):
                J = idxs[col, j]
                for i in range(m):
                    U_T[i + j*m] = U[i, J]

            for row in range(N):
                # Populate U_ST for each row
                for i in range(n):
                    I = idxs[row, i]
                    for j in range(n):
                        U_ST[i + j*n] = U_T[I + j*m]

                # Calculate permanent of U_ST
                # (Ignoring the sign due to n being even or odd; that
                #  got rolled into the normalization above)
                p = 0

                for k in range(2**n):
                    gray = k ^ (k >> 1)
                    rowsumprod = 1 - 2 * (k % 2)
                    for i in range(n):
                        rowsum = 0
                        for j in range(n):
                            if (gray >> j) & 1 == 1:
                                rowsum = rowsum + U_ST[i + j*n]
                        rowsumprod = rowsumprod * rowsum
                    p = p + rowsumprod
                phiU[row, col] = sgn*p
        free(U_T)
        free(U_ST)
    return phiU / normalization


kIdxLookup = {}


# @cython.boundscheck(False)  # turn off bounds-checking for entire function
# # turn off negative index wrapping for entire function
# @cython.wraparound(False)
@memoize
def build_kIdxs(int n):
    try:
        return kIdxLookup[n]
    except KeyError:
        pass

    cdef np.ndarray[np.int_t, ndim= 1] kIdxs = np.empty([2**n, ], dtype=np.int)
    cdef np.ndarray[np.int_t, ndim= 1] kSgns = np.empty([2**n, ], dtype=np.int)
    cdef int k, gray, lastGray, deltaGray, count, kIdx, kSgn

    # Construct lookup tables for which bit in the gray code flipped from
    # the previous k and whether it was a 0->1 transition or a 1-> 0
    # transition.
    kIdxs[0] = 0
    kSgns[0] = 1
    with nogil, parallel():
        for k in prange(1, 2**n, schedule='dynamic'):
            # Since this is in parallel, can't save the previous gray number
            lastGray = (k-1) ^ ((k-1) >> 1)
            # Get the current gray number
            gray = k ^ (k >> 1)
            # If gray is less than lastGray, a bit was flipped from 1->0
            if gray < lastGray:
                kSgns[k] = -1
            else:
                kSgns[k] = 1
            # XOR the two together; there should only be one bit set to 1
            deltaGray = gray ^ lastGray
            # Shift until we find the bit set to 1
            count = 0
            while (deltaGray & 1) == 0:
                deltaGray = deltaGray >> 1
                count = count + 1
            # Save how many bit shifts we needed
            kIdxs[k] = count
            lastGray = gray
    kIdxLookup[n] = (kIdxs, kSgns)
    return (kIdxs, kSgns)


# Cython implementation of permanent calculation
@cython.boundscheck(False)  # turn off bounds-checking for entire function
# turn off negative index wrapping for entire function
@cython.wraparound(False)
def permanent(a):
    # def permanent(np.ndarray[np.complex128_t, ndim=2] a):
    cdef int n = a.shape[0]
    cdef np.ndarray[np.complex128_t, ndim = 2] partials = np.empty([n, n], dtype=np.complex128)
    cdef int i
    cdef int j
    cdef int k
    cdef int gray
    cdef complex rowsum
    cdef complex rowsumprod
    cdef complex p = 0

    for k in range(2**n):
        gray = k ^ (k >> 1)
        rowsumprod = 1 - 2 * (k % 2)
        for i in range(n):
            rowsum = 0
            for j in range(n):
                if (gray >> j) & 1 == 1:
                    rowsum += a[i, j]
            rowsumprod *= rowsum
        p += rowsumprod
    if n % 2 == 1:
        p *= -1
    return p


# Cython implementation of permanent calculation
@cython.boundscheck(False)  # turn off bounds-checking for entire function
# turn off negative index wrapping for entire function
@cython.wraparound(False)
def minor(np.ndarray[np.complex128_t, ndim=2] a, size_t x, size_t y):
    cdef size_t n = a.shape[0]
    cdef np.ndarray[np.complex128_t, ndim = 2] b = np.zeros((n-1, n-1), dtype=np.complex128)
    cdef size_t i
    cdef size_t j
    cdef size_t ii = 0
    cdef size_t jj = 0
    for i in range(n):
        if i == x:
            continue
        jj = 0
        for j in range(n):
            if j == y:
                continue
            b[ii, jj] = a[i, j]
            jj += 1
        ii += 1
    return b


# Cython implementation of permanent calculation
@cython.boundscheck(False)  # turn off bounds-checking for entire function
# turn off negative index wrapping for entire function
@cython.wraparound(False)
def permanent_vjp(complex ans, np.ndarray[np.complex128_t, ndim=2] x):
    cdef size_t n = x.shape[0]
    J = np.zeros((n, n), dtype=np.complex128)
    cdef size_t i
    cdef size_t j
    for i in range(n):
        for j in range(n):
            J[i, j] = permanent(minor(x, i, j))

    def vjp(complex g):
        return g * J

    return vjp


@memoize
def build_ust_idxs(n, m):
    _, idxs = build_norm_and_idxs(n, m)
    N = basis_size(n, m)
    ustIdxs = np.zeros((N, N, n, n, 2), dtype=int)
    U_Tidx = np.zeros((m, n, 2), dtype=int)
    for col in range(N):
        for i in range(m):
            for j in range(n):
                U_Tidx[i, j, :] = (i, idxs[col][j])

        for row in range(N):
            for i in range(n):
                for j in range(n):
                    ustIdxs[row, col, i, j, :] = U_Tidx[idxs[row][i], j, :]
    return ustIdxs

@cython.boundscheck(False)  # turn off bounds-checking for entire function
# turn off negative index wrapping for entire function
@cython.wraparound(False)
def build_Js(np.ndarray[np.complex128_t, ndim=2] a, size_t n, size_t m):
    cdef size_t N = basis_size(n, m)
    cdef size_t minorN = n - 1
    cdef np.ndarray[np.complex128_t, ndim = 4] Js = np.zeros((N, N, n, n), dtype=complex)
    cdef np.ndarray[np.int_t, ndim = 5] ustIdxs = build_ust_idxs(n, m)
    cdef np.ndarray[np.double_t, ndim = 2] normalization
    cdef np.ndarray[np.int_t, ndim = 2] idxs
    cdef np.ndarray[np.int_t, ndim = 1] kIdxs
    cdef np.ndarray[np.int_t, ndim = 1] kSgns
    cdef int kIdx
    cdef int kSgn

    # Get the indexes of which element of the gray code changes each
    # iteration. Only run this once per n, so it's wrapped into its own
    # function with a dictionary memoization.
    kIdxs, kSgns = build_kIdxs(minorN)
    # Get normalization matrix for phi(U) and indexes of which elements
    # of U map to elements of U_T and U_ST (see function for more
    # details)
    normalization, idxs = build_norm_and_idxs(n, m)

    cdef size_t row
    cdef size_t col
    cdef size_t i
    cdef size_t j
    cdef size_t x
    cdef size_t y
    cdef size_t ii
    cdef size_t jj
    cdef int k
    cdef int sgn = 1
    cdef complex rowsum = 0
    cdef complex rowsumprod
    cdef complex perm = 1

    if (minorN % 2) == 1:
        sgn = -1
    cdef complex * U_ST
    cdef complex * minorU
    cdef complex * rowsums

    with nogil, parallel(num_threads=12):
        U_ST = < complex * >malloc(sizeof(complex) * n * n)
        if U_ST == NULL:
            abort()

        minorU = < complex * >malloc(sizeof(complex) * (n-1) * (n-1))
        if minorU == NULL:
            abort()

        rowSums = < complex*>malloc(sizeof(complex) * minorN)
        if rowSums == NULL:
            abort()

        for row in prange(N, schedule='dynamic'):
            for col in range(N):
                for x in range(n):
                    for y in range(n):
                        # Step 1: Build U_ST
                        for i in range(n):
                           for j in range(n):
                                U_ST[n*i + j] = a[ustIdxs[row, col, i, j, 0],
                                                  ustIdxs[row, col, i, j, 1]]

                        # Step 2: Build minor
                        ii = 0
                        for i in range(n):
                            if i == x:
                                continue
                            jj = 0
                            for j in range(n):
                                if j == y:
                                    continue
                                minorU[ii * minorN + jj] = U_ST[n*i + j]
                                jj = jj + 1
                            ii = ii + 1
                        
                        # Step 3: Calculate permanent of minor
                        perm = 0

                        # Initialize the rowSums to 0 for the permanent
                        memset(rowSums, 0, (n-1) * sizeof(complex))

                        # Iterate over all the set combinations (whee exponential algorithms)
                        # Don't have to start at 0 since gray(0) = 0, i.e. there
                        # are no bits set and thus no elements included.
                        for k in range(1, 2**minorN):
                            # Slightly more efficient to write these to a local variable
                            # for rapid access
                            kIdx = kIdxs[k]
                            kSgn = kSgns[k]

                            # Update the rowSums, adding if sgn is 1, subtracting otherwise
                            for i in range(minorN):
                                rowSums[i] = rowSums[i] + kSgn * minorU[minorN * kIdx + i]

                            # Set rowsumprod to 1 for even k, -1 for odd k
                            rowsumprod = 1 - 2 * (k % 2)

                            # Compute product over the rowsums
                            for i in range(minorN):
                                rowsumprod = rowsumprod * rowSums[i]

                            # Collect the permanent sum
                            perm = perm + rowsumprod
                        Js[row, col, x, y] = sgn * perm / normalization[row, col]
        free(U_ST)
        free(minorU)
        free(rowSums)
    return Js
    
#@cython.boundscheck(False)  # turn off bounds-checking for entire function
# turn off negative index wrapping for entire function
#@cython.wraparound(False)
def aa_phi_vjp_fast(np.ndarray[np.complex128_t, ndim=2] g,
                    np.ndarray[np.complex128_t, ndim=2] a, size_t n, size_t m):
    cdef np.ndarray[np.complex128_t, ndim= 2] out = np.zeros((m, m), dtype=np.complex128)

    cdef complex J
    cdef size_t N = basis_size(n, m)
    cdef size_t minorN = n - 1
    cdef np.ndarray[np.complex128_t, ndim = 4] Js = np.zeros((N, N, n, n), dtype=complex)
    cdef np.ndarray[np.int_t, ndim = 5] ustIdxs = build_ust_idxs(n, m)
    cdef np.ndarray[np.double_t, ndim = 2] normalization
    cdef np.ndarray[np.int_t, ndim = 2] idxs
    cdef np.ndarray[np.int_t, ndim = 1] kIdxs
    cdef np.ndarray[np.int_t, ndim = 1] kSgns
    
    cdef int kIdx
    cdef int kSgn

    # Get the indexes of which element of the gray code changes each
    # iteration. Only run this once per n, so it's wrapped into its own
    # function with a dictionary memoization.
    kIdxs, kSgns = build_kIdxs(minorN)
    # Get normalization matrix for phi(U) and indexes of which elements
    # of U map to elements of U_T and U_ST (see function for more
    # details)
    normalization, idxs = build_norm_and_idxs(n, m)

    cdef size_t row
    cdef size_t col
    cdef size_t i
    cdef size_t j
    cdef size_t x
    cdef size_t y
    cdef size_t ii
    cdef size_t jj
    cdef int k
    cdef int sgn = 1
    cdef complex rowsum = 0
    cdef complex rowsumprod
    cdef complex perm = 1

    if (minorN % 2) == 1:
        sgn = -1
    cdef complex * U_ST
    cdef complex * minorU
    cdef complex * rowsums

    with nogil, parallel(num_threads=12):
        U_ST = < complex * >malloc(sizeof(complex) * n * n)
        if U_ST == NULL:
            abort()

        minorU = < complex * >malloc(sizeof(complex) * (n-1) * (n-1))
        if minorU == NULL:
            abort()

        rowSums = < complex*>malloc(sizeof(complex) * minorN)
        if rowSums == NULL:
            abort()

        for row in prange(N, schedule='dynamic'):
            for col in range(N):
                for x in range(n):
                    for y in range(n):
                        # Step 1: Build U_ST
                        for i in range(n):
                           for j in range(n):
                                U_ST[n*i + j] = a[ustIdxs[row, col, i, j, 0],
                                                  ustIdxs[row, col, i, j, 1]]

                        # Step 2: Build minor
                        ii = 0
                        for i in range(n):
                            if i == x:
                                continue
                            jj = 0
                            for j in range(n):
                                if j == y:
                                    continue
                                minorU[ii * minorN + jj] = U_ST[n*i + j]
                                jj = jj + 1
                            ii = ii + 1
                        
                        # Step 3: Calculate permanent of minor
                        perm = 0

                        # Initialize the rowSums to 0 for the permanent
                        memset(rowSums, 0, (n-1) * sizeof(complex))

                        # Iterate over all the set combinations (whee exponential algorithms)
                        # Don't have to start at 0 since gray(0) = 0, i.e. there
                        # are no bits set and thus no elements included.
                        for k in range(1, 2**minorN):
                            # Slightly more efficient to write these to a local variable
                            # for rapid access
                            kIdx = kIdxs[k]
                            kSgn = kSgns[k]

                            # Update the rowSums, adding if sgn is 1, subtracting otherwise
                            for i in range(minorN):
                                rowSums[i] = rowSums[i] + kSgn * minorU[minorN * kIdx + i]

                            # Set rowsumprod to 1 for even k, -1 for odd k
                            rowsumprod = 1 - 2 * (k % 2)

                            # Compute product over the rowsums
                            for i in range(minorN):
                                rowsumprod = rowsumprod * rowSums[i]

                            # Collect the permanent sum
                            perm = perm + rowsumprod
                        Js[row, col, x, y] = g[row, col] * sgn * perm / normalization[row, col]
                
        free(U_ST)
        free(minorU)
        free(rowSums)
    for row in range(N):
        for col in range(N):
            for x in range(n):
                for y in range(n):
                    i = ustIdxs[row, col, x, y, 0]
                    j = ustIdxs[row, col, x, y, 1]
                    out[i, j] = out[i, j] + Js[row, col, x, y]
                    
    return out
# @cython.boundscheck(False)  # turn off bounds-checking for entire function
# # turn off negative index wrapping for entire function
# @cython.wraparound(False)
def aa_phi_vjp(ans, a, size_t n, size_t m):
    def vjp(np.ndarray[np.complex128_t, ndim= 2] g):
        return aa_phi_vjp_fast(g, a, n, m)
    return vjp


# @cython.boundscheck(False)  # turn off bounds-checking for entire function
# # turn off negative index wrapping for entire function
# @cython.wraparound(False)
# def aa_phi_vjp(np.ndarray[np.complex128_t, ndim= 2] ans,
#                np.ndarray[np.complex128_t, ndim= 2] a, size_t n, size_t m):
#     Js = build_Js(a, n, m)
#     cdef size_t N = basis_size(n, m)

#     ustIdxs = build_ust_idxs(n, m)
#     cdef np.ndarray[np.double_t, ndim= 2] normalization = build_norm_and_idxs(n, m)[0]
#     cdef np.ndarray[np.complex128_t, ndim= 2] U_ST
    
#     @cython.boundscheck(False)  # turn off bounds-checking for entire function
#     # turn off negative index wrapping for entire function
#     @cython.wraparound(False)
#     def vjp(np.ndarray[np.complex128_t, ndim= 2] g):
#         cdef np.ndarray[np.complex128_t, ndim= 2] out = np.zeros((m, m), dtype=np.complex128)
#         cdef size_t row, col, i, j, ii, jj

#         for row in range(N):
#             for col in range(N):
#                 for i in range(n):
#                     for j in range(n):
#                         ii, jj = ustIdxs[row, col, i, j, :]
#                         out[ii, jj] = out[ii,jj] + g[row, col] * Js[row, col, i, j]
#         return out
#     return vjp



# Calculate the block-diagonal version of phi over the lossy basis


@cython.boundscheck(False)  # turn off bounds-checking for entire function
# turn off negative index wrapping for entire function
@cython.wraparound(False)
def aa_phi_lossy(np.ndarray[np.complex128_t, ndim = 2] U, size_t n):
    assert U.dtype == np.complex128
    cdef size_t m=U.shape[0]
    cdef size_t N=lossy_basis_size(n, m)
    cdef np.ndarray[np.complex128_t, ndim= 2] S=np.eye(N, dtype = np.complex128)

    cdef size_t nn=n
    cdef size_t count=0
    cdef size_t NN

    while nn > 0:
        NN=basis_size(nn, m)
        phiU=aa_phi(U, nn)
        S[count:count+NN, count:count+NN]=phiU
        nn -= 1
        count += NN
    return S

# Improvement of the threaded version of aa_phi to take advantage of
# iterating in gray code order (i.e. the rowsums only change by one
# element of U_ST for each k, so we can save them and update accordingly)


@cython.boundscheck(False)  # turn off bounds-checking for entire function
# turn off negative index wrapping for entire function
@cython.wraparound(False)
def aa_phi(np.ndarray[np.complex128_t, ndim = 2] U, size_t n):
    """Computes multi-particle unitary for a given number of photons

    Parameters
    ----------
    U : array of complex
        Single-photon unitary to be transformed
    n : int
        Number of photons to compute unitary for

    Returns
    -------
    array of complex128
        Multiparticle unitary over the fock basis.

    See Also
    -------
    fock_basis : Generates fock basis for `n` photons over `m` modes

    Notes
    -----
    The computational complexity of this function is not for the
    faint of heart. Specifically it goes as :math:`O(Choose[n+m-1,n] * n * 2^n)`
    where `m` is dimensionality of `U` and `n` is the number of photons.

    References
    ----------
    [1] Aaronson, Scott, and Alex Arkhipov. "The computational
    complexity of linear optics." In Proceedings of the forty-third
    annual ACM symposium on Theory of computing, pp. 333-342. ACM, 2011.
    """
    assert U.dtype == np.complex128
    cdef size_t m=U.shape[0]
    cdef size_t N=basis_size(n, m)

    cdef size_t row, col

    cdef size_t i, j, I, J

    cdef int k
    cdef int sgn=1
    cdef complex rowsum=0
    cdef complex rowsumprod
    cdef complex perm=1

    cdef complex * U_T
    cdef complex * U_ST
    cdef complex * rowsums

    cdef np.ndarray[np.complex128_t, ndim= 2] phiU=np.empty([N, N], dtype = np.complex128)

    cdef np.ndarray[np.double_t, ndim= 2] normalization
    cdef np.ndarray[np.int_t, ndim= 2] idxs

    cdef np.ndarray[np.int_t, ndim= 1] kIdxs
    cdef np.ndarray[np.int_t, ndim= 1] kSgns
    cdef int kIdx
    cdef int kSgn

    # Get the indexes of which element of the gray code changes each
    # iteration. Only run this once per n, so it's wrapped into its own
    # function with a dictionary memoization.
    kIdxs, kSgns=build_kIdxs(n)
    # Get normalization matrix for phi(U) and indexes of which elements
    # of U map to elements of U_T and U_ST (see function for more
    # details)
    normalization, idxs=build_norm_and_idxs(n, m)
    # If n is odd, we flip the sign of the permanents. More efficient
    # to flip the sign of the normalizations since we have to divide by
    # it later anyway
    if (n % 2) == 1:
        sgn = -1

    with nogil, parallel(num_threads = 12):
        # Note: malloc'd 2d arrays need to be accessed as a 1d array.
        # Access pattern in C is arr[col + row * numCol]
        U_T=< complex * >malloc(sizeof(complex) * m * n)
        if U_T == NULL:
            abort()
        U_ST = < complex * >malloc(sizeof(complex) * n * n)
        if U_ST == NULL:
            abort()
        rowSums = < complex*>malloc(sizeof(complex) * n)
        if rowSums == NULL:
            abort()

        for col in prange(N, schedule='dynamic'):
            # Populate U_T once per column
            for j in range(n):
                J = idxs[col, j]
                for i in range(m):
                    U_T[i + j*m] = U[i, J]

            for row in range(N):
                # Populate U_ST for each row
                for i in range(n):
                    I = idxs[row, i]
                    for j in range(n):
                        U_ST[i + j*n] = U_T[I + j*m]

                # Calculate permanent of U_ST
                perm = 0

                # Initialize the rowSums to 0 for the permanent
                memset(rowSums, 0, n * sizeof(complex))

                # Iterate over all the set combinations (whee exponential algorithms)
                # Don't have to start at 0 since gray(0) = 0, i.e. there
                # are no bits set and thus no elements included.
                for k in range(1, 2**n):
                    # Slightly more efficient to write these to a local variable
                    # for rapid access
                    kIdx = kIdxs[k]
                    kSgn = kSgns[k]

                    # Update the rowSums, adding if sgn is 1, subtracting otherwise
                    for i in range(n):
                        rowSums[i] = rowSums[i] + kSgn * U_ST[i + kIdx*n]

                    # Set rowsumprod to 1 for even k, -1 for odd k
                    rowsumprod = 1 - 2 * (k % 2)

                    # Compute product over the rowsums
                    for i in range(n):
                        rowsumprod = rowsumprod * rowSums[i]

                    # Collect the permanent sum
                    perm = perm + rowsumprod

                # Save the permanent to its entry of phiU, divdided
                # by normalization constant. Without threads, division is
                # better done at the final return (dividing the matrices
                # rather than explicitly doing it elementwise) but seems
                # to be more efficient this way w/ threads.
                phiU[row, col] = sgn * perm / normalization[row, col]
        free(U_T)
        free(U_ST)
        free(rowSums)

    return phiU

# Improvement of the threaded version of aa_phi to take advantage of
# iterating in gray code order (i.e. the rowsums only change by one
# element of U_ST for each k, so we can save them and update accordingly)


@cython.boundscheck(False)  # turn off bounds-checking for entire function
# turn off negative index wrapping for entire function
@cython.wraparound(False)
def aa_phi_restricted(np.ndarray[np.complex128_t, ndim=2] U, size_t n,
                      np.ndarray[np.int_t, ndim=1] idxIn, np.ndarray[np.int_t, ndim=1] idxOut):
    """Computes multi-particle unitary for a given number of photons

    Parameters
    ----------
    U : array of complex
        Single-photon unitary to be transformed
    n : int
        Number of photons to compute unitary for
    idxIn : array of int
            Indices of input states
    idxOut : array of int
             Indices of output states

    Returns
    -------
    array of complex128
        Multiparticle unitary over the fock basis. 

    See Also
    -------
    fock_basis : Generates fock basis for `n` photons over `m` modes

    Notes
    -----
    The computational complexity of this function is not for the 
    faint of heart. Specifically it goes as :math:`O(Choose[n+m-1,n] * n * 2^n)`
    where `m` is dimensionality of `U` and `n` is the number of photons.

    References
    ----------
    [1] Aaronson, Scott, and Alex Arkhipov. "The computational 
    complexity of linear optics." In Proceedings of the forty-third 
    annual ACM symposium on Theory of computing, pp. 333-342. ACM, 2011.
    """
    assert U.dtype == np.complex128
    cdef size_t m = U.shape[0]
    cdef size_t N = basis_size(n, m)

    cdef size_t row, col, rowId, colId

    cdef size_t i, j, I, J

    cdef int k
    cdef int sgn = 1
    cdef complex rowsum = 0
    cdef complex rowsumprod
    cdef complex perm = 1

    cdef complex * U_T
    cdef complex * U_ST
    cdef complex * rowsums

    cdef int numIn = idxIn.size
    cdef int numOut = idxOut.size

    cdef np.ndarray[np.complex128_t, ndim = 2] phiU = np.empty([numOut, numIn], dtype=np.complex128)

    cdef np.ndarray[np.double_t, ndim = 2] normalization
    cdef np.ndarray[np.int_t, ndim = 2] idxs

    cdef np.ndarray[np.int_t, ndim= 1] kIdxs
    cdef np.ndarray[np.int_t, ndim= 1] kSgns
    cdef int kIdx
    cdef int kSgn

    # Get the indexes of which element of the gray code changes each
    # iteration. Only run this once per n, so it's wrapped into its own
    # function with a dictionary memoization.
    kIdxs, kSgns = build_kIdxs(n)
    # Get normalization matrix for phi(U) and indexes of which elements
    # of U map to elements of U_T and U_ST (see function for more
    # details)
    normalization, idxs = build_norm_and_idxs(n, m)
    # If n is odd, we flip the sign of the permanents. More efficient
    # to flip the sign of the normalizations since we have to divide by
    # it later anyway
    if (n % 2) == 1:
        sgn = -1

    with nogil, parallel(num_threads=12):
        # Note: malloc'd 2d arrays need to be accessed as a 1d array.
        U_T = <complex * >malloc(sizeof(complex) * m * n)
        if U_T == NULL:
            abort()
        U_ST = <complex * >malloc(sizeof(complex) * n * n)
        if U_ST == NULL:
            abort()
        rowSums = <complex*>malloc(sizeof(complex) * n)
        if rowSums == NULL:
            abort()

        for colId in prange(numIn, schedule='dynamic'):
            col = idxIn[colId]
            # Populate U_T once per column
            for j in range(n):
                J = idxs[col, j]
                for i in range(m):
                    U_T[i + j*m] = U[i, J]

            for rowId in range(numOut):
                row = idxOut[rowId]
                # Populate U_ST for each row
                for i in range(n):
                    I = idxs[row, i]
                    for j in range(n):
                        U_ST[i + j*n] = U_T[I + j*m]

                # Calculate permanent of U_ST
                perm = 0

                # Initialize the rowSums to 0 for the permanent
                memset(rowSums, 0, n * sizeof(complex))

                # Iterate over all the set combinations (whee exponential algorithms)
                # Don't have to start at 0 since gray(0) = 0, i.e. there
                # are no bits set and thus no elements included.
                for k in range(1, 2**n):
                    # Slightly more efficient to write these to a local variable
                    # for rapid access
                    kIdx = kIdxs[k]
                    kSgn = kSgns[k]

                    # Update the rowSums, adding if sgn is 1, subtracting otherwise
                    for i in range(n):
                        rowSums[i] = rowSums[i] + kSgn * U_ST[i + kIdx*n]

                    # Set rowsumprod to 1 for even k, -1 for odd k
                    rowsumprod = 1 - 2 * (k % 2)

                    # Compute product over the rowsums
                    for i in range(n):
                        rowsumprod = rowsumprod * rowSums[i]

                    # Collect the permanent sum
                    perm = perm + rowsumprod

                # Save the permanent to its entry of phiU, divdided
                # by normalization constant. Without threads, division is
                # better done at the final return (dividing the matrices
                # rather than explicitly doing it elementwise) but seems
                # to be more efficient this way w/ threads.
                phiU[rowId, colId] = sgn * perm / normalization[row, col]
        free(U_T)
        free(U_ST)
        free(rowSums)

    return phiU
