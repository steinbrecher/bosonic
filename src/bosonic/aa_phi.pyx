from __future__ import division

import numpy as np
from libc.stdlib cimport abort, malloc, free
from libc.string cimport memset
from .bosonic_util import memoize
from .fock import fock_basis, basis_size, lossy_basis_size, factorial

# Needed for compile-time information about numpy
cimport numpy as np

# Memoized function to build the basis efficiently
# Note: basis is a numpy array here, not a list of lists as fock_basis
# returns
@memoize
def build_basis(int n, int m):
    cdef np.ndarray[np.int_t, ndim=2] basis = np.array(fock_basis(n, m), dtype=np.int)
    return basis

def fock_to_idx(np.ndarray[np.int_t, ndim=1] S, int n):
    """Converts fock state S to list with s_i copies of the number i
    i.e. state [0,2,1,0]->[1,1,2]
    """
    cdef np.ndarray idx = np.zeros([n,], dtype=np.int)
    cdef int s
    cdef int count = 0
    for i in range(S.shape[0]):
        s = S[i]
        if s==0:
            continue
        for j in range(s):
            idx[count] = i
            count += 1
    return idx

# This stuff only gets run once per basis (dim of U and # of photons)
# so may as well cache all of it
@memoize
def build_norm_and_idxs(int n, int m):
    cdef np.ndarray basis = build_basis(n, m)
    cdef int N = basis_size(n, m)
    cdef np.ndarray[np.double_t, ndim=1] factProducts = np.zeros([N, ], dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2] normalization = np.zeros([N,N], dtype=np.double)
    cdef np.ndarray[np.int_t, ndim=2] idxs = np.zeros([N, n], dtype=np.int)
    
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
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def aa_phi3(np.ndarray[np.complex128_t, ndim=2] U, int n):
    assert U.dtype == np.complex128
    cdef int m = U.shape[0]
    cdef int N = basis_size(n, m)
    cdef int i
    cdef int j
    cdef int I
    cdef int J
    cdef np.ndarray[np.complex128_t, ndim=2] phiU = np.empty([N, N], dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=2] U_T = np.empty([m, n], dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=2] U_ST = np.empty([n, n], dtype=np.complex128)
    cdef np.ndarray[np.int_t, ndim=2] idxs
    cdef np.ndarray[np.double_t, ndim=2] normalization

    normalization, idxs = build_norm_and_idxs(n, m)

    for col in range(N):
        for j in range(n):
            J = idxs[col,j]
            for i in range(m):
                U_T[i,j] = U[i, J]

        for row in range(N):
            for i in range(n):
                I = idxs[row,i]
                for j in range(n):
                    U_ST[i,j] = U_T[I, j]

            phiU[row, col] = permanent(U_ST) 
    return phiU / normalization

from cython.parallel cimport prange, parallel

# New, optimized version of aa_phi that uses OpenMP threads to speed calculation
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def aa_phi2(np.ndarray[np.complex128_t, ndim=2] U, size_t n):
    assert U.dtype == np.complex128
    cdef size_t m = U.shape[0]
    cdef size_t N = basis_size(n, m)

    cdef size_t row, col

    cdef size_t i, j, I, J

    cdef int gray
    cdef int k # k needs to be int, not size_t for how we calculate gray
    cdef complex rowsum = 0
    cdef complex rowsumprod
    cdef complex p = 1
    cdef int sgn = 1

    cdef complex* U_T
    cdef complex* U_ST
    
    cdef np.ndarray[np.complex128_t, ndim=2] phiU = np.empty([N, N], dtype=np.complex128)
    
    cdef np.ndarray[np.double_t, ndim=2] normalization
    cdef np.ndarray[np.int_t, ndim=2] idxs


    normalization, idxs = build_norm_and_idxs(n, m)
    # If n is odd, we flip the sign of the permanents. More efficient
    # to flip the sign of the normalizations since we have to divide by
    # it later anyway
    if (n%2) == 1:
        sgn = -1

    with nogil, parallel(num_threads=8):
        U_T = <complex *>malloc(sizeof(complex) * m * n)
        if U_T == NULL:
            abort()
        U_ST = <complex *>malloc(sizeof(complex) * n * n)
        if U_ST == NULL:
            abort()
        for col in prange(N,schedule='dynamic'):
            # Populate U_T once per column
            for j in range(n):
                J = idxs[col,j]
                for i in range(m):
                    U_T[i + j*m] = U[i,J]

            for row in range(N):
                # Populate U_ST for each row
                for i in range(n):
                    I = idxs[row,i]
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
    return  phiU / normalization

kIdxLookup = {}
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def build_kIdxs(int n):
    try:
        return kIdxLookup[n]
    except KeyError:
        pass

    cdef np.ndarray[np.int_t, ndim=1] kIdxs = np.empty([2**n,], dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] kSgns = np.empty([2**n,], dtype=np.int)
    cdef int k, gray, lastGray, deltaGray, count, kIdx, kSgn

    # Construct lookup tables for which bit in the gray code flipped from
    # the previous k and whether it was a 0->1 transition or a 1-> 0
    # transition. 
    kIdxs[0] = 0
    kSgns[0] = 1
    with nogil,parallel():
        for k in prange(1, 2**n,schedule='dynamic'):
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
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def permanent(np.ndarray[np.complex128_t, ndim=2] a):
    cdef int n = a.shape[0]
    cdef np.ndarray[np.complex128_t, ndim=2] partials = np.empty([n, n], dtype=np.complex128)
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
                    rowsum += a[i,j]
            rowsumprod *= rowsum
        p += rowsumprod
    if n % 2 == 1:
        p *= -1
    return p

# Calculate the block-diagonal version of phi over the lossy basis
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def aa_phi_lossy(np.ndarray[np.complex128_t, ndim=2] U, size_t n):
    assert U.dtype == np.complex128
    cdef size_t m = U.shape[0]
    cdef size_t N = lossy_basis_size(n,m)
    cdef np.ndarray[np.complex128_t, ndim=2] S = np.eye(N, dtype=np.complex128)

    cdef size_t nn = n
    cdef size_t count = 0
    cdef size_t NN
   
    while nn > 0:
        NN = basis_size(nn, m)
        phiU = aa_phi(U,nn)
        S[count:count+NN, count:count+NN] = phiU
        nn -= 1
        count += NN
    return S

# Improvement of the threaded version of aa_phi to take advantage of
# iterating in gray code order (i.e. the rowsums only change by one
# element of U_ST for each k, so we can save them and update accordingly)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def aa_phi(np.ndarray[np.complex128_t, ndim=2] U, size_t n):
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
    cdef size_t m = U.shape[0]
    cdef size_t N = basis_size(n, m)

    cdef size_t row, col

    cdef size_t i, j, I, J

    cdef int k 
    cdef int sgn = 1
    cdef complex rowsum = 0
    cdef complex rowsumprod
    cdef complex perm = 1

    cdef complex* U_T
    cdef complex* U_ST
    cdef complex* rowsums
    
    cdef np.ndarray[np.complex128_t, ndim=2] phiU = np.empty([N, N], dtype=np.complex128)
    
    cdef np.ndarray[np.double_t, ndim=2] normalization
    cdef np.ndarray[np.int_t, ndim=2] idxs
    
    cdef np.ndarray[np.int_t, ndim=1] kIdxs 
    cdef np.ndarray[np.int_t, ndim=1] kSgns 
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
    if (n%2) == 1:
        sgn = -1

    with nogil,parallel(num_threads=16):
        # Note: malloc'd 2d arrays need to be accessed as a 1d array. 
        U_T = <complex *>malloc(sizeof(complex) * m * n)
        if U_T == NULL:
            abort()
        U_ST = <complex *>malloc(sizeof(complex) * n * n)
        if U_ST == NULL:
            abort()
        rowSums = <complex*>malloc(sizeof(complex) * n)
        if rowSums == NULL:
            abort()
            
        for col in prange(N,schedule='dynamic'):
            # Populate U_T once per column
            for j in range(n):
                J = idxs[col,j]
                for i in range(m):
                    U_T[i + j*m] = U[i,J]

            for row in range(N):
                # Populate U_ST for each row
                for i in range(n):
                    I = idxs[row,i]
                    for j in range(n):
                        U_ST[i + j*n] = U_T[I + j*m]

                # Calculate permanent of U_ST
                perm = 0
                
                # Initialize the rowSums to 0 for the permanent
                memset(rowSums, 0, n * sizeof(complex))

                # Iterate over all the set combinations (whee exponential algorithms)
                # Don't have to start at 0 since gray(0) = 0, i.e. there
                # are no bits set and thus no elements included.
                for k in range(1,2**n):
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
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
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

    cdef complex* U_T
    cdef complex* U_ST
    cdef complex* rowsums

    cdef int numIn = idxIn.size
    cdef int numOut = idxOut.size
    
    cdef np.ndarray[np.complex128_t, ndim=2] phiU = np.empty([numOut, numIn], dtype=np.complex128)
    
    cdef np.ndarray[np.double_t, ndim=2] normalization
    cdef np.ndarray[np.int_t, ndim=2] idxs
    
    cdef np.ndarray[np.int_t, ndim=1] kIdxs 
    cdef np.ndarray[np.int_t, ndim=1] kSgns 
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
    if (n%2) == 1:
        sgn = -1

    with nogil,parallel(num_threads=12):
        # Note: malloc'd 2d arrays need to be accessed as a 1d array. 
        U_T = <complex *>malloc(sizeof(complex) * m * n)
        if U_T == NULL:
            abort()
        U_ST = <complex *>malloc(sizeof(complex) * n * n)
        if U_ST == NULL:
            abort()
        rowSums = <complex*>malloc(sizeof(complex) * n)
        if rowSums == NULL:
            abort()
            
        for colId in prange(numIn, schedule='dynamic'):
            col = idxIn[colId]
            # Populate U_T once per column
            for j in range(n):
                J = idxs[col,j]
                for i in range(m):
                    U_T[i + j*m] = U[i,J]

            for rowId in range(numOut):
                row = idxOut[rowId]
                # Populate U_ST for each row
                for i in range(n):
                    I = idxs[row,i]
                    for j in range(n):
                        U_ST[i + j*n] = U_T[I + j*m]

                # Calculate permanent of U_ST
                perm = 0
                
                # Initialize the rowSums to 0 for the permanent
                memset(rowSums, 0, n * sizeof(complex))

                # Iterate over all the set combinations (whee exponential algorithms)
                # Don't have to start at 0 since gray(0) = 0, i.e. there
                # are no bits set and thus no elements included.
                for k in range(1,2**n):
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

