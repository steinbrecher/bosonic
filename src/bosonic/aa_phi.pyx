from __future__ import print_function, absolute_import, division
from cython.parallel cimport prange, parallel

import numpy as np
from libc.stdlib cimport abort, malloc, free
from libc.string cimport memset
from .util import memoize
from .fock import basis as fock_basis
from .fock import basis_array, basis_size, lossy_basis_size, factorial
from autograd.extend import primitive, defvjp

# Needed for compile-time information about numpy
cimport numpy as np
cimport cython


def fock_to_idx(np.ndarray[np.int_t, ndim=1] S, int n):
    """Converts fock state S to list with s_i copies of the number i
    e.g. state [0,2,1,0]->[1,1,2]
    """
    assert n == sum(S)
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


@memoize
def build_norm_and_idxs(size_t n, size_t m):
    """Helper function for aa_phi to build normalization array and index slices

    The entry of phi(U) corresponding to output S=[s_1,...,s_m] and input 
    T=[t_1,...,t_m] is perm(U_ST) / sqrt(s_1! * ... * s_m! * t_1 * ... * t_m!). 

    This function computes both the necessary index slices to convert U to U_ST 
    and computes the value of the sqrt normalization for each S,T pair.

    Parameters
    ----------
    n : size_t
        Number of photons
    m : size_t
        Number of modes

    Returns
    -------
    normalization : np.ndarray[np.double_t, ndim = 2]
                    Normalization coefficients for each entry in phi(U)
    idxs : np.ndarray[np.int_t, ndim = 2]
           Index slices for reducing U to U_T and U_T to U_ST

    See Also
    --------
    aa_phi : Function that computes phi(U), which this is a helper function for

    Notes
    -----
    Function is memoized
    """

    cdef int N
    cdef size_t i
    cdef np.ndarray basis
    cdef np.ndarray[np.double_t, ndim= 1] factProducts
    cdef np.ndarray[np.double_t, ndim= 2] normalization
    cdef np.ndarray[np.int_t, ndim= 2] idxs

    N = basis_size(n, m)
    basis = basis_array(n, m)
    factProducts = np.zeros([N, ], dtype=np.double)
    normalization = np.zeros([N, N], dtype=np.double)
    idxs = np.zeros([N, n], dtype=np.int)

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


@memoize
def build_kIdxs(int n):
    """Helper function for permanent calculation

    The way we compute permanents here is by iterating over the permutations
    in gray code order. Iterating over the permutations in this way reduces
    the asymptotic complexity of Ryser's algorithm from O(2^{n-1} n^2) to
    O(2^{n-1} n) for an n x n matrix. 

    Gray codes are orderings of the integers {0,...,N} such that each number 
    has only one bit flipped from the previous number in the sequence. This 
    function computes /which/ bit to flip at each step, and whether that flip 
    was from 0->1 or 1->0. 

    Parameters
    ----------
    n : int
        Dimension of matrix

    Returns
    -------
    kIdxs : np.ndarray[np.int_t, ndim= 1]
            Index of which bit is flipped for each of the gray code entries
    kSgns : np.ndarray[np.int_t, ndim= 1] kSgns
            Sign of the flip; -1 if bit was flipped 1->0, 1 otherwise

    Notes
    -----
    Function is memoized
    """
    cdef np.ndarray[np.int_t, ndim = 1] kIdxs
    cdef np.ndarray[np.int_t, ndim = 1] kSgns
    cdef int k, gray, lastGray, deltaGray, count, kIdx, kSgn

    kIdxs = np.empty([2**n, ], dtype=np.int)
    kSgns = np.empty([2**n, ], dtype=np.int)

    # Construct lookup tables for which bit in the gray code flipped from
    # the previous k and whether it was a 0->1 transition or a 1-> 0
    # transition.
    kIdxs[0] = 0
    kSgns[0] = 1

    # Since the function is memoized, the parallel computation here is
    # probably unnecessary complexity. That said, it works, so there's no
    # reason to make it slower at this point
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
    return (kIdxs, kSgns)


@primitive
# Cython implementation of permanent calculation
@cython.boundscheck(False)  # turn off bounds-checking for entire function
# turn off negative index wrapping for entire function
@cython.wraparound(False)
def permanent(np.ndarray[np.complex128_t, ndim=2] a):
    """Compute the permanent of the complex matrix a

    Parameters
    ----------
    a : np.ndarray[np.complex128_t, ndim=2]
        2D square matrix

    Returns
    -------
    Perm(a) : complex

    """
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


@cython.boundscheck(False)
@cython.wraparound(False)
def minor(np.ndarray[np.complex128_t, ndim=2] a, size_t row, size_t col):
    """Build the (x, y) minor of matrix a

    Parameters
    ----------
    a : np.ndarray[np.complex128_t, ndim=2]
        Input matrix
    row : size_t
          Row index of the minor
    col : size_t
          Column index of the minor

    Returns
    -------
    b : np.ndarray[np.complex128_t, ndim=2]
        a with specified row and column removed
    """
    cdef size_t n
    cdef np.ndarray[np.complex128_t, ndim = 2] b
    cdef size_t i, j, ii, jj

    n = a.shape[0]
    b = np.zeros((n-1, n-1), dtype=np.complex128)
    ii = 0
    jj = 0

    # Iterate over the rows and columns of a, walking the pointers ii and jj
    # with us, except when i == row or j == col; in those cases i or j
    # increase, but ii or jj do not
    for i in range(n):
        if i == row:
            continue
        jj = 0
        for j in range(n):
            if j == col:
                continue
            b[ii, jj] = a[i, j]
            jj += 1
        ii += 1
    return b


@cython.boundscheck(False)
@cython.wraparound(False)
def permanent_vjp(complex ans, np.ndarray[np.complex128_t, ndim=2] x):
    """Efficient computation of the vector jacobian product for the permanent

    The derivative of Perm(a) w/r/t a_{i,j} is equal to the permanent of the
    (i, j) minor of a. 
    """
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


defvjp(permanent, permanent_vjp)


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


@primitive
@cython.boundscheck(False)
@cython.wraparound(False)
def aa_phi(np.ndarray[np.complex128_t, ndim=2] U, size_t n):
    """Computes multi-particle unitary for a given number of photons

    This implementation computes the elements of phi(U) in parallel, using the 
    O(2^{n-1} n) implementation of Ryser's algorithm to compute the various 
    matrix permanents. 

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
    faint of heart; it goes as :math:`O(Choose[n+m-1,n]^2 * n * 2^n)`, 
    where `m` is dimensionality of `U` and `n` is the number of photons.

    References
    ----------
    [1] Aaronson, Scott, and Alex Arkhipov. "The computational
    complexity of linear optics." In Proceedings of the forty-third
    annual ACM symposium on Theory of computing, pp. 333-342. ACM, 2011.
    """

    cdef size_t N, m

    cdef size_t row, col, i, j, I, J

    cdef int k, sgn
    cdef complex rowsumprod, perm

    cdef np.ndarray[np.int_t, ndim = 2] idxs
    cdef np.ndarray[np.double_t, ndim = 2] normalization

    cdef np.ndarray[np.complex128_t, ndim = 2] phiU

    cdef np.ndarray[np.int_t, ndim= 1] kIdxs
    cdef np.ndarray[np.int_t, ndim= 1] kSgns
    cdef int kIdx, kSgn

    cdef complex * U_T
    cdef complex * U_ST
    cdef complex * rowsums

    m = U.shape[0]  # Number of optical modes
    N = basis_size(n, m)  # Size of the Fock basis
    phiU = np.empty([N, N], dtype=np.complex128)

    # Get the indexes of which element of the gray code changes each
    # iteration. Wrapped into a separate function for memoization, since
    # these values don't change for a given n
    kIdxs, kSgns = build_kIdxs(n)

    # Get normalization matrix for phi(U) and indexes of which elements
    # of U map to elements of U_T and U_ST (see function for more
    # details)
    normalization, idxs = build_norm_and_idxs(n, m)

    # If n is odd, flip the sign of the permanents.
    if (n % 2) == 1:
        sgn = -1
    else:
        sgn = 1

    with nogil, parallel(num_threads=12):
        # Note: malloc'd 2d arrays need to be accessed as a 1d array.
        # C is row major, so access pattern is arr[row * numCol + col]
        U_T = < complex * >malloc(sizeof(complex) * m * n)
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


@cython.boundscheck(False)
@cython.wraparound(False)
def aa_phi_vjp_fast(np.ndarray[np.complex128_t, ndim=2] g,
                    np.ndarray[np.complex128_t, ndim=2] a,
                    size_t n, size_t m):
    """Compute the vector Jacobian product of aa_phi

    We know that the S,T entry of phi(U) is perm(U_ST)/norm[S,T], so the
    derivative of this entry, with respect to an entry of U, is the derivative
    of the permanent with respect to the corresponding entry of U_ST. 

    This function computes each of those partials, back-propagating their effect
    from g to the matrix out. 
    """
    cdef np.ndarray[np.complex128_t, ndim = 2] out

    cdef complex J
    cdef size_t N
    cdef size_t minorN
    cdef np.ndarray[np.complex128_t, ndim= 4] Js
    cdef np.ndarray[np.int_t, ndim = 5] ustIdxs
    cdef np.ndarray[np.double_t, ndim= 2] normalization
    cdef np.ndarray[np.int_t, ndim= 2] idxs
    cdef np.ndarray[np.int_t, ndim= 1] kIdxs
    cdef np.ndarray[np.int_t, ndim= 1] kSgns

    cdef int kIdx
    cdef int kSgn

    N = basis_size(n, m)
    minorN = n - 1
    out = np.zeros((m, m), dtype=np.complex128)
    Js = np.zeros((N, N, n, n), dtype=complex)
    ustIdxs = build_ust_idxs(n, m)

    # Get the indexes of which element of the gray code changes each
    # iteration. Only run this once per n, so it's wrapped into its own
    # function with a dictionary memoization.
    kIdxs, kSgns = build_kIdxs(minorN)

    # Get normalization matrix for phi(U) and indexes of which elements
    # of U map to elements of U_T and U_ST (see function for more
    # details)
    normalization, idxs = build_norm_and_idxs(n, m)

    cdef size_t row, col, i, j, x, y, ii, jj
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
        U_ST = <complex*>malloc(sizeof(complex) * n * n)
        if U_ST == NULL:
            abort()

        minorU = <complex*>malloc(sizeof(complex) * (n-1) * (n-1))
        if minorU == NULL:
            abort()

        rowSums = <complex*>malloc(sizeof(complex) * minorN)
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
                                rowSums[i] = rowSums[i] + kSgn * \
                                    minorU[minorN * kIdx + i]

                            # Set rowsumprod to 1 for even k, -1 for odd k
                            rowsumprod = 1 - 2 * (k % 2)

                            # Compute product over the rowsums
                            for i in range(minorN):
                                rowsumprod = rowsumprod * rowSums[i]

                            # Collect the permanent sum
                            perm = perm + rowsumprod
                        Js[row, col, x, y] = g[row, col] * \
                            sgn * perm / normalization[row, col]

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


@cython.boundscheck(False)
@cython.wraparound(False)
def aa_phi_vjp(np.ndarray[np.complex128_t, ndim=2] ans, a, size_t n):
    """aa_phi_vjp_fast wrapper for autograd vjp interface"""
    cdef size_t m = a.shape[0]

    def vjp(np.ndarray[np.complex128_t, ndim=2] g):
        return aa_phi_vjp_fast(g, a, n, m)
    return vjp


defvjp(aa_phi, aa_phi_vjp, None)


@cython.boundscheck(False)
@cython.wraparound(False)
def aa_phi_restricted(np.ndarray[np.complex128_t, ndim=2] U, size_t n,
                      np.ndarray[np.int_t, ndim=1] idxIn,
                      np.ndarray[np.int_t, ndim=1] idxOut):
    """Compute restricted multi-particle unitary for only some Fock states

    Parameters
    ----------
    U : np.ndarray[np.complex128_t, ndim=2]
        Single-photon unitary to be transformed
    n : size_t
        Number of photons to compute unitary for
    idxIn : np.ndarray[np.complex128_t, ndim=1]
            Indices of input states
    idxOut : np.ndarray[np.complex128_t, ndim=1]
             Indices of output states

    Returns
    -------
    array of complex128
        Multiparticle unitary over the fock basis. 

    See Also
    --------
    aa_phi : Computation of the unrestricted multi-particle unitary

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


@cython.boundscheck(False)
@cython.wraparound(False)
def aa_phi_lossy(np.ndarray[np.complex128_t, ndim=2] U, size_t n):
    """Compute the block-diagonal multi particle unitary over the lossy basis
    """
    cdef size_t m = U.shape[0]
    cdef size_t N = lossy_basis_size(n, m)
    cdef np.ndarray[np.complex128_t, ndim = 2] S = np.eye(N, dtype = np.complex128)

    cdef size_t nn = n
    cdef size_t count = 0
    cdef size_t NN

    while nn > 0:
        NN = basis_size(nn, m)
        phiU = aa_phi(U, nn)
        S[count:count+NN, count:count+NN] = phiU
        nn -= 1
        count += NN
    return S
