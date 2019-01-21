from __future__ import print_function, absolute_import, division
import unittest

import bosonic as b
import numpy as np
import itertools as it
from scipy.misc import factorial


def permanent_slow(a):
    """Reference permanent calculation implementation
    No effort has been made to optimize this, deliberately.
    The permanent is defined as:
    perm(A) = \sum_{\sigma \in S_n} \prod_{i=0}^{n-1} a_{i, \sigma(i)}
    (Note the sum is zero indexed, as python is)
    """
    a = np.array(a)
    N = a.shape[0]
    Sn = it.permutations(range(N))
    out = 0
    for sigma in Sn:
        out += np.prod([a[i, si] for i, si in enumerate(sigma)])
    return out


def aa_phi_slow(U, n):
    """Reference implementation of aa_phi
    No effort has been made to optimize this, deliberately.
    """
    U = np.array(U)
    m = U.shape[0]
    # TODO: Test the functions used here
    basis = b.fock.basis(n, m)
    N = b.fock.basis_size(n, m)
    idxs = [b.fock_to_idx(np.array(S), n) for S in basis]

    # Generate the normalization coefficients for phiU
    factProducts = np.zeros((N,))
    for i, S in enumerate(basis):
        factProducts[i] = np.sqrt(np.prod([factorial(x) for x in S]))
    normalization = np.outer(factProducts, factProducts)

    # Build phiU
    U_T = np.zeros((m, n), dtype=complex)
    U_ST = np.zeros((n, n), dtype=complex)
    phiU = np.zeros((N, N), dtype=complex)
    for col in range(N):
        for i in range(m):
            for j in range(n):
                U_T[i, j] = U[i, idxs[col][j]]

        for row in range(N):
            for i in range(n):
                for j in range(n):
                    U_ST[i, j] = U_T[idxs[row][i], j]
            phiU[row, col] = permanent_slow(U_ST) / normalization[row, col]
    return phiU


class TestReferenceImplementations(unittest.TestCase):
    def test_perm2_real(self):
        a = np.random.randn(2, 2)
        permA = permanent_slow(a)
        permACorrect = a[0, 0] * a[1, 1] + a[0, 1] * a[1, 0]
        self.assertTrue(np.allclose(permA, permACorrect))

    def test_perm2_complex(self):
        a = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
        permA = permanent_slow(a)
        permACorrect = a[0, 0] * a[1, 1] + a[0, 1] * a[1, 0]
        self.assertTrue(np.allclose(permA, permACorrect))

    def test_perm3_real(self):
        a = np.random.randn(3, 3)
        permA = permanent_slow(a)
        permACorrect = 0
        permACorrect += a[0, 0] * (a[1, 1] * a[2, 2] + a[1, 2] * a[2, 1])
        permACorrect += a[0, 1] * (a[1, 0] * a[2, 2] + a[1, 2] * a[2, 0])
        permACorrect += a[0, 2] * (a[1, 0] * a[2, 1] + a[1, 1] * a[2, 0])
        self.assertTrue(np.allclose(permA, permACorrect))

    def test_perm3_complex(self):
        a = np.random.randn(3, 3)
        permA = permanent_slow(a)
        permACorrect = 0
        permACorrect += a[0, 0] * (a[1, 1] * a[2, 2] + a[1, 2] * a[2, 1])
        permACorrect += a[0, 1] * (a[1, 0] * a[2, 2] + a[1, 2] * a[2, 0])
        permACorrect += a[0, 2] * (a[1, 0] * a[2, 1] + a[1, 1] * a[2, 0])
        self.assertTrue(np.allclose(permA, permACorrect))

    def test_hom_symm(self):
        U = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
        phiU = aa_phi_slow(U, 2)
        phiUCorrect = np.array(
            [[1, 1j*np.sqrt(2), -1],
             [1j*np.sqrt(2), 0, 1j*np.sqrt(2)],
             [-1, 1j*np.sqrt(2), 1]]) / 2
        self.assertTrue(np.allclose(phiU, phiUCorrect))

    def test_hom_hadamard(self):
        U = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        phiU = aa_phi_slow(U, 2)
        phiUCorrect = np.array(
            [[1, np.sqrt(2), 1],
             [np.sqrt(2), 0, -np.sqrt(2)],
             [1, -np.sqrt(2), 1]], dtype=complex) / 2
        self.assertTrue(np.allclose(phiU, phiUCorrect))


class TestAAPhi(unittest.TestCase):
    def test_aa_phi(self):
        ms = range(1, 5)
        ns = range(1, 5)
        for m in ms:
            U = b.util.haar_rand(m)
            for n in ns:
                phiU = b.aa_phi(U, n)
                phiUCorrect = aa_phi_slow(U, n)
                self.assertTrue(np.allclose(phiU, phiUCorrect))
