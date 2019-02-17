from __future__ import print_function, absolute_import, division
import unittest

import bosonic as b
import numpy as np
from scipy.special import factorial, binom


MAX_PHOTONS = 5
MAX_MODES = 10


class TestMath(unittest.TestCase):
    def test_factorial(self):
        """Test the custom factorial implementation up to 15"""
        for x in range(16):
            f1 = b.fock.factorial(x)
            f2 = int(factorial(x))
            self.assertEqual(f1, f2)

    def test_binom(self):
        """Test the custom binomial implementation on random inputs"""
        low = 0
        high = 10
        nTest = 25

        ns = np.random.randint(low=low+1, high=high, size=nTest, dtype=int)
        for n in ns:
            # Test edge case ks
            for k in [0, n]:
                b1 = b.fock.binom(n, k)
                b2 = int(binom(n, k))
                self.assertEqual(b1, b2)
            # Test random ks
            ks = np.random.randint(low=0, high=n, size=nTest, dtype=int)
            for k in ks:
                b1 = b.fock.binom(n, k)
                b2 = int(binom(n, k))
                self.assertEqual(b1, b2)


class TestFockBasis(unittest.TestCase):
    def test_basis_first_and_last(self):
        """Check first and last basis elements"""
        for n in range(1, MAX_PHOTONS+1):
            for m in range(1, MAX_MODES+1):
                basis = b.fock.basis(n, m)
                self.assertEqual(basis[0][0], n)
                self.assertEqual(basis[-1][-1], n)

    def test_basis_sums(self):
        """Check that each basis element has the right number of photons"""
        for n in range(1, MAX_PHOTONS+1):
            for m in range(1, MAX_MODES+1):
                basis = b.fock.basis(n, m)
                for elem in basis:
                    self.assertEqual(sum(elem), n)

    def test_basis_unique(self):
        """Check that all elements in each basis are unique"""
        for n in range(1, MAX_PHOTONS+1):
            for m in range(1, MAX_MODES+1):
                basis = b.fock.basis(n, m)
                elements = set()
                for elem in basis:
                    et = tuple(elem)
                    self.assertFalse(et in elements)
                    elements.add(et)

    def test_basis_spot_checks(self):
        """Test fock.basis for selected random inputs

        Code that generated these (with verified implementation):
            import random
            import bosonic as b

            tests = []
            for _ in range(10):
                n = random.randint(1, 5)
                m = random.randint(1, 10)
                basis = b.fock.basis(n, m)
                i = random.randint(0, len(basis)-1)
                element = basis[i]
                tests.append((n, m, i, tuple(element)))
        """
        tests = [(5, 9, 571, (0, 2, 0, 1, 0, 1, 1, 0, 0)),
                 (5, 6, 195, (0, 1, 0, 0, 0, 4)),
                 (2, 5, 11, (0, 0, 1, 0, 1)),
                 (2, 8, 11, (0, 1, 0, 0, 1, 0, 0, 0)),
                 (1, 7, 6, (0, 0, 0, 0, 0, 0, 1)),
                 (5, 4, 13, (2, 1, 2, 0)),
                 (5, 10, 1136, (0, 1, 0, 0, 1, 0, 0, 0, 3, 0)),
                 (3, 5, 20, (0, 1, 1, 1, 0)),
                 (2, 6, 10, (0, 1, 0, 0, 0, 1)),
                 (2, 3, 0, (2, 0, 0))]

        for n, m, i, refElem in tests:
            elem = b.fock.basis(n, m)[i]
            self.assertEqual(tuple(elem), refElem)

    def test_lossy_basis(self):
        """Check that lossy_basis is the concatenation of the lossless bases"""
        for n in range(1, MAX_PHOTONS+1):
            for m in range(1, MAX_MODES+1):
                lossyBasis = b.fock.lossy_basis(n, m)
                while n >= 0:
                    refBasis = b.fock.basis(n, m)
                    for i, elem in enumerate(refBasis):
                        refElem = tuple(elem)
                        testElem = tuple(lossyBasis[i])
                        self.assertEqual(refElem, testElem)

                    # Cut off this section of the lossy basis
                    lossyBasis = lossyBasis[len(refBasis):]
                    n -= 1

class TestBasisUtilFunctions(unittest.TestCase):
    def test_basis_size(self):
        """Check that fock.basis_size returns the right size"""
        for n in range(1, MAX_PHOTONS+1):
            for m in range(1, MAX_MODES+1):
                n1 = len(b.fock.basis(n, m))
                n2 = b.fock.basis_size(n, m)
                self.assertEqual(n1, n2)

    def test_lossy_basis_size(self):
        """Check that fock.lossy_basis_size returns the right size"""
        for n in range(1, MAX_PHOTONS+1):
            for m in range(1, MAX_MODES+1):
                n1 = len(b.fock.lossy_basis(n, m))
                n2 = b.fock.lossy_basis_size(n, m)
                self.assertEqual(n1, n2)

    def test_basis_array(self):
        """Check that fock.basis_array and fock.basis match"""
        for n in range(1, MAX_PHOTONS+1):
            for m in range(1, MAX_MODES+1):
                b1 = b.fock.basis_array(n, m)
                b2 = np.array(b.fock.basis(n, m))
                diff = np.sum(np.abs(b1 - b2))
                self.assertEqual(diff, 0)

    def test_loossy_basis_array(self):
        """Check that fock.lossy_basis_array and fock.lossy_basis match"""
        for n in range(1, MAX_PHOTONS+1):
            for m in range(1, MAX_MODES+1):
                b1 = b.fock.lossy_basis_array(n, m)
                b2 = np.array(b.fock.lossy_basis(n, m))
                diff = np.sum(np.abs(b1 - b2))
                self.assertEqual(diff, 0)

    def test_basis_lookup(self):
        """Check fock.basis_lookup dictionaries"""
        for n in range(1, MAX_PHOTONS+1):
            for m in range(1, MAX_MODES+1):
                basis = b.fock.basis(n, m)
                basisLookup = b.fock.basis_lookup(n, m)
                for i, elem in enumerate(basis):
                    lookupI = basisLookup[tuple(elem)]
                    self.assertEqual(i, lookupI)

    def test_lossy_basis_lookup(self):
        """Check fock.basis_lookup dictionaries"""
        for n in range(1, MAX_PHOTONS+1):
            for m in range(1, MAX_MODES+1):
                basis = b.fock.lossy_basis(n, m)
                basisLookup = b.fock.lossy_basis_lookup(n, m)
                for i, elem in enumerate(basis):
                    lookupI = basisLookup[tuple(elem)]
                    self.assertEqual(i, lookupI)
