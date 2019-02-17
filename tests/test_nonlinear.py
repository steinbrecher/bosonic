from __future__ import print_function, absolute_import, division
import unittest

import bosonic as b
import random
import numpy as np

MAX_PHOTONS = 3
MAX_MODES = 6


class TestNonlinear(unittest.TestCase):
    def test_static_layer(self):
        """Check lossless nonlinear layer with static phase"""
        phis = [random.random() * np.pi for _ in range(5)]
        # Check edge cases
        phis.append(0)
        phis.append(np.pi)
        for n in range(1, MAX_PHOTONS+1):
            for m in range(1, MAX_MODES+1):
                basis = b.fock.basis(n, m)
                N = len(basis)
                for phi in phis:
                    A = b.nonlinear.build_fock_nonlinear_layer(
                        n, m, phi, lossy=False, matrix=True)
                    AV = b.nonlinear.build_fock_nonlinear_layer(
                        n, m, phi, lossy=False, matrix=False)
                    refDiag = np.ones(A.shape[0], dtype=complex)
                    for i, elem in enumerate(basis):
                        phase = 0
                        for c in elem:
                            phase += phi * c * (c - 1) / 2
                        refDiag[i] = np.exp(1j * phase)

                    refA = np.diag(refDiag)
                    diff = np.sum(np.abs(A - refA))

                    refV = np.reshape(refDiag, (N, 1))
                    diffV = np.sum(np.abs(refV - AV))

                    self.assertAlmostEqual(diff, 0)
                    self.assertAlmostEqual(diffV, 0)
                    self.assertEqual(A.shape, (N, N))
                    self.assertEqual(AV.shape, (N, 1))

    def test_variable_layer_matrix(self):
        """Check lossless nonlinear layer matrix with variable phase"""
        for n in range(1, MAX_PHOTONS+1):
            for m in range(1, MAX_MODES+1):
                basis = b.fock.basis(n, m)
                N = len(basis)
                phis = np.random.random((m,)) * np.pi
                A = b.nonlinear.build_fock_nonlinear_layer(
                    n, m, phis, lossy=False, matrix=True)
                AV = b.nonlinear.build_fock_nonlinear_layer(
                    n, m, phis, lossy=False, matrix=False)
                refDiag = np.ones(A.shape[0], dtype=complex)
                for i, elem in enumerate(basis):
                    phase = 0
                    for j, c in enumerate(elem):
                        phase += phis[j] * c * (c - 1) / 2
                    refDiag[i] = np.exp(1j * phase)

                refA = np.diag(refDiag)
                diff = np.sum(np.abs(A - refA))

                refV = np.reshape(refDiag, (N, 1))
                diffV = np.sum(np.abs(refV - AV))

                self.assertAlmostEqual(diff, 0)
                self.assertAlmostEqual(diffV, 0)
                self.assertEqual(A.shape, (N, N))
                self.assertEqual(AV.shape, (N, 1))

    def test_lossy_static_layer_matrix(self):
        """Check lossy nonlinear layer matrix with static phase"""
        phis = [random.random() * np.pi for _ in range(5)]
        # Check edge cases
        phis.append(0)
        phis.append(np.pi)
        for n in range(1, MAX_PHOTONS+1):
            for m in range(1, MAX_MODES+1):
                basis = b.fock.lossy_basis(n, m)
                N = len(basis)
                for phi in phis:
                    A = b.nonlinear.build_fock_nonlinear_layer(
                        n, m, phi, lossy=True, matrix=True)
                    AV = b.nonlinear.build_fock_nonlinear_layer(
                        n, m, phi, lossy=True, matrix=False)
                    refDiag = np.ones(A.shape[0], dtype=complex)
                    for i, elem in enumerate(basis):
                        phase = 0
                        for c in elem:
                            phase += phi * c * (c - 1) / 2
                        refDiag[i] = np.exp(1j * phase)

                    refA = np.diag(refDiag)
                    diff = np.sum(np.abs(A - refA))

                    refV = np.reshape(refDiag, (N, 1))
                    diffV = np.sum(np.abs(refV - AV))

                    self.assertAlmostEqual(diff, 0)
                    self.assertAlmostEqual(diffV, 0)
                    self.assertEqual(A.shape, (N, N))
                    self.assertEqual(AV.shape, (N, 1))

    def test_lossy_variable_layer_matrix(self):
        """Check lossy nonlinear layer matrix with variable phase"""
        for n in range(1, MAX_PHOTONS+1):
            for m in range(1, MAX_MODES+1):
                basis = b.fock.lossy_basis(n, m)
                N = len(basis)
                phis = np.random.random((m,)) * np.pi
                A = b.nonlinear.build_fock_nonlinear_layer(
                    n, m, phis, lossy=True, matrix=True)
                AV = b.nonlinear.build_fock_nonlinear_layer(
                    n, m, phis, lossy=True, matrix=False)
                refDiag = np.ones(A.shape[0], dtype=complex)
                for i, elem in enumerate(basis):
                    phase = 0
                    for j, c in enumerate(elem):
                        phase += phis[j] * c * (c - 1) / 2
                    refDiag[i] = np.exp(1j * phase)

                refA = np.diag(refDiag)
                diff = np.sum(np.abs(A - refA))

                refV = np.reshape(refDiag, (N, 1))
                diffV = np.sum(np.abs(refV - AV))

                self.assertAlmostEqual(diff, 0)
                self.assertAlmostEqual(diffV, 0)
                self.assertEqual(A.shape, (N, N))
                self.assertEqual(AV.shape, (N, 1))

    def test_static_nonlinear_vector_vs_matrix(self):
        """Check multiplying by nonlinear vec is same as matrix"""
        phis = [random.random() * np.pi for _ in range(5)]
        # Check edge cases
        phis.append(0)
        phis.append(np.pi)
        for n in range(1, MAX_PHOTONS+1):
            for m in range(1, MAX_MODES+1):
                N = b.fock.basis_size(n, m)
                NL = b.fock.lossy_basis_size(n, m)
                for phi in phis:
                    U = b.util.haar_rand(N)
                    UL = b.util.haar_rand(NL)

                    A = b.nonlinear.build_fock_nonlinear_layer(
                        n, m, phi, lossy=False, matrix=True)
                    AV = b.nonlinear.build_fock_nonlinear_layer(
                        n, m, phi, lossy=False, matrix=False)

                    AL = b.nonlinear.build_fock_nonlinear_layer(
                        n, m, phi, lossy=True, matrix=True)
                    AVL = b.nonlinear.build_fock_nonlinear_layer(
                        n, m, phi, lossy=True, matrix=False)

                    AU = np.dot(A, U)
                    AVU = np.multiply(AV, U)
                    diff = np.sum(np.abs(AU - AVU))

                    ALUL = np.dot(AL, UL)
                    AVLUL = np.multiply(AVL, UL)
                    diff = np.sum(np.abs(ALUL - AVLUL))

                    self.assertAlmostEqual(diff, 0)

    def test_variable_nonlinear_vector_vs_matrix(self):
        """Check multiplying by variable nonlinear vector is same as matrix"""
        for n in range(1, MAX_PHOTONS+1):
            for m in range(1, MAX_MODES+1):
                N = b.fock.basis_size(n, m)
                NL = b.fock.lossy_basis_size(n, m)

                phis = np.random.random((m,)) * np.pi

                U = b.util.haar_rand(N)
                UL = b.util.haar_rand(NL)

                A = b.nonlinear.build_fock_nonlinear_layer(
                    n, m, phis, lossy=False, matrix=True)
                AV = b.nonlinear.build_fock_nonlinear_layer(
                    n, m, phis, lossy=False, matrix=False)

                AL = b.nonlinear.build_fock_nonlinear_layer(
                    n, m, phis, lossy=True, matrix=True)
                AVL = b.nonlinear.build_fock_nonlinear_layer(
                    n, m, phis, lossy=True, matrix=False)

                AU = np.dot(A, U)
                AVU = np.multiply(AV, U)
                diff = np.sum(np.abs(AU - AVU))

                ALUL = np.dot(AL, UL)
                AVLUL = np.multiply(AVL, UL)
                diff = np.sum(np.abs(ALUL - AVLUL))

                self.assertAlmostEqual(diff, 0)

    def test_nonlinear_two_photons_four_modes(self):
        """Spot check nonlinear layer for two photons in four modes"""
        phis = [random.random() * np.pi for _ in range(10)]
        # Edge cases
        phis.append(0)
        phis.append(np.pi)
        for phi in phis:
            refA = np.ones((10,), dtype=complex)
            refA[0] = np.exp(1j * phi)
            refA[4] = np.exp(1j * phi)
            refA[7] = np.exp(1j * phi)
            refA[9] = np.exp(1j * phi)
            A = b.nonlinear.build_fock_nonlinear_layer(2, 4, phi)
            diff = np.sum(np.abs(refA - np.diag(A)))
            self.assertAlmostEqual(diff, 0)
            self.assertEqual(A.shape, (10, 10))

    def test_wrong_size_phase_vec(self):
        """Make sure error is thrown for wrong theta vector"""
        # Run three random n/m pairs
        for _ in range(3):
            n = random.randint(1, 4)
            m = random.randint(1, 8)

            # Supply too many thetas
            thetas = np.random.random((m+1,)) * np.pi
            with self.assertRaises(ValueError):
                b.nonlinear.build_fock_nonlinear_layer(
                    n, m, thetas, lossy=False, matrix=False)
            with self.assertRaises(ValueError):
                b.nonlinear.build_fock_nonlinear_layer(
                    n, m, thetas, lossy=False, matrix=True)
            with self.assertRaises(ValueError):
                b.nonlinear.build_fock_nonlinear_layer(
                    n, m, thetas, lossy=True, matrix=False)
            with self.assertRaises(ValueError):
                b.nonlinear.build_fock_nonlinear_layer(
                    n, m, thetas, lossy=True, matrix=True)

            # Supply zero thetas
            thetas = np.array([])
            with self.assertRaises(ValueError):
                b.nonlinear.build_fock_nonlinear_layer(
                    n, m, thetas, lossy=False, matrix=False)
            with self.assertRaises(ValueError):
                b.nonlinear.build_fock_nonlinear_layer(
                    n, m, thetas, lossy=False, matrix=True)
            with self.assertRaises(ValueError):
                b.nonlinear.build_fock_nonlinear_layer(
                    n, m, thetas, lossy=True, matrix=False)
            with self.assertRaises(ValueError):
                b.nonlinear.build_fock_nonlinear_layer(
                    n, m, thetas, lossy=True, matrix=True)
