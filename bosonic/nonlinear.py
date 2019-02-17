from __future__ import print_function, absolute_import, division

import numpy as np
from .fock import basis as fock_basis
from .fock import lossy_basis as lossy_fock_basis
from .util import memoize


def expi(x):
    if x == np.pi:
        return complex(-1)
    if x == 0:
        return complex(1)
    return np.exp(1j * x)


def build_fock_nonlinear_layer(numPhotons, numModes, theta, lossy=False,
                               matrix=True):
    """Build a kerr-nonlinear layer in the fock basis
    Inputs:
        numPhotons: number of photons
        numModes: number of optical modes
        theta: Either a single phase or an array of per-site
               phases indicating the strength of the nonlinearity
        lossy: If true, returns nonlinear layer over full lossy basis
        matrix: If true, returns NxN diagonal matrix. If false, returns
                diagonal of that matrix in an Nx1 vector.

    Returns:
        Matrix or vector as described above

    The matrix is an identity transform for any state with at most one photon
    per mode. States with more than one photon per mode pick up a factor of
    e^{i*theta} for each extra photon. So |2, 1, 0, 0> picks up a factor
    of e^{i*theta} while |3, 1, 2, 0> picks up e^{3*i*theta} -- two thetas for
    the extra photons in the first mode and another theta for the extra photon
    in the third mode.
    """
    thetaArr = np.array(theta)
    if thetaArr.size == 1:
        return build_fock_nonlinear_layer_constant(
            numPhotons, numModes, theta, lossy=lossy, matrix=matrix)
    if thetaArr.size == numModes:
        return build_fock_nonlinear_layer_variable(
            numPhotons, numModes, theta, lossy=lossy, matrix=matrix)
    else:
        raise ValueError(
            "Theta must be a single number or an array of size numModes")


# @memoize
def build_fock_nonlinear_layer_variable(numPhotons, numModes, theta,
                                        lossy=False, matrix=True):
    if lossy:
        basis = lossy_fock_basis(numPhotons, numModes)
    else:
        basis = fock_basis(numPhotons, numModes)

    N = len(basis)
    A = np.ones((N, 1), dtype=complex)
    for i, state in enumerate(basis):
        # Calculate phase of the nonlinearity
        phase = 0
        for j in xrange(numModes):
            if state[j] > 1:
                phase += state[j] * (state[j]-1) * theta[j] / 2

        # Update A
        A[i] = expi(phase)
    if matrix:
        return np.diag(A[:, 0])
    return A


# @memoize
def build_fock_nonlinear_layer_constant(numPhotons, numModes, theta,
                                        lossy=False, matrix=True):
    if lossy:
        basis = lossy_fock_basis(numPhotons, numModes)
    else:
        basis = fock_basis(numPhotons, numModes)

    N = len(basis)
    A = np.ones((N, 1), dtype=complex)
    for i, state in enumerate(basis):
        # Set all modes with zero or one photons equal to zero
        phase = 0
        for j in xrange(numModes):
            if state[j] > 1:
                phase += state[j] * (state[j]-1) * theta / 2

        # Update A
        A[i] = expi(phase)
    if matrix:
        return np.diag(A[:, 0])
    return A
