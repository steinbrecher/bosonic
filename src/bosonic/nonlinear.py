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


def build_fock_nonlinear_layer(numPhotons, numModes, theta, lossy=False):
    """Build a kerr-nonlinear layer in the fock basis
    Inputs:
        numPhotons: number of photons
        numModes: number of optical modes
        theta: Either a single phase or an array of per-site
               phases indicating the strength of the nonlinearity

    Returns:
        Matrix corresponding to a chi-2 nonlinearity of strength theta

    The matrix is an identity transform for any state with at most one photon
    per mode. States with more than one photon per mode pick up a factor of
    e^{i*theta} for each extra photon. So |2, 1, 0, 0> picks up a factor
    of e^{i*theta} while |3, 1, 2, 0> picks up e^{3*i*theta} -- two thetas for
    the extra photons in the first mode and another theta for the extra photon
    in the third mode.
    """
    theta = np.array(theta)
    if theta.size == 1:
        if lossy:
            return build_lossy_fock_nonlinear_layer_constant(
                numPhotons, numModes, theta)
        return build_fock_nonlinear_layer_constant(numPhotons, numModes, theta)
    if theta.size == numModes:
        if lossy:
            raise NotImplementedError(
                "Lossy layer with variable nonlinearity not yet implemented")
        return build_fock_nonlinear_layer_variable(numPhotons, numModes, theta)
    else:
        raise IOError(
            "Theta must be a single number or an array of size numModes")


@memoize
def build_fock_nonlinear_layer_variable(numPhotons, numModes, theta):
    basis = fock_basis(numPhotons, numModes)
    N = len(basis)
    A = np.eye(N, dtype=complex)
    for i, state in enumerate(basis):
        # Convert state to an array
        s = np.array(state)

        # Calculate phase of the nonlinearity
        phase = 0
        for j in xrange(numModes):
            if s[j] > 1:
                phase += s[j] * (s[j]-1) * theta[j] / 2

        # Update A
        A[i, i] = expi(phase)
    return A


@memoize
def build_fock_nonlinear_layer_constant(numPhotons, numModes, theta):
    basis = fock_basis(numPhotons, numModes)
    N = len(basis)
    A = np.eye(N, dtype=complex)
    for i, state in enumerate(basis):
        # Convert state to an array
        s = np.array(state)

        # Set all modes with zero or one photons equal to zero
        s -= 1
        s[s <= 0] = 0

        # Calculate phase of the nonlinearity
        phase = np.sum(s) * theta

        # Update A
        A[i, i] = expi(phase)
    return A


@memoize
def build_lossy_fock_nonlinear_layer_constant(numPhotons, numModes, theta):
    basis = lossy_fock_basis(numPhotons, numModes)
    N = len(basis)
    A = np.eye(N, dtype=complex)
    for i, state in enumerate(basis):
        # Convert state to an array
        s = np.array(state)

        # Set all modes with zero or one photons equal to zero
        s -= 1
        s[s <= 0] = 0

        # Calculate phase of the nonlinearity
        phase = np.sum(s) * theta

        # Update A
        A[i, i] = expi(phase)
    return A


# Test that this works for two photons in four modes
_A = np.eye(10, dtype=complex)
theta = np.pi
_A[0, 0] = expi(theta)
_A[4, 4] = expi(theta)
_A[7, 7] = expi(theta)
_A[9, 9] = expi(theta)

_Atest = build_fock_nonlinear_layer(2, 4, theta)
_diff = np.sum(np.ravel(np.abs(_A - _Atest)**2))
try:
    assert _diff < 1e-16
except AssertionError as e:
    print("Error: build_fock_nonlinear_layer is broken")
    raise e
