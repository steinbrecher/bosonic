from __future__ import print_function, absolute_import, division
import autograd.numpy as np
from autograd.extend import primitive, defvjp
from ..clements import build_bs_layer
from ..clements import build as clements_build_fast
from ..fock import basis_size
from ..fock import basis as fock_basis
from ..nonlinear import build_fock_nonlinear_layer as fast_nonlin
from .. import fock_to_idx
from .. import aa_phi
from ..util import memoize


@memoize
def fock_basis_array(n, m):
    """Only convert fock_basis to an array once"""
    return np.array(fock_basis(n, m))


def phase_two(theta1, theta2=0):
    """Generate two mode phase screen"""
    return np.array([[np.exp(1j*theta1), 0],
                     [0, np.exp(1j*theta2)]])


def mzi(phis):
    """Generate 2x2 unitary transform for a backwards MZI
    phis: 2-tuple of form (inner, outer)
    """
    BS = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    return reduce(np.dot, [BS, phase_two(phis[0]), BS, phase_two(phis[1])])


def embedded_mzi(mode, phis, numModes):
    """Generate identity matrix with mzi embedded
    mzi(phis) acts on modes (mode, mode+1)
    Returned transform is of dimension (numModes x numModes)
    """
    M = [[0+0j for i in range(numModes)] for j in range(numModes)]
    for i in range(numModes):
        M[i][i] = 1
    m = mzi(phis)
    M[mode][mode] = m[0, 0]
    M[mode][mode+1] = m[0, 1]
    M[mode+1][mode] = m[1, 0]
    M[mode+1][mode+1] = m[1, 1]
    return np.array(M)


@memoize
def build_mzi_list(numModes, numPhotons=None):
    """gives reck MZI addresses in [diagonal, mode], in order of 
    construction
    """
    if numPhotons is None:
        ls = []
        for j in range(numModes-1):
            lsloc = []
            for i in range(j, numModes-1):
                lsloc.append((j, i))
            lsloc = list(lsloc)
            ls.append(lsloc)
        ls = list(reversed(ls))
        return [item for sublist in ls for item in sublist][::-1]
    else:
        ls = []
        for j in range(0, numPhotons):
            lsloc = []
            for i in range(j, numModes-1):
                lsloc.append((j, i))
            lsloc = list(lsloc)
            ls.append(lsloc)
        ls = list(reversed(ls))
        return [item for sublist in ls for item in sublist][::-1]


def reck_build(phiList, numModes, numPhotons=None):
    """Generate a unitary using the Reck-Zeilinger encoding
    phiList: n*(n-1) phases, two for each of the n*(n-1) mzis
    Ordering is [inner1, outer1, inner2, outer2, ...]
    numModes: how many modes total in the reck encoding
    """
    "takes a phiList, will be easier to iterate over"
    ls = []
    mzi_list = build_mzi_list(numModes, numPhotons)
    for i, m in enumerate(mzi_list):
        # load phase
        phases = (phiList[2*i], phiList[2*i+1])
        mode = m[1]
        ls.append(embedded_mzi(mode, phases, numModes))
    return reduce(np.dot, ls[::-1])


@primitive
def build_phi_layer(phis, m, offset):
    d = np.ones((m, 1), dtype=complex)
    for i, j in enumerate(range(offset, m-1, 2)):
        d[j, 0] = np.exp(1j*phis[i])
    return d


def build_phi_layer_vjp(ans, phis, m, offset):
    def _build_phi_layer_vjp(g):
        out = np.zeros(phis.shape)
        for i, j in enumerate(range(offset, m-1, 2)):
            out[i] += np.real(ans[j, 0] * 1j * g[j, 0])
        return out
    return _build_phi_layer_vjp


defvjp(build_phi_layer, build_phi_layer_vjp, None, None)


def clements_build(phis, m):
    U = np.eye(m, dtype=complex)
    ptr = 0
    bss = [build_bs_layer(m, 0), build_bs_layer(m, 1)]
    for i in range(m):
        offset = i % 2
        # Phis per layer
        ppl = (m - offset) // 2
        bs = bss[offset]
        phi1 = build_phi_layer(phis[ptr:ptr+ppl], m, offset)
        phi2 = build_phi_layer(phis[ptr+ppl:ptr+2*ppl], m, offset)
        ptr += 2*ppl

        U = np.multiply(phi1, U)
        U = np.dot(bs, U)
        U = np.multiply(phi2, U)
        U = np.dot(bs, U)
    outPhases = np.reshape(np.exp(1j*phis[ptr:ptr+m]), (m, 1))
    U = np.multiply(outPhases, U)
    return U


m = 6
phases = 2 * np.pi * np.random.random(m**2)
U1 = clements_build(phases, m)
U2 = clements_build_fast(phases, m)
diff = np.sum(np.abs(U1 - U2)**2)
assert diff < 1e-16


@memoize
def build_ust_idxs(n, m):
    N = basis_size(n, m)
    basis = fock_basis(n, m)
    idxs = [fock_to_idx(np.array(S), n) for S in basis]
    ustIdxs = np.zeros((N, N, n, n, 2), dtype=int)
    U_Tidx = [[None for i in range(n)] for j in range(m)]

    for col in range(N):
        for i in range(m):
            for j in range(n):
                U_Tidx[i][j] = (i, idxs[col][j])

        for row in range(N):
            for i in range(n):
                for j in range(n):
                    ustIdxs[row, col, i, j, :] = U_Tidx[idxs[row][i]][j]
    return ustIdxs


@memoize
def build_nonlin_products(numPhotons, numModes):
    basis = fock_basis(numPhotons, numModes)
    N = len(basis)
    D = np.zeros((N,))
    for i, state in enumerate(basis):
        # Convert state to an array
        s = np.array(state)

        # Set all modes with zero or one photons equal to zero
        phase = 0
        for j in xrange(numModes):
            if s[j] > 1:
                phase += s[j] * (s[j]-1) / 2
        D[i] = phase
    return D


def build_fock_nonlinear_layerD(numPhotons, numModes, theta):
    # Update A
    D = np.exp(1j * theta * build_nonlin_products(numPhotons, numModes))
    return np.reshape(D, (D.size, 1))


@primitive
def var_nonlin_diag(thetas, n, m):
    assert len(thetas) == m
    basis = fock_basis_array(n, m)
    coeffs = basis * (basis - 1) / 2.0
    # N = basis.shape[0]
    # D = np.zeros((N, 1), dtype=complex)
    D = np.sum(np.multiply(np.reshape(thetas, (m, 1)), coeffs.T).T, axis=1)
    D = np.exp(1j * D)
    # for i in range(N):
    #     D[i, 0] = np.exp(1j * np.sum(thetas * coeffs[i, :]))
    return D


def var_nonlin_diag_vjp(ans, thetas, n, m):
    basis = fock_basis_array(n, m)
    N = basis.shape[0]
    coeffs = basis * (basis-1) / 2.0
    coeffs = np.multiply(np.reshape(ans, (N, 1)), coeffs)

    def _var_nonlin_diag_vjp(g):
        gcoeffs = np.multiply(np.reshape(g, (N, 1)), coeffs)
        out = np.sum(np.real(1j * gcoeffs), axis=0)
        return out

    return _var_nonlin_diag_vjp


defvjp(var_nonlin_diag, var_nonlin_diag_vjp, None, None)


def build_system_function(n, m, numLayers, phi=np.pi, method='clements'):
    validMethods = ['clements', 'reck', 'unitaries']
    if method not in validMethods:
        msg = "Error: method not in {}".format(validMethods)
        raise ValueError(msg)

    # Phases per layer
    if method == 'reck':
        ppl = m * (m - 1)
    elif method == 'clements':
        ppl = m * m
    numPhases = numLayers * ppl
    N = basis_size(n, m)
    info = {
        'ppl': ppl,
        'numPhases': numPhases,
        'numLayers': numLayers,
    }

    if phi is not None:
        # Save number of free parameters
        info['nX'] = numLayers * ppl

        # Build the nonlinear layer (static phi if phi is not None)
        nonlin = fast_nonlin(n, m, phi)
        nonlinD = np.diag(nonlin)[:, None]

        if method == 'clements':
            def build_system(phases):
                S = np.eye(N, dtype=complex)
                for l in range(numLayers):
                    U = clements_build(phases[ppl*l:ppl*(l+1)], m)
                    phiU = aa_phi(U, n)
                    if l < numLayers-1:
                        phiU = np.multiply(nonlinD, phiU)
                    S = np.dot(phiU, S)
                return S

        elif method == 'reck':
            def build_system(phases):
                S = np.eye(N, dtype=complex)
                for l in range(numLayers):
                    U = reck_build(phases[ppl*l:ppl*(l+1)], m)
                    phiU = aa_phi(U, n)
                    layer = np.multiply(nonlinD, phiU)
                    S = np.dot(layer, S)
                return S

        elif method == 'unitaries':
            m2 = m * m

            def build_system(x):
                S = np.eye(N, dtype=complex)
                for i in range(numLayers):
                    U = np.reshape(x[i*m2:(i+1)*m2], (m, m))
                    phiU = aa_phi(U, n)
                    S = np.dot(phiU, S)
                    S = np.dot(nonlin, S)
                return S

    else:  # No phi provided
        if method == 'clements':
            def build_system(x):
                phases = x[:numPhases]
                thetas = x[numPhases:]

                S = np.eye(N, dtype=complex)
                for l in range(numLayers):
                    U = clements_build(phases[ppl*l:ppl*(l+1)], m)
                    phiU = aa_phi(U, n)
                    if l < numLayers - 1:
                        nonlinD = var_nonlin_diag(thetas[m*l:m*(l+1)], n, m)
                        phiU = np.multiply(nonlinD, phiU)
                    S = np.dot(phiU, S)
                return S

        elif method == 'reck':
            def build_system(phases):
                nonlinD = build_fock_nonlinear_layerD(n, m, phases[-1])

                S = np.eye(N, dtype=complex)
                for l in range(numLayers):
                    U = reck_build(phases[ppl*l:ppl*(l+1)], m)
                    phiU = aa_phi(U, n)
                    layer = np.multiply(nonlinD, phiU)
                    S = np.dot(layer, S)
                return S

        elif method == 'unitaries':
            m2 = m * m

            def build_system(x):
                phi = x[-1]
                x = x[:-1]
                nonlinD = build_fock_nonlinear_layerD(n, m, phi)
                S = np.eye(N, dtype=complex)
                for i in range(numLayers):
                    U = np.reshape(x[i*m2:(i+1)*m2], (m, m))
                    phiU = aa_phi(U, n)
                    layer = np.multiply(nonlinD, phiU)
                    S = np.dot(layer, S)
                return S

    return build_system, info
