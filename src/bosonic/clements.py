from __future__ import print_function, absolute_import, division
import numpy as np
from .util import memoize

dc = np.array([[1, 1j], [1j, 1]], dtype=complex) / np.sqrt(2)


@memoize
def build_bs_layer(m, offset):
    S = np.eye(m, dtype=complex)
    for i in range(offset, m-1, 2):
        S[i:i+2, i:i+2] = dc
    return S


def build_phi_layer(phis, m, offset):
    phis = np.array(phis)
    d = np.ones((m,), dtype=complex)
    d[offset:m-1:2] = np.exp(1j*phis)
    # d[offset+1:m:2] = np.exp(-1j*phis/2)
    return np.diag(d)


def build(phis, m):
    U = np.eye(m, dtype=complex)
    ptr = 0
    for i in range(m):
        offset = i % 2
        # Phis per layer
        ppl = (m - offset) // 2
        bs = build_bs_layer(m, offset)
        phi1 = build_phi_layer(phis[ptr:ptr+ppl], m, offset)
        phi2 = build_phi_layer(phis[ptr+ppl:ptr+2*ppl], m, offset)
        U = bs.dot(phi2).dot(bs).dot(phi1).dot(U)
        ptr += 2*ppl
    assert ptr == len(phis)
    return U


def build_phi_layer_asymm(phis, m, offset):
    phis = np.array(phis)
    d = np.ones((m,), dtype=complex)
    d[offset:m-1:2] = np.exp(1j*phis/2)
    d[offset+1:m:2] = np.exp(-1j*phis/2)
    return np.diag(d)


def build_phi_layer_symm(phis, m, offset):
    phis = np.array(phis)
    d = np.ones((m,), dtype=complex)
    d[offset:m-1:2] = np.exp(1j*phis)
    d[offset+1:m:2] = np.exp(1j*phis)
    return np.diag(d)


def build2(phis, m):
    U = np.eye(m, dtype=complex)
    ptr = 0
    for i in range(m):
        offset = i % 2
        # Phis per layer
        ppl = (m - offset) // 2
        bs = build_bs_layer(m, offset)
        phi1A = build_phi_layer_asymm(phis[ptr:ptr+ppl], m, offset)
        phi1S = build_phi_layer_symm(phis[ptr+ppl:ptr+2*ppl], m, offset)
        phi1 = phi1A.dot(phi1S)

        phi2A = build_phi_layer(phis[ptr+2*ppl:ptr+3*ppl], m, offset)
        phi2S = build_phi_layer(phis[ptr+3*ppl:ptr+4*ppl], m, offset)
        phi2 = phi2A.dot(phi2S)

        U = bs.dot(phi2).dot(bs).dot(phi1).dot(U)
        ptr += 4*ppl
    assert ptr == len(phis)
    return U
