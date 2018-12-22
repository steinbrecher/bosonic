from __future__ import print_function, absolute_import, division
import numpy as np
from .util import memoize


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
    M = np.eye(numModes, dtype=complex)
    M[mode:mode+2, mode:mode+2] = mzi(phis)
    return M


@memoize
def build_mzi_list(numModes, numPhotons=None):
    "gives reck MZI addresses in [diagonal, mode], in order of construction"
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


def build(phiList, numModes, numPhotons=None):
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


def rand(numModes):
    numPhis = numModes * (numModes-1)
    phiList = np.random.uniform(0, 2*np.pi, size=(numPhis, ))
    return build(phiList, numModes)
