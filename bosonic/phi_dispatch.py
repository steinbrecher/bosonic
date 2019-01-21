from __future__ import print_function, absolute_import, division

import sys
import numpy as np

from .aa_phi import aa_phi as aa_phi_cpu
from .aa_phi import binom

# Try to import cuda library & test
gpu_avail = True
try:
    from .gpu_phi import GpuPhiDispatcher
    from .util import haar_rand

    aa_phi_gpu = GpuPhiDispatcher()
    U = haar_rand(4)
    phiU_GPU = aa_phi_gpu(U, 2)
    phiU_CPU = aa_phi_cpu(U, 2)
    assert np.mean(np.abs(phiU_CPU - phiU_GPU)) < 1e-12
except:
    gpu_avail = False


class PhiDispatcher(object):
    estimatorCutoff = 1732

    def __init__(self):
        self.gpu = aa_phi_gpu
        self.cpu = aa_phi_cpu
        self.lastUsed = None

    def estimator(self, n, m):
        return binom(n+m-1, n) * n**2

    def __call__(self, U, n):
        m = U.shape[0]
        est = self.estimator(n, m)
        if est < self.estimatorCutoff:
            self.lastUsed = 'cpu'
            return self.cpu(U, n)
        else:
            self.lastUsed = 'gpu'
            return self.gpu(U, n)


if gpu_avail:
    aa_phi = PhiDispatcher()
else:
    print("Warning: bosonic could not set up GPU; falling back to CPU",
          file=sys.stderr)
    aa_phi = aa_phi_cpu
