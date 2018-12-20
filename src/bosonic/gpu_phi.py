from __future__ import absolute_import, division
import numpy as np
import numba
from numba import complex128
import time
from .aa_phi import build_norm_and_idxs, build_kIdxs

from numba import cuda

class GpuPhiDispatcher(object):
    lastNM = (None, None)

    normalization = None
    N = None
    idxs = None
    phiU = None
    d_phiU = None
    d_idxs = None

    threadsperblock = (8,8)
    blockspergrid = None
    gpu_phi = None
    
    gpuPhis = {}
    
    lastTNorm = 0
    lastTSetup = 0

    def __init__(self, dev=0):
        self.cuda = cuda
        self.select_device(dev)
        self.stream = cuda.stream()

    def select_device(self, dev):
        self.cuda.close()
        self.cuda.select_device(dev)
    
    def __call__(self, U, n):
        m = U.shape[0]
        # If dimensionality of problem has changed, re-do setup
        if (n,m) != self.lastNM:
            t1 = time.time()
            self.lastNM = (n,m)

            normalization, self.idxs = build_norm_and_idxs(n,m)
            self.normalization = normalization.copy()

            if n % 2 == 1:
                self.normalization *= -1

            N = self.idxs.shape[0]
            self.N = N

            # Allocate mapped array for final result
            self.phiU = cuda.mapped_array((N,N), dtype=np.complex128, stream=self.stream)

            # Allocate array on device for computation
            self.d_phiU = cuda.device_array((N,N), dtype=np.complex128, stream=self.stream)

            # Copy idxs to the device
            self.d_idxs = cuda.to_device(self.idxs,stream=self.stream)
            
            # Copy normalization to the device
            self.d_normalization = cuda.to_device(self.normalization.astype('complex128'), stream=self.stream)

            # Set up call parameters
            blockspergrid_x = (N + (self.threadsperblock[0] - 1)) // self.threadsperblock[0]
            blockspergrid_y = (N + (self.threadsperblock[1] - 1)) // self.threadsperblock[1]
            self.blockspergrid = (blockspergrid_x, blockspergrid_y)

            # Get the JITed kernel
            self.gpu_phi = self.get_phi(n,m)
            self.lastTSetup = time.time() - t1
    
        # Copy U to the device
        d_U = cuda.to_device(U,stream=self.stream)
        
        # Run the computation
        self.gpu_phi[self.blockspergrid,self.threadsperblock,self.stream](d_U, self.d_phiU, self.d_normalization, self.d_idxs)

        self.stream.synchronize()

        # Move the results to the mapped array
        self.d_phiU.copy_to_host(self.phiU, stream=self.stream)

        self.stream.synchronize()
        
        return self.phiU
    
    def get_phi(self, n, m):
        try:
            return self.gpuPhis[(m,n)]
        except KeyError:
            pass

        @cuda.jit('void(complex128[:,:], complex128[:,:], complex128[:,:], int64[:,:])')
        def gpu_phi(U, phiU, normalization, idxs):
            row, col = cuda.grid(2)
            if row >= phiU.shape[0]:
                return
            if col >= phiU.shape[1]:
                return

            U_T = cuda.local.array((m,n), complex128)
            U_ST = cuda.local.array((n,n), complex128)
            

            for j in range(n):
                J = idxs[col][j]
                for i in range(m):
                    U_T[i,j] = U[i,J]

            for i in range(n):
                for j in range(n):
                    I = idxs[row][i]
                    U_ST[i,j] = U_T[I,j]

            # Calculate the permanent
            perm = 0 
            for k in range(2**n):
                gray = k ^ (k >> 1)
                rowSumProd = 1 - 2*(k%2)
                for i in range(n):
                    rowSum = 0
                    for j in range(n):
                        if (gray >> j) & 1 == 1:
                            rowSum += U_ST[i,j]
                    rowSumProd *= rowSum
                perm += rowSumProd
            phiU[row, col] = perm / normalization[row, col]
            
            
        self.gpuPhis[(m,n)] = gpu_phi
        return gpu_phi