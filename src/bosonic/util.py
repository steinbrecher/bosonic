from __future__ import print_function, absolute_import, division

import numpy as np

def memoize(f):
    memory = dict()
    def decorated(*args):
        if args not in memory:
            memory[args] = f(*args)
        return memory[args]
    return decorated

def haar_rand(m):
    """Returns haar-random matrix of dimension m x m
    The haar-random algorithm is taken from this paper:
    http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
    and is a direct adaptation of the following Mathematica code:

    RR:=RandomReal[NormalDistribution[0,1]];
    RC:=RR+I*RR;
    RG[n_]:=Table[RC,{n},{n}];
    RU[n_]:=Orthogonalize[RG[n]];
    RU[n_]:=Module[{Q,R,r,L},
      {Q,R}=QRDecomposition[RG[n]];
      r=Diagonal[R];
      L=DiagonalMatrix[r/Abs[r]];
      Q.L
    ];
    """
    z = np.random.randn(m,m) + 1j*np.random.randn(m,m)
    Q,R = np.linalg.qr(z)
    r = np.diag(R)
    L = np.diag(r / np.abs(r))
    return Q.dot(L)
