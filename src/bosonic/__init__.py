from __future__ import print_function, absolute_import, division
from .aa_phi import build_norm_and_idxs, build_kIdxs, binom
from . import density
from .phi_dispatch import aa_phi, aa_phi_cpu, aa_phi_gpu

__all__ = [
    "__title__", "__summary__", "__uri__", "__version__", "__author__",
    "__email__", "__license__", "__copyright__", "aa_phi", "density",
]

__title__ = "bosonic"
__version__ = "0.1"
__description__ = "Library for fast computations involving indistinguishable bosons"
__url__ = "https://github.com/steinbrecher/bosonic"

__author__ = "Greg Steinbrecher"
__email__ = "steinbrecher@alum.mit.edu"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2017-2018 Greg Steinbrecher"
