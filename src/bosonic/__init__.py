from __future__ import print_function, absolute_import, division
from .aa_phi import aa_phi, aa_phi_lossy, aa_phi_restricted
from .fock import binom
from . import density
# from .phi_dispatch import aa_phi, aa_phi_cpu, aa_phi_gpu

__all__ = [
    "__title__", "__summary__", "__uri__", "__version__", "__author__",
    "__email__", "__license__", "__copyright__", "aa_phi", "aa_phi_lossy",
    "aa_phi_restricted", "density", "binom"
]

__title__ = "bosonic"
__version__ = "0.2"
__description__ = "Library for fast indistinguishable boson computations"
__url__ = "https://github.com/steinbrecher/bosonic"

__author__ = "Greg Steinbrecher"
__email__ = "steinbrecher@alum.mit.edu"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2017-2018 Greg Steinbrecher"
