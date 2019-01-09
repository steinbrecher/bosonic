Bosonic: A Quantum Optics Library
=======

Bosonic is a library developed for the siimulation of photonic systems whose
inputs are indistinguishable bosons (in the case of the authors' interest,
photons). In particular, it focuses on the rapid computation of the
multi-particle transfer functions for these systems, and supports computation
of the gradient of a cost function with respect to the system parameters.
It was originally developed for the devleopment of our Quantum Optical
Neural Networks [1] and contains specialized functionality for their
simulation and optimization.

Key Functionality
=================
The core motivation for this package was the rapid computation of the
multi-particle unitary transform as a function of the single particle unitary
and the number of bosonic inputs. That is, if we have a four dimensional 
unitary U, and we know there are 3 photons at the input, we want to know the
transformation over the Choose[4+3-1, 3]-dimensional basis [3,0,0,0],
[2,1,0,0], [2,0,1,0], ... etc.

This is supported by the function `bosonic.aa_phi`, which is named after
Aaronson and Arkhipov, who specified the form of this function that we use
as their phi(U) function in [2]. 

References
==========
[1] Steinbrecher, G. R., Olson, J. P., Englund, D., & Carolan, J. (2018). Quantum optical neural networks. arXiv preprint arXiv:1808.10047. https://arxiv.org/abs/1808.10047
[2] Aaronson, Scott, and Alex Arkhipov. "The computational complexity of linear optics." Proceedings of the forty-third annual ACM symposium on Theory of computing. ACM, 2011. https://arxiv.org/pdf/1011.3245.pdf
