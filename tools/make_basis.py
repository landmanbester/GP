"""
Constructs the basis functions
"""

import numpy as np

def get_eigenvectors(x, M, L, phi):
    """
    Constructs the basis functions
    :param x: NxD vector with which to evaluate the basis functions 
    :param M: Dx1 array of ints specifying the number of basis functions to use for each dimension
    :param L: Dx1 array of floats specifying the domain boundaries in each dimension
    :param phi: the eigenfunctions to evaluate
    """
    N = x.shape[0]
    # Get the total number of basis functions over all dimensions
    Ntot = int(np.prod(M))

    # Create array to store basis functions
    PHI = np.zeros([N, Ntot])

    # iterate over index list of basis funcs
    for i, index in enumerate(np.ndindex(tuple(M))):
        PHI[:, i] = phi(x, np.asarray(index) + 1, L)
    return PHI

def get_eigenvals(M, L, Lambda):
    """
    Constructs eigenvalues corresponding to basis functions
    :param M: Dx1 array of ints specifying the number of basis functions to use for each dimension
    :param L: Dx1 array of floats specifying the domain boundaries in each dimension
    :param Lambda: the corresponding eigenvectors
    """
    # Get the total number of basis functions over all dimensions
    Ntot = int(np.prod(M))

    # Create array to store eigenvalues
    LAMBDA = np.zeros(Ntot)

    # iterate over index list of basis funcs
    for i, index in enumerate(np.ndindex(tuple(M))):
        LAMBDA[i] = Lambda(np.asarray(index) + 1, L)
    return LAMBDA