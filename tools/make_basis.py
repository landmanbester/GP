"""
Constructs the basis functions
"""

import numpy as np

def make_basis(x, M, L, phi):
    """
    Constructs the basis functions
    :param x: NxD vector with which to evaluate the basis functions 
    :param M: Dx1 array of ints specifying the number of basis functions to use for each dimension
    :param L: Dx1 array of floats specifying the domain boundaries in each dimension
    :param phi: the basis functions to evaluate 
    """
    D = M.size # the dimension of the space
    Ntot = np.prod(M) # the total number of basis functions over all dimensions
    # make the integer grid

    for i in xrange(Ntot):
        print "x"
    return