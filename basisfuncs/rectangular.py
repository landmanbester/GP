"""
Basis functions in the form of eigenvectors of Laplacian on a square domain with Dirichlet boundary conditions as well
as associated eigenvals 
"""

import numpy as np

def phi(x, j, L):
    """
    Evaluates eigenfuncs 
    :param x: NxD array of coordinates
    :param j: Dx1 array of integers (order of the basis function in the summation)
    :param L: Dx1 array of floats (boundary of domain)
    """
    return np.prod(np.sin(j*np.pi*(x + L)/(2*L))/np.sqrt(L), axis=1)

def Lambda(j, L):
    """
    Evaluates eigenvals
    :param j: Dx1 array of integers (order of the basis function in the summation)
    :param L: Dx1 array of floats (boundary of domain)
    """
    return np.sum(j*np.pi/(2*L)**2)