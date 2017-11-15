#!/usr/bin/env python
"""
15 Nov 2017
 
@author: landman
 
A collection of tools for matrices defined by multiple kronecker products (kronecker matrix for short) i.e.

A = K1 kron K2 kron K3 kron ...
 
These are useful for performing GPR when data are given on a (not necessarily uniform) Cartesian grid and when the 
covariance function is a tensor product kernel i.e. when it can be written as the product of 1D covariance functions.

"""

import numpy as np

def kron_vec(A, b):
    """
    Computes matrix vector product of kronecker matrix in linear time. 
    :param A: an array of arrays holding matrices [..., K3, K2, K1] (note ordering)
    :param b: the RHS vector
    :return: A.dot(b)
    """
    D = A.shape[0]
    N = b.size
    x = b
    for d in np.arange(D)[::-1]:
        Gd = A[d].shape[0]
        X = np.reshape(x,(Gd, N//Gd))
        Z = np.einsum("ab,bc->ac", A[d], X)
        Z = np.einsum("ab -> ba", Z)
        x = Z.flatten()
    return x

def kron_trace(A):
    """
    Computes the trace of a kronecker matrix
    :param A: an array of arrays holding matrices [K1, K2, K3, ...]
    :return: trace(A)
    """
    D = A.shape[0]
    x = 1.0
    for i in xrange(D):
        x *= np.trace(A[i])
    return x

def kron_cholesky(A):
    """
    Computes the cholesky decomposition of a kronecker matrix
    :param A: an array of arrays holding matrices [K1, K2, K3, ...]
    :return: 
    """