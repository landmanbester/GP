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
    D = A.shape[0]
    L = np.zeros_like(A)
    for i in xrange(D):
        L[i] = np.linalg.cholesky(A[i])
    return L

def kron_det(L):
    """
    Computes the determinant of kronecker matrix 
    :param L: an array of arrays holding matrices [L1, L2, L3, ...]
    :return: 
    """
    D = L.shape[0]
    # first get dimensions and determinants of individual matrices
    dims = np.zeros(D)
    dets = np.zeros(D)
    for i in xrange(D):
        dims[i] = L[i].shape[0]
        dets[i] = 2.0*np.sum(np.log(np.diag(L[i])))

    # now compute combined determinant
    x = 0.0
    n_prod = np.prod(dims)
    for i in xrange(D):
        x += n_prod*dets[i]/dims[i]

    return x

def kron_diag(A):
    """
    Computes the diag of a kronecker matrix
    :param A: an array of arrays holding matrices [K1, K2, K3, ...]
    :return: 
    """
    D = A.shape[0]
    return

def kron_inverse(A):
    """
    Computes the inverse of a kronecker matrix
    :param A: an array of arrays holding matrices [K1, K2, K3, ...]
    :return: 
    """
    D = A.shape[0]
    Ainv = np.zeros_like(A)
    for i in xrange(D):
        Ainv[i] = np.linalg.inv(A[i])
    return Ainv

def kron_eig(A):
    """
    Computes the eigendecomposition of a kronecker matrix 
    :param A: an array of arrays holding matrices [K1, K2, K3, ...]
    :return: 
    """
    D = A.shape[0]
    dims = np.zeros(D)
    for i in xrange(D):
        dims[i] = A[i].shape[0]

    Qs = np.zeros_like(A)
    lams = np.zeros(D, A)