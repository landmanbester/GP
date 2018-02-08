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

def kron_N(x):
    """
    Computes N = N_1 x N_2 x ... x N_D i.e. the total number of points in x 
    :param x: an array of arrays holding matrices/vectors [x1, x2, ..., xD]
    :return: 
    """
    D = x.shape[0]
    dims = np.zeros(D)
    for i in xrange(D):
        dims[i] = x[i].shape[0]
    N = int(np.prod(dims))
    return N

def kron_kron(A):
    """
    Computes the Kronecker product of a series of matrices or vectors 
    :param A: an array of arrays holding matrices/vectors [A1, A2, ..., AD]
    :return: the Kronecker product of all elements of A
    """
    D = A.shape[0]
    # expand the kronecker product for B
    K = np.kron(A[0], A[1])
    if D > 2:
        for i in xrange(2, D):
            K = np.kron(K, A[i])
    return K

def kron_transpose(A):
    """
    Transposes the elements of a kronecker matrix
    :param A: an array of arrays holding matrices/vectors [A1, A2, ..., AD]
    :return: an array of arrays holding matrices/vectors [A1.T, A2.T, ..., AD.T]
    """
    D = A.shape[0]
    AT = np.zeros_like(A)
    for i in xrange(D):
        AT[i] = A[i].T
    return AT

def kron_matvec(A, b):
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

def kron_matmat(A, B):
    """
    Computes the product of two kronecker matrices
    :param A: 
    :param B: 
    :return: 
    """
    D = B.shape[0]
    K = kron_kron(B)
    M = K.shape[1]  # the product of Np_1 x Np_2 x ... x Np_3

    # do kron_matvec on each column of result
    dims = np.zeros(D)
    for i in xrange(D):
        dims[i] = A[i].shape[0]
    N = int(np.prod(dims))
    C = np.zeros([N, M])
    for i in xrange(M):
        C[:,i] = kron_matvec(A, K[:,i])
    return C

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
        try:
            L[i] = np.linalg.cholesky(A[i])
        except: # add jitter
            L[i] = np.linalg.cholesky(A[i] + 1e-13*np.eye(A[i].shape[0]))
    return L

def kron_logdet(L):
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
    Computes the kronecker product of a series of diagonal matrices defined as vectors
    :param A: an array of arrays holding vectors [x1, x2, x3, ...] where each vector is the diagonal of a matrix
    :return: the diagonal matrix of the kronecker product
    """
    D = A.shape[0]
    Adiag = []
    for i in xrange(D):
        Adiag.append(A[i])
    Adiag = np.asarray(Adiag)
    return kron_kron(Adiag)

def kron_diag_diag(A):
    """
    Computes the kronecker product of a series of diagonal matrices defined as vectors
    :param A: an array of arrays holding vectors [x1, x2, x3, ...] where each vector is the diagonal of a matrix
    :return: the diagonal matrix of the kronecker product
    """
    D = A.shape[0]
    Adiag = []
    for i in xrange(D):
        Adiag.append(np.diag(A[i]))
    Adiag = np.asarray(Adiag)
    return kron_kron(Adiag)

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

def kron_eig(A, dtype='real'):
    """
    Computes the eigendecomposition of a kronecker matrix 
    :param A: an array of arrays holding matrices [K1, K2, K3, ...]
    :return: Eigenvectors Q and eigenvalues Lambda of a all submatrices
    """
    D = A.shape[0]
    # dims = np.zeros(D)
    # for i in xrange(D):
    #     dims[i] = A[i].shape[0]

    Qs = np.zeros_like(A)
    Lambdas = []

    if dtype=='real':
        for i in xrange(D):
            Lambda, Q = np.linalg.eigh(A[i])
            Qs[i] = Q
            Lambdas.append(Lambda)
        Lambdas = np.asarray(Lambdas)
    elif dtype=='complex':
        for i in xrange(D):
            Lambda, Q = np.linalg.eig(A[i])
            Qs[i] = Q
            Lambdas.append(Lambda)
        Lambdas = np.asarray(Lambdas)
    return Lambdas, Qs

if __name__=="__main__":
    from GP.tools import abs_diff
    from GP.kernels import exponential_squared


    kernel = exponential_squared.sqexp()
    Nx = 7
    thetax = np.array([0.25, 0.25])
    x = np.linspace(-1, 1, Nx)
    xx = abs_diff.abs_diff(x, x)
    Kx = kernel.cov_func(thetax, xx, noise=False)

    Nt = 9
    thetat = np.array([1.0, 1.0])
    t = np.linspace(0, 1, Nt)
    tt = abs_diff.abs_diff(t, t)
    Kt = kernel.cov_func(thetat, tt, noise=False)

    Nz = 11
    thetaz = np.array([1.5, 1.5])
    z = np.linspace(0, 1, Nz)
    zz = abs_diff.abs_diff(z, z)
    Kz = kernel.cov_func(thetaz, zz, noise=False)

    N = Nx * Nt * Nz
    b = np.random.randn(N)

    K = np.kron(Kz, np.kron(Kx, Kt))
    A = np.array([Kt, Kx, Kz])  # note ordering!!!

    # test kron_kron
    K2 = kron_kron(A[::-1])
    print "kron diff = ", np.abs(K - K2).max()

    # test matvec
    res1 = np.dot(K, b)
    res2 = kron_matvec(A, b)
    print "matvec diff = ", np.abs(res1 - res2).max()

    # test matmat
    A2 = np.array([Kt + np.random.randn(Nt, Nt), Kx + np.random.randn(Nx, Nx), Kz + np.random.randn(Nz, Nz)])
    K2 = kron_kron(A2)
    res1 = np.dot(K, K2)
    res2 = kron_matmat(A, A2)
    print "matmat diff =", np.abs(res1 - res2).max()

    # test trace
    res1 = np.trace(K)
    res2 = kron_trace(A)
    print "Trace diff = ", np.abs(res1 - res2).max()

    # Test Cholesky (seems numerically unstable because K + jitter doesn't factor as kronecker product)
    res1 = np.linalg.cholesky(K + 1e-13*np.eye(K.shape[0]))
    L = kron_cholesky(A[::-1])
    res2 = kron_kron(L)
    print "Cholesky diff = ", np.abs(res1 - res2).max()

    # test logdet
    res1 = np.linalg.slogdet(K)[1]
    res2 = kron_logdet(L)
    print "cholesky logdet diff = ", np.abs(res1 - res2).max()

    # test eigenvalues
    Lambda, Q = np.linalg.eigh(K)
    Lambdas, Qs = kron_eig(A[::-1])
    Lambda2 = kron_diag(Lambdas)
    # sort to test eigenvalues
    I = np.argsort(Lambda2)
    Lambda3 = Lambda2[I]
    res2 = np.sum(np.log(Lambda2))
    print "eigenvalue logdet diff = ", np.abs(res1 - res2).max()

    print "Lambda diff = ", np.abs(Lambda - Lambda3).max()

    # test reconstructed K (eigenvectors not unique)
    Q2 = kron_kron(Qs)
    K2 = Q2.dot(np.dot(np.diag(Lambda2), Q2.T))
    print "K diff from eigen decomp = ", np.abs(K - K2).max()

