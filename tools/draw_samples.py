#!/usr/bin/env python
"""
A function to draw samples from GP with specified kernel
"""

import numpy as np
from GP.tools import abs_diff
from GP.kernels import exponential_squared as expsq
from GP.tools import kronecker_tools as kt

def draw_samples(meanf, x, theta, cov_func, N):
    XX = abs_diff.abs_diff(x, x)
    covf = cov_func(XX, theta)
    samps = np.random.multivariate_normal(meanf, covf, N)
    return samps

def draw_spatio_temporal_samples(meanf, x, t, theta_x, theta_t, Nsamps):
    Nt = t.size
    Nx = x.size  # assumes 1D
    Ntot = Nt*Nx
    XX = abs_diff.abs_diff(x, x)
    TT = abs_diff.abs_diff(t, t)
    Kcov = expsq.sqexp()
    Kx = Kcov.cov_func(theta_x, XX, noise=False) + 1e-13*np.eye(Nx)
    Kt = Kcov.cov_func(theta_t, TT, noise=False) + 1e-13*np.eye(Nt)
    K = np.array([Kt, Kx])
    L = kt.kron_cholesky(K)
    samps = np.zeros([Nsamps, Nt, Nx])
    for i in xrange(Nsamps):
        xi = np.random.randn(Ntot)
        samps[i] = meanf + kt.kron_matvec(L, xi).reshape(Nt, Nx)
    return samps

def draw_samples_ND_grid(x, theta, Nsamps, meanf=None, dtype=np.complex128):
    """
    Draw N dimensional samples on a Euclidean grid
    :param meanf: mean function
    :param x: array of arrays containing targets [x_1, x_2, ..., x_D]
    :param theta: array of arrays containing [theta_1, theta_2, ..., theta_D]
    :param Nsamps: number of samples to draw
    :return: array containing samples [Nsamps, N_1, N_2, ..., N_D]
    """

    D = x.shape[0]
    Ns = []
    K = np.empty(D, dtype=object)
    Kcov = expsq.sqexp()
    Ntot=1
    for i in xrange(D):
        Ns.append(x[i].size)
        XX = abs_diff.abs_diff(x[i], x[i])
        K[i] = Kcov.cov_func(theta[i], XX, noise=False) + 1e-13*np.eye(Ns[i])
        Ntot *= Ns[i]

    L = kt.kron_cholesky(K)
    samps = np.zeros([Nsamps]+Ns, dtype=dtype)
    for i in xrange(Nsamps):
        xi = np.random.randn(Ntot) + 1.0j * np.random.randn(Ntot)
        if meanf is not None:
            samps[i] = meanf(x) + kt.kron_matvec(L, xi).reshape(Ns)
        else:
            samps[i] = kt.kron_matvec(L, xi).reshape(Ns)
    return samps, K

