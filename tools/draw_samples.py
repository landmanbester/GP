#!/usr/bin/env python
"""
A function to draw samples from GP with specified kernel
"""

import numpy as np
from GP.tools import abs_diff

def draw_samples(meanf, x, theta, cov_func, N):
    XX = abs_diff.abs_diff(x, x)
    covf = cov_func(XX, theta)
    samps = np.random.multivariate_normal(meanf, covf, N)
    return samps

def draw_spatio_temporal_samples(meanf, x, t, theta_x, theta_t, Nsamps):
    from GP.kernels import exponential_squared as expsq
    from GP.tools import kronecker_tools as kt
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




