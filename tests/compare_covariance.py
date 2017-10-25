#!/usr/bin/env python
"""
Eventually the unittests will go in here.
Currently just tests the RR expansion of the covariance function i.t.o. eigenfuncs of the Laplacian against the true 
covariance function.
"""

import numpy as np
from GP.kernels import exponential_squared
from GP.tools import make_basis
from GP.basisfuncs import rectangular

if __name__=="__main__":
    # set params and get matrix of differences
    N = 50
    Np = 250
    tmin = 0.0
    tmax = 1.0
    t = tmin + (tmax - tmin)*np.sort(np.random.random(N))
    tp = np.linspace(tmin, tmax, Np)
    from GP.tools import abs_diff
    tt = abs_diff.abs_diff(t, t)
    # ttp = abs_diff.abs_diff(t, tp)
    # ttpp = abs_diff.abs_diff(tp, tp)

    # instantiate kernel
    kernel = exponential_squared.sqexp()

    # get some random values of theta to compare RR to full
    sigfm = 1.0
    lm = 0.25
    signm = 1.0
    thetam = np.array([sigfm, lm, signm])

    Ntheta = 10
    theta = np.zeros([Ntheta, 3])
    deltheta = 1.0e-1
    for i in xrange(Ntheta):
        theta[i] = thetam + deltheta*np.random.randn(3)

    # construct true covariance matrices
    # K = np.zeros([Ntheta, N, N])
    # Kp = np.zeros([Ntheta, N, Np])
    # Kpp = np.zeros([Ntheta, Np, Np])
    # for i in xrange(Ntheta):
        # K[i] = kernel.cov_func(theta[i], tt, noise=False)
        # Kp[i] = kernel.cov_func(theta[i], ttp, noise=False)
        # Kpp[i] = kernel.cov_func(theta[i], ttpp, noise=False)

    # now get approximated covariance matrices
    from GP.tools import make_basis
    from GP.basisfuncs import rectangular
    M = np.array([24])
    L = np.array([2*(tmax - tmin)])
    Phi = make_basis.get_eigenvectors(t.reshape(N, 1), M, L, rectangular.phi)
    Lambda = make_basis.get_eigenvals(M, L, rectangular.Lambda)
    s = np.sqrt(Lambda)
    # S = np.zeros([Ntheta, M[0]])
    for i in xrange(Ntheta):
        K = kernel.cov_func(theta[i], tt, noise=False)
        S = kernel.spectral_density(theta[i], s)
        Ktilde = Phi.dot(np.diag(S).dot(Phi.T))
        print K[:, 0]/Ktilde[:, 0], theta[i,0:2], 2*L/M, L