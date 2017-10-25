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
    N = 100
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
    lm = 0.5
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
    L = np.array([3*(tmax - tmin)])
    Phi = make_basis.get_eigenvectors(t.reshape(N, 1), M, L, rectangular.phi)
    Lambda = make_basis.get_eigenvals(M, L, rectangular.Lambda)
    s = np.sqrt(Lambda)
    S = np.zeros([Ntheta, M[0]])
    for i in xrange(Ntheta):
        K = kernel.cov_func(theta[i], tt, noise=False)
        dKdtheta = kernel.dcov_func(theta[i], tt, K, mode=1)
        S = kernel.spectral_density(theta[i], s)
        dSdtheta = kernel.dspectral_density(theta[i], S, s, mode=1)
        Ktilde = Phi.dot(np.diag(S).dot(Phi.T))
        dKtildedtheta = Phi.dot(np.diag(dSdtheta).dot(Phi.T))
        #print K[:, 0]/Ktilde[:, 0], theta[i,0:2], theta[i, 0:2], 2*L/M, L
        #print np.mean(K[:, 0]/Ktilde[:, 0])
        #print dKdtheta[:, 1] / dKtildedtheta[:, 1], theta[i, 0:2], 2 * L / M, L
        print np.max(np.abs(dKdtheta - dKtildedtheta)), theta[i, 0:2], 2 * L / M, L
    # compare evidence
    # set parameters
    a = 5.0
    b = 1.0
    c = 1.0

    # create a simple function
    yf = lambda x: a*x**2 + b*x + c

    # simulate some data
    ytrue = yf(t)
    sy = signm * np.random.randn(N)
    y = ytrue + sy

    from GP.tools import marginal_posterior
    evidencefull = marginal_posterior.evidence(t, y, kernel, mode="Full", XX=tt)
    evidenceRR = marginal_posterior.evidence(t, y, kernel, mode="RR", Phi=Phi, PhiTPhi=np.dot(Phi.T, Phi), s=s)

    for i in xrange(Ntheta):
        print evidencefull.logL(theta[i])
        print evidenceRR.logL(theta[i])