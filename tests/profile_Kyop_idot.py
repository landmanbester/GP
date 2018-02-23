#!/usr/bin/env python


import numpy as np
from GP.operators import covariance_ops

if __name__=="__main__":
    # set inputs
    Nx = 100
    sigmaf = 1.0
    lx = 0.25
    x = np.linspace(0, 1, Nx)

    Nt = 100
    lt = 0.5
    t = np.linspace(-1, 0, Nt)

    Nz = 100
    lz = 0.35
    z = np.linspace(-2, -1, Nz)

    N = Nx * Nt * Nz

    # draw random vector to multiply with
    b = np.random.randn(N)

    # join up all targets
    X = np.array([t, x, z])
    sigman = 0.1
    theta0 = np.array([sigmaf, lt, lx, lz, sigman])

    # instantiate K operator
    Kop = covariance_ops.K_op(X, theta0, kernels=["sqexp", "sqexp", "sqexp"])

    # set diagonal noise matrix
    Sigmay = np.ones(N) + np.abs(0.2 * np.random.randn(N))

    # instantiate Ky operator
    Kyop = covariance_ops.Ky_op(Kop, Sigmay)

    # test preconditioner
    res = Kyop(b)
    res2 = Kyop.idot(res)
    print np.abs(b - res2).max(), np.abs(b - res2).min()

    print "Niter = ", Kop.count