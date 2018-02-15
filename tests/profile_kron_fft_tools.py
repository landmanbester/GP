#!/usr/bin/env python

import numpy as np
from GP.tools import kronecker_tools as kt
from GP.tools import abs_diff
from GP.kernels import exponential_squared
from GP.operators import covariance_ops

if __name__=="__main__":
    # set inputs
    Nx = 64
    sigmaf = 1.0
    lx = 0.25
    x = np.linspace(-1, 1, Nx)

    Nt = 64
    lt = 1.0
    t = np.linspace(0, 1, Nt)

    Nz = 64
    lz = 1.5
    z = np.linspace(0, 1, Nz)

    N = Nx * Nt * Nz

    # draw random vector to multiply with
    b = np.random.randn(N)

    # join up all targets
    X = np.array([t, x, z])
    sigman = 0.1
    theta0 = np.array([sigmaf, lt, lx, lz, sigman])

    # instantiate K operator
    Kop = covariance_ops.K_op(X, theta0, kernels=["sqexp", "sqexp", "sqexp"], grid_regular=True)

    # set diagonal noise matrix
    Sigmay = 0.1 * np.ones(N) + np.abs(0.1 * np.random.randn(N))

    # instantiate Ky operator
    Kyop = covariance_ops.Ky_op(Kop, Sigmay)

    # try solve Ky x = b
    res2 = Kyop.idot(b)