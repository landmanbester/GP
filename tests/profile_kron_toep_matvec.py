#!/usr/bin/env python


import numpy as np
from GP.kernels import exponential_squared

if __name__=="__main__":
    N = 2**15
    xmax = 5.0
    x = np.linspace(-xmax, xmax, N)
    theta = np.array([1.0, 1.0, 1.0])

    M = 20
    Kop = exponential_squared.sqexp_op(x, theta, M, wisdom_file='/home/landman/Projects/GP/fft_wisdom/test.wisdom.npy', reset_wisdom=False)

    b = np.random.randn(N, M)

    for i in xrange(100):
        Kop.matmat(b)

