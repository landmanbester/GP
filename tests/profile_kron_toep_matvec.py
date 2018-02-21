#!/usr/bin/env pytho


import numpy as np
from GP.kernels import exponential_squared

if __name__=="__main__":
    N = 1024*2**10
    xmax = 5.0
    x = np.linspace(-xmax, xmax, N)
    theta = np.array([1.0, 1.0, 1.0])

    Kop = exponential_squared.sqexp_op(x, theta)

    b = np.random.randn(N)

    res = Kop(b)

