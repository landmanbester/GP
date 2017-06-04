#!/usr/bin/env python
"""
In this example we see how to do a 1D GP in both Full and RR modes. We also compare perfomance of these  
"""

import numpy as np
from GP import temporal_GP
import matplotlib.pyplot as plt

if __name__=="__main__":
    # Set some inputs
    N = 100
    xmax = 5.0
    xmin = -5.0
    x = np.linspace(xmin, xmax, N)

    # create a simple function
    yf = lambda x, a, b, c: a*x**2 + b*x + c

    # set parameters
    a = 1.0
    b = 1.0
    c = 1.0

    # simulate some data
    ytrue = yf(x, a, b, c)
    sigma_n = 0.25
    y = ytrue + sigma_n**2*np.random.randn(N)

    # Set mode and targets
    mode = "Full"
    Np = 1000
    xp = np.linspace(xmin, xmax, Np)
    # instantiate GP class
    GP = temporal_GP.TemporalGP(x, xp, y, covariance_function='sqexp', mode=mode)

    # Guess inital hypers
    sigmaf0 = 1.0
    l0 = 1.0
    sigman0 = 0.5
    theta0 = np.array([sigmaf0, l0, sigman0])

    # Train GP
    thetaf = GP.logp(theta0)

    # Evaluate mean and covariance with these values
    fp = GP.meanf(thetaf)
    fcov = GP.covf(thetaf)

    # Draw some samples
    Nsamps = 10
    samps = np.random.multivariate_normal(fp, fcov, Nsamps).T

    # Plot the results
    dir = '/home/landman/Projects/GP/examples/'
    plt.figure("temp_GP")
    plt.plot(x, fp, 'k', lw=2, label='Mean')
    plt.plot(x, samps, 'b', alpha=0.5)
    plt.savefig(dir+"temp_GP.png", dpi=250)

