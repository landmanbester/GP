#!/usr/bin/env python
"""
In this example we see how to do a 1D GP in both Full and RR modes. We also compare perfomance of these  
"""

import numpy as np
from GP import temporal_GP
import matplotlib.pyplot as plt

if __name__=="__main__":
    # Set some inputs
    N = 250
    xmax = 5.5
    xmin = -5.5
    x = xmin + (xmax - xmin)*np.random.random(N)

    # set parameters
    a = 5.0
    b = 1.0
    c = 1.0

    # create a simple function
    yf = lambda x: a*x**2 + b*x + c

    # simulate some data
    ytrue = yf(x)
    sigma_n = 2.5
    sy = sigma_n*np.random.randn(N)
    y = ytrue + sy

    # Set mode and targets
    mode = "RR"
    Np = 150
    xp = np.linspace(xmin, xmax, Np)
    # instantiate GP class
    GP = temporal_GP.TemporalGP(x, xp, y, prior_mean=yf, covariance_function='sqexp', mode=mode, M=25, L=10)

    # Guess inital hypers
    sigmaf0 = 1.0
    l0 = 1.0
    sigman0 = 1.5
    theta0 = np.array([sigmaf0, l0, sigman0])

    # Train GP
    thetaf = GP.train(theta0)

    print "Optimised hyperparameters = ", thetaf

    # Evaluate mean and covariance with these values
    fp = GP.meanf(thetaf)

    # Draw some samples
    Nsamps = 10
    samps = GP.draw_samps(Nsamps,  thetaf)

    # Plot the results
    dir = '/home/landman/Projects/GP/examples/'
    plt.figure("temp_GP")
    plt.plot(xp, fp, 'k', lw=2, label='Mean')
    plt.plot(xp, yf(xp), 'g', label="Model")
    plt.plot(xp, samps, 'b', alpha=0.25, label="samps")
    plt.errorbar(x, y, sy, fmt='xr', alpha=0.25)
    plt.savefig(dir+"temp_GP"+mode+".png", dpi=250)

