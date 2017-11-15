


import numpy as np
from GP import temporal_GP
import matplotlib.pyplot as plt

if __name__=="__main__":
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
    sigma_n = 5.5
    sy = sigma_n * np.random.randn(N)
    y = ytrue + sy

    Np = 150
    xp = np.linspace(xmin, xmax, Np).reshape(Np, 1)
    mode = "Full"
    GP = temporal_GP.TemporalGP(x, xp, y, prior_mean=yf, covariance_function='sqexp', mode=mode, M=25, L=20)

    sigmaf0 = np.std(y)
    l0 = (xmax - xmin) / 2.0
    sigman0 = 1.0
    theta0 = np.array([sigmaf0, l0, sigman0])

    # Train GP
    thetaf = GP.train(theta0)

    GP.set_posterior(thetaf)

    postm = GP.post_mean.squeeze()
    postcov = GP.post_cov
    uncert = np.sqrt(np.diag(postcov))

    plt.figure("Posterior")
    plt.fill_between(xp.squeeze(), postm - uncert, postm + uncert, facecolor='b', alpha=0.5)
    plt.plot(xp, postm, 'k')

    xp2 = np.linspace(xmin, xmax, 75)
    GP.set_posterior(thetaf, xp=xp2)

    postm2 = GP.post_mean.squeeze()
    postcov2 = GP.post_cov
    uncert2 = np.sqrt(np.diag(postcov2))

    plt.figure("Posterior 2")
    plt.fill_between(xp2, postm2 - uncert2, postm2 + uncert2, facecolor='b', alpha=0.5)
    plt.plot(xp2, postm2, 'k')
    plt.show()
