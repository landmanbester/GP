#!/usr/bin/env python
"""
A function to draw samples from GP with specified kernel
"""

import numpy as np
from GP.tools import abs_diff
from GP.kernels import exponential_squared as expsq
from GP.tools import kronecker_tools as kt

def draw_samples(meanf, x, theta, cov_func, N):
    XX = abs_diff.abs_diff(x, x)
    covf = cov_func(XX, theta)
    samps = np.random.multivariate_normal(meanf, covf, N)
    return samps

def draw_spatio_temporal_samples(meanf, x, t, theta_x, theta_t, Nsamps):
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

def draw_samples_ND_grid(x, theta, Nsamps, meanf=None):
    """
    Draw N dimensional samples on a Euclidean grid
    :param meanf: mean function
    :param x: array of arrays containing targets [x_1, x_2, ..., x_D]
    :param theta: array of arrays containing [theta_1, theta_2, ..., theta_D]
    :param Nsamps: number of samples to draw
    :return: array containing samples [Nsamps, N_1, N_2, ..., N_D]
    """

    D = x.shape[0]
    Ns = []
    K = np.empty(D, dtype=object)
    Kcov = expsq.sqexp()
    Ntot=1
    for i in xrange(D):
        Ns.append(x[i].size)
        XX = abs_diff.abs_diff(x[i], x[i])
        print Ns[i], theta[i], K[i]
        K[i] = Kcov.cov_func(theta[i], XX, noise=False) + 1e-13*np.eye(Ns[i])
        Ntot *= Ns[i]

    L = kt.kron_cholesky(K)
    samps = np.zeros([Nsamps]+Ns, dtype=np.float64)
    for i in xrange(Nsamps):
        xi = np.random.randn(Ntot)
        if meanf is not None:
            samps[i] = meanf(x) + kt.kron_matvec(L, xi).reshape(Ns)
        else:
            samps[i] = kt.kron_matvec(L, xi).reshape(Ns)
    return samps


def draw_time_freq_gain_samps(t, nu, St_func, Snu_func, Nsamps):
    # get time grid
    Nt = t.size
    t = np.sort(t)
    delt = t[1]-t[0]
    assert np.allclose(t[1::]-t[0:-1], delt)
    # get frequency grid
    Nnu = nu.size
    nu = np.sort(nu)
    delnu = nu[1] - nu[0]
    assert np.allclose(nu[1::] - nu[0:-1], delnu)
    # get time freqs and evaluate spectra
    Ntfreq = 2*Nt-2
    st = np.fft.fftfreq(Ntfreq, delt)
    St = St_func(st)
    # import matplotlib.pyplot as plt
    # plt.plot(st, St)
    # plt.show()
    # get nu freqs and evaluate spectra
    Nnufreq = 2*Nnu-2
    snu = np.fft.fftfreq(Nnufreq, delnu)
    Snu = Snu_func(snu)
    # plt.plot(snu, Snu)
    # plt.show()

    samps = np.zeros((Nsamps, Nt, Nnu), dtype=np.float64)
    Ntot = Ntfreq * Nnufreq
    for i in xrange(Nsamps):
        # draw random vector to generate sample
        xi = np.random.randn(Ntot)

        # use kronecker product trick to perform matvec
        # first for t axis
        Xi = np.reshape(xi, (Ntfreq, Nnufreq))  # reshape into matrix
        y = np.zeros((Ntfreq, Nnufreq), dtype=np.float64)  # tmp storage
        # get matrix vector product with each column
        for k in xrange(Nnufreq):
            xk = np.fft.fft(Xi[:, k])
            y[:, k] = np.fft.ifft(np.sqrt(St) * xk).real
        xi = y.T.flatten()

        # next for nu axis
        Xi = np.reshape(xi, (Nnufreq, Ntfreq))  # reshape into matrix
        y = np.zeros((Nnufreq, Ntfreq), dtype=np.float64)  # tmp storage
        # get matrix vector product with each column
        for k in xrange(Ntfreq):
            xk = np.fft.fft(Xi[:, k])
            y[:, k] = np.fft.ifft(np.sqrt(Snu) * xk).real
        samps[i] = y.T[0:Nt, 0:Nnu]
    return samps