#!/usr/bin/env python
"""
This class performs GPR on a regular cartesian grid in O(N) time where N = N_1 x N_2 X ... x N_D

So far only the vanilla implementation is available, still need to figure out how to implement RR and SS versions in 
a consistent way
"""

import numpy as np
from scipy import optimize as opt
from GP.tools import posterior_mean, posterior_covariance, marginal_posterior, abs_diff
from GP.tools import kronecker_tools as kt
from GP.operators import covariance_ops

class KronToepGP(object):
    def __init__(self, x, xp, y, theta, Sigmay=None, kernels='sqexp'):
        """
        :param x: an array of arrays holding inputs i.e. x = [[x_1], [x_2], ..., [x_3]] where x_1 in R^{N_1} etc.
        :param xp: an array of arrays holding targets i.e. xp = [[xp_1], [xp_2], ..., [xp_3]] where xp_1 in R^{Np_1} etc.
        :param y: an array of arrays holding data i.e. y = [[y_1], [y_2], ..., [y_3]] where y_1 in R^{N_1} etc.
        :param Sigmay: an array of arrays holding variances i.e. Sigmay = [[Sigmay_1], [Sigmay_2], ..., [Sigmay_3]] 
                where Sigmay_1 in R^{N_1} etc.
        :param covariance_function: a list holding the covariance functions to use for each dimension
        """
        self.D = x.shape[0]
        self.N = y.size
        self.x = x
        self.Np = kt.kron_N(xp)
        self.xp = xp
        self.y = y.flatten()

        # instantiate
        self.theta = theta
        self.kernels = kernels
        self.Kop = covariance_ops.K_op(x, theta, kernels)
        self.Kyop = covariance_ops.Ky_op(self.Kop, Sigmay)

        # Instantiate posterior mean, cov and evidence classes
        # self.meano = posterior_mean.meanf(self.x, self.xp, self.y, self.kernels, mode="kron",
        #                                   XX=self.XX, XXp=self.XXp)
        # self.meanf = lambda theta: self.meano.give_mean(theta)
        # self.covo = posterior_covariance.covf(self.kernels, mode="kron", XX=self.XX, XXp=self.XXp, XXpp=self.XXpp)
        # self.covf = lambda theta: self.covo.give_covariance(theta)
        self.logpo = marginal_posterior.evidence_op(self.y, self.Kyop)

    def train(self, theta0, bounds=None):
        """
        Trains the hypers using built in scipy optimizer
        :param theta0: initial guess for hypers
        :param bounds: optional set min and max bounds on hypers
        :return: the optimised hyperparameters
        """
        if bounds is None:
            Ntheta = theta0.size
            bnds = []
            for i in xrange(Ntheta):
                bnds.append((1e-6, None))
            # Set default bounds for hypers (they must be strictly positive)
            bnds = tuple(bnds)
        else:
            bnds = bounds # make sure above criteria on bounds satisfied if passing default bounds

        # Do optimisation
        thetap = opt.fmin_l_bfgs_b(self.logpo.logL, theta0, fprime=None, bounds=bnds, approx_grad=True) #, factr=1e10, pgtol=0.1)

        theta = thetap[0]

        #Check for convergence
        if thetap[2]["warnflag"]:
            print "Warning flag raised"
        # Return optimised value of theta
        return theta


def func(nu, t, sigma):
    return 5*np.sinc(nu)*5*np.sinc(t) + sigma*np.random.randn(int(np.sqrt(nu.size)), int(np.sqrt(t.size)))

if __name__ == "__main__":
    # generate some data
    Nnu = 100
    numax = 5.0
    nu = np.linspace(-numax, numax, Nnu)
    Nnup = 100
    nup = np.linspace(-numax, numax, Nnup)

    Nt = 100
    t = np.linspace(-numax, numax, Nt)
    Ntp = 100
    tp = np.linspace(-numax, numax, Ntp)

    nunu, tt = np.meshgrid(nu, t)
    x = np.array([nu, t])
    nunup, ttp = np.meshgrid(nup, tp)
    xp = np.vstack((nunup.flatten(), ttp.flatten()))
    xp = np.array([nup, tp])

    sigma = 0.1
    y = func(nunu, tt, sigma)

    # get the true function at target points
    fp = func(nunup, ttp, 0.0)

    # set hypers
    sigmaf = 2.12662296
    l1 = 0.91048988
    l2 = 0.90298655
    sigman = 0.10111004
    theta0 = np.array([sigmaf, l1, l2, sigman])

    # create GP object
    print "Instantiationg GP"
    # do for transformed data
    N = Nnu*Nt
    Sigmay = (0.1 * np.ones([Nnu, Nt]) + np.abs(0.1 * np.random.randn(Nnu, Nt))).flatten()
    GP = KronToepGP(x, xp, y, theta0, Sigmay=Sigmay, kernels=["sqexp", "sqexp"])

    print "Training"
    theta = GP.train(theta0)

    # fmean = GP.meanf(theta).reshape(nunup.shape)



    # # plot expected result
    # import matplotlib.pyplot as plt
    # plt.figure('Data')
    # plt.imshow(y)
    # plt.colorbar()
    # plt.figure('True')
    # plt.imshow(fp)
    # plt.colorbar()
    # plt.figure('fp')
    # plt.imshow(fmean)
    # plt.colorbar()
    # plt.figure('Data2')
    # plt.imshow(GP.ytransformed)
    # plt.colorbar()
    # plt.figure('fp2')
    # plt.imshow(fmean2)
    # plt.colorbar()
    # plt.show()



