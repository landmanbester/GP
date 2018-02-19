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

class KronGP(object):
    def __init__(self, x, xp, y, Sigmay=None, covariance_function='sqexp'):
        """
        :param x: an array of arrays holding inputs i.e. x = [[x_1], [x_2], ..., [x_3]] where x_1 in R^{N_1} etc.
        :param xp: an array of arrays holding targets i.e. xp = [[xp_1], [xp_2], ..., [xp_3]] where xp_1 in R^{Np_1} etc.
        :param y: an array of arrays holding data i.e. y = [[y_1], [y_2], ..., [y_3]] where y_1 in R^{N_1} etc.
        :param Sigmay: an array of arrays holding variances i.e. Sigmay = [[Sigmay_1], [Sigmay_2], ..., [Sigmay_3]] 
                where Sigmay_1 in R^{N_1} etc.
        :param covariance_function: a list holding the covariance functions to use for each dimension
        """
        self.D = x.shape[0]
        self.N = kt.kron_N(x)
        self.x = x
        self.Np = kt.kron_N(xp)
        self.xp = xp
        self.y = y
        if Sigmay is not None:
            self.Sigmay = Sigmay
        else:
            self.Sigmay = None
        self.covariance_function = covariance_function

        # initialise the individual diff square grids
        self.XX = []
        self.XXp = []
        self.XXpp = []
        for i in xrange(self.D):
            self.XX.append(abs_diff.abs_diff(self.x[i], self.x[i]))
            self.XXp.append(abs_diff.abs_diff(self.x[i], self.xp[i]))
            self.XXpp.append(abs_diff.abs_diff(self.xp[i], self.xp[i]))
        self.XX = np.asarray(self.XX)
        self.XXp = np.asarray(self.XXp)
        self.XXpp = np.asarray(self.XXpp)

        # set the kernels
        from GP.kernels import exponential_squared  # only using this one for now
        self.kernels = []
        self.kernels_transformed = []
        for i in xrange(self.D):
            if self.Sigmay is not None:
                self.kernels.append(exponential_squared.sqexp(Sigmay=self.Sigmay[i], mode='kron'))
                #self.kernels_transformed.append(exponential_squared.sqexp(mode='kron'))
            else:
                self.kernels.append(exponential_squared.sqexp(mode='kron'))

        # Instantiate posterior mean, cov and evidence classes
        self.meano = posterior_mean.meanf(self.x, self.xp, self.y, self.kernels, mode="kron",
                                          XX=self.XX, XXp=self.XXp)
        self.meanf = lambda theta: self.meano.give_mean(theta)
        self.covo = posterior_covariance.covf(self.kernels, mode="kron", XX=self.XX, XXp=self.XXp, XXpp=self.XXpp)
        self.covf = lambda theta: self.covo.give_covariance(theta)
        self.logpo = marginal_posterior.evidence(self.x, self.y, self.kernels, mode="kron", XX=self.XX)
        self.logp = lambda theta: self.logpo.logL(theta)

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
        thetap = opt.fmin_l_bfgs_b(self.logp, theta0, fprime=None, bounds=bnds) #, approx_grad=True) #, factr=1e10, pgtol=0.1)

        theta = thetap[0]

        #Check for convergence
        if thetap[2]["warnflag"]:
            print "Warning flag raised"
        # Return optimised value of theta
        return theta


def func(nu, t, sigma):
    return 5*np.sinc(nu)*5*np.sinc(t) + sigma*np.random.randn(int(np.sqrt(nu.size)), int(np.sqrt(t.size)))

if __name__ == "__main__":
    from GP.operators import covariance_ops
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

    # create GP object
    print "Doing unweighted GP"
    GP = KronGP(x, xp, y)

    # set hypers
    sigmaf = 1.0
    l1 = 1.0
    l2 = 1.0
    sigman = 0.5
    theta0 = np.array([sigmaf, l1, l2, sigman])

    #K = covariance_ops.K_op(x, theta0, kernels=["sqexp", "sqexp"])



    theta = GP.train(theta0)

    print theta

    print "Training"
    fmean = GP.meanf(theta).reshape(nunup.shape)

    # do for transformed data
    N = Nnu*Nt
    Sigmay = 0.1 * np.ones([Nnu, Nt]) + np.abs(0.1 * np.random.randn(Nnu, Nt))

    #print "Simulating weighted noise"
    #Noise = np.reshape(np.random.multivariate_normal(np.zeros(N), np.diag(Sigmay.flatten())), (Nnu, Nt))

    # y2 = func(nunu, tt, Sigmay)
    #
    # print "Doing weighted GP"
    # GP = KronGP(x, xp, y, Sigmay=Sigmay)
    #
    # print "Training"
    # theta2 = GP.train_transformed(theta0)
    #
    # fmean2 = GP.meanft(theta).reshape(nunup.shape)
    #
    # print fmean2/fmean



    # plot expected result
    import matplotlib.pyplot as plt
    plt.figure('Data')
    plt.imshow(y)
    plt.colorbar()
    plt.figure('True')
    plt.imshow(fp)
    plt.colorbar()
    plt.figure('fp')
    plt.imshow(fmean)
    plt.colorbar()
    # plt.figure('Data2')
    # plt.imshow(GP.ytransformed)
    # plt.colorbar()
    # plt.figure('fp2')
    # plt.imshow(fmean2)
    # plt.colorbar()
    plt.show()



