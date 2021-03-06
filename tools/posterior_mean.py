"""
Computes the posterior mean function of the Gaussian process with specified hypers and assuming that the targets don't
change. 
TODO: Generalise for arbitrary targets?
"""

import numpy as np
from GP.tools import kronecker_tools as kt

class meanf(object):
    def __init__(self, x, xp, y, kernel, Sigmay=None, mode="Full", prior_mean=None, XX=None, XXp=None, Phi=None, Phip=None, \
                 PhiTPhi = None, s=None, grid_regular=False):
        """
        :param x: N x D vector of inputs
        :param xp: Np x D vector of targets
        :param y: the data
        :param kernel: a class holding the covariance function and spectral densities 
        :param mode: whether to use full or RR GPR
        :param prior_mean: callable prior mean function
        :param XX: absolute differences of inputs
        :param XXp: absolute differences between targets and inputs 
        :param Phi: basis functions evaluated at x
        :param Phip: basis functions evaluated at xp
        :param PhiTPhi: the product dot(Phi.T, Phi)
        :param s: square root of eigenvalues
        :param grid_regular: whether x is on a regular grid or not
        """
        self.x = x
        self.xp = xp
        self.mode = mode
        if self.mode != "kron":
            self.N = self.x.shape[0]
            self.Np = xp.shape[0]
        else:
            self.N = kt.kron_N(self.x)
            self.Np = kt.kron_N(self.xp)

        self.kernel = kernel
        self.mode = mode
        self.Sigmay = Sigmay
        if prior_mean is None:
            self.yp = np.zeros([self.N, 1])
            self.fp = np.zeros([self.Np, 1])
        else:
            self.yp = (prior_mean(x)).reshape(self.N, 1)
            self.fp = (prior_mean(xp)).reshape(self.Np, 1)
        self.yDat = y.reshape(self.N, 1) - self.yp
        if self.mode == "Full" or self.mode == "kron":
            self.XX = XX
            self.XXp = XXp
        elif self.mode == "RR":
            self.grid_regular = grid_regular
            self.Phi = Phi
            self.PhiTPhi = PhiTPhi
            self.Phip = Phip
            self.s = s

    def give_mean(self, theta):
        """
        Computes the posterior mean function
        :param theta: hypers
        """
        if self.mode == "Full":
            Ky = self.kernel.cov_func(theta, self.XX, noise=True)
            Kp = self.kernel.cov_func(theta, self.XXp, noise=False)
            L = np.linalg.cholesky(Ky)
            Linv = np.linalg.inv(L)
            LinvKp = np.dot(Linv, Kp)
            return self.fp + np.dot(LinvKp.T, np.dot(Linv, self.yDat))
        elif self.mode == "RR":
            fcoeffs = self.give_RR_coeffs(theta)
            return self.fp + np.dot(self.Phip, fcoeffs)
        elif self.mode == "kron":
            D = self.XX.shape[0]
            # broadcast theta (sigmaf and sigman is a shared hyperparameter but easiest to deal with this way)
            thetan = np.zeros([D, 3])
            thetan[:, 0] = theta[0]
            thetan[:, -1] = theta[-1]
            for i in xrange(D): # this is how many length scales will be involved in te problem
                thetan[i, 1] = theta[i+1] # assuming we set up the theta vector as [[sigmaf, l_1, sigman], [sigmaf, l_2, ..., sigman]]
            # get the Kronecker matrices
            Kp = []
            K = []
            for i in xrange(D):
                Kp.append((self.kernel[i].cov_func(thetan[i], self.XXp[i], noise=False)).T)
                K.append(self.kernel[i].cov_func(thetan[i], self.XX[i], noise=False))
            Kp = np.asarray(Kp)
            K = np.asarray(K)
            Lambdas, Qs = kt.kron_eig(K)
            QsT = kt.kron_transpose(Qs)
            alpha = kt.kron_matvec(QsT[::-1], self.yDat)
            Lambda = kt.kron_diag(Lambdas)
            if self.Sigmay is not None:
                Lambda += theta[-1]**2*kt.kron_diag(self.Sigmay)  # absorb weights into Lambdas
            else:
                Lambda += theta[-1] ** 2 * np.ones(self.N)
            alpha = alpha/Lambda  # dot with diagonal inverse
            alpha = kt.kron_matvec(Qs[::-1], alpha)
            return kt.kron_tensorvec(Kp[::-1], alpha)
        else:
            raise Exception('Mode %s not supported yet'%self.mode)

    def give_RR_coeffs(self, theta):
        """
        Computes the coefficients of the posterior mean function
        :param theta: hypers
        :return fcoeffs: the coefficients of the posterior mean function
        """
        S = self.kernel.spectral_density(theta, self.s)
        if self.grid_regular:  # if on a regular grid PhiTPhi (and therefore Z) is diagonal
            Z = self.PhiTPhi + theta[2] ** 2 / S
            L = np.sqrt(Z)
            Linv = np.diag(1.0 / L)
        else:
            Z = self.PhiTPhi + theta[2] ** 2 * np.diag(1.0 / S)
            L = np.linalg.cholesky(Z)
            Linv = np.linalg.inv(L)
        return np.dot(Linv.T, np.dot(Linv, np.dot(self.Phi.T, self.yDat)))