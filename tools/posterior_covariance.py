"""
Computes the posterior covariance matrix of the Gaussian process with specified hypers
"""

import numpy as np

class covf(object):
    def __init__(self, kernel, mode="Full", XX=None, XXp=None, XXpp=None, Phi=None, Phip=None, s=None,
                 grid="Regular"):
        """
        :param kernel: a class holding the covariance function and spectral densities 
        :param mode: whether to use full or RR GPR
        :param XX: absolute differences of inputs
        :param XXp: absolute differences between targets and inputs 
        :param XXpp: absolute differences between targets
        :param Phi: basis functions evaluated at x
        :param Phip: basis functions evaluated at xp
        :param s: square root of eigenvalues
        :param grid: whether x is on a regular grid or not
        """
        self.kernel = kernel
        if self.mode=="Full":
            self.XX = XX
            self.XXp = XXp
            self.XXpp = XXpp
        elif self.mode=="RR":
            self.grid = grid
            self.Phi = Phi
            if self.grid=="Regular": # save only the diagonal if on regular grid
                self.PhiTPhi = np.diag(np.dot(Phi.T, Phi))
            else:
                self.PhiTPhi = np.dot(Phi.T, Phi)
            self.Phip = Phip
            self.s = s

    def give_covaraince(self, theta):
        """
        :param theta: hypers
        """
        if self.mode == "Full":
            Kp = self.kernel.cov_func(theta, self.XXp, mode="nn")
            Kpp = self.kernel.cov_func(theta, self.XXpp, mode="nn")
            Ky = self.kernel.cov_func(theta, self.XX, mode="Noise")
            L = np.linalg.cholesky(Ky)
            Linv = np.linalg.inv(L)
            LinvKp = np.dot(Linv, Kp)
            return Kpp - np.dot(LinvKp.T, LinvKp)
        elif self.mode == "RR":
            return


    def give_RR_covcoeffs(self, theta):
        """
        :param theta: hypers 
        """
