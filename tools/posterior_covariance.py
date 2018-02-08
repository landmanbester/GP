"""
Computes the posterior covariance matrix of the Gaussian process with specified hypers
"""

import numpy as np
from GP.tools import kronecker_tools as kt

class covf(object):
    def __init__(self, kernel, Sigmay=None, mode="Full", XX=None, XXp=None, XXpp=None, Phi=None, Phip=None, s=None,
                 PhiTPhi=None, grid_regular=False):
        """
        :param kernel: a class holding the covariance function and spectral densities 
        :param mode: whether to use full or RR GPR
        :param XX: absolute differences of inputs
        :param XXp: absolute differences between targets and inputs 
        :param XXpp: absolute differences between targets
        :param Phi: basis functions evaluated at x
        :param Phip: basis functions evaluated at xp
        :param PhiTPhi: the product dot(Phi.T, Phi)
        :param s: square root of eigenvalues
        :param grid: whether x is on a regular grid or not
        """
        self.kernel = kernel
        self.mode = mode
        self.Sigmay = Sigmay

        if self.mode == "Full" or self.mode == "kron":
            self.XX = XX
            self.XXp = XXp
            self.XXpp = XXpp
        elif self.mode == "RR":
            self.Phi = Phi
            self.grid_regular = grid_regular
            self.PhiTPhi = PhiTPhi
            self.Phip = Phip
            self.s = s

    def give_covariance(self, theta):
        """
        :param theta: hypers
        """
        if self.mode == "Full":
            Kp = self.kernel.cov_func(theta, self.XXp, noise=False)
            Kpp = self.kernel.cov_func(theta, self.XXpp, noise=False)
            Ky = self.kernel.cov_func(theta, self.XX, noise=True)
            L = np.linalg.cholesky(Ky)
            Linv = np.linalg.inv(L)
            LinvKp = np.dot(Linv, Kp)
            return Kpp - np.dot(LinvKp.T, LinvKp)
        elif self.mode == "RR":
            coeffs = self.give_RR_covcoeffs(theta)
            return np.dot(self.Phip, np.dot(coeffs, self.Phip.T))
        elif self.mode == "Kron":
            # get the Kronecker matrices
            D = self.XX.shape[0]
            Kp = []
            Kpp = []
            K = []
            for i in xrange(D):
                Kp.append(self.kernel[0].cov_func(theta[i], self.XXp[i], noise=False))
                Kpp.append(self.kernel[0].cov_func(theta[i], self.XXpp[i], noise=False))
                K.append(self.kernel[0].cov_func(theta[i], self.XX[i], noise=False))
            Kp = np.asarray(Kp)
            Kpp = np.asarray(Kpp)
            K = np.asarray(K)
            Qs, Lambdas = kt.kron_eig(K)
            QsT = kt.kron_transpose(Qs)
            KpT = kt.kron_transpose(Kp)
            A = kt.kron_matmat(QsT[::-1], KpT)
            Lambda = kt.kron_diag(Lambdas)
            if self.Sigmay is not None:
                Lambda += self.Sigmay
            A = A/Lambda[:, None]  # dot with diagonal inverse
            A = kt.kron_matmat(Qs[::-1], A)
            return kt.kron_kron(Kpp) - kt.kron_kron(Kp).dot(A)






        else:
            raise Exception('Mode %s not supported yet'%self.mode)


    def give_RR_covcoeffs(self, theta):
        """
        :param theta: hypers 
        """
        S = self.kernel.spectral_density(theta, self.s)
        if self.grid_regular:
            Z = self.PhiTPhi + theta[2] ** 2 / S
            L = np.sqrt(Z)
            Linv = np.diag(1.0 / L)
        else:
            Z = self.PhiTPhi + theta[2] ** 2 * np.diag(1.0 / S)
            L = np.linalg.cholesky(Z)
            Linv = np.linalg.inv(L)
        return theta[2] ** 2 * np.dot(Linv.T, Linv)