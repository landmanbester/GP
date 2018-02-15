"""
The exponential squared covariance function, its derivative and spectral density
"""

import numpy as np
from GP.tools import abs_diff
from scipy.sparse import linalg as ssl
from scipy.sparse import diags
from GP.tools import kronecker_tools as kt
from GP.tools import FFT_tools as ft

class sqexp(object):
    def __init__(self, p=None, D=1, Sigmay=None, mode=None):
        # p is not relevant to sq exp
        # Check that D is an integer
        if not isinstance(D, (int, long)):
            print "Converting D to integer"
            D = int(D)
        self.D = D
        if Sigmay is not None:
            self.Sigmay = Sigmay
        else:
            self.Sigmay = None
        self.mode = mode


    def cov_func(self, theta, x, noise=True, i_dim=None):
        """
        Covariance function possibly including noise variance
        :param theta: hypers
        :param x: the inputs x[i] - x[j] (usually matrix of differences)
        :param noise: whether to add noise or not
        """
        if noise:
            if self.Sigmay is not None:
                if i_dim is None:
                    return theta[0] ** 2.0 * np.exp(-x ** 2.0 / (2.0 * theta[1] ** 2.0)) + theta[2] ** 2.0 * self.Sigmay
                else:
                    return theta[0] ** 2.0 * np.exp(-x ** 2.0 / (2.0 * theta[1] ** 2.0)) + theta[2] ** 2.0 * self.Sigmay[i_dim]
            else:
                return theta[0] ** 2.0 * np.exp(-x ** 2.0 / (2.0 * theta[1] ** 2.0)) + theta[2] ** 2.0 * np.eye(x.shape[0])
        else:
            return theta[0] ** 2.0 * np.exp(-x ** 2.0 / (2.0 * theta[1] ** 2.0))

    def dcov_func(self, theta, x, K, mode=None):
        """
        Derivates of the covariance function w.r.t. the hyperparameters
        :param theta: hypers
        :param x: the inputs x[i] - x[j] (usually matrix of differences)
        :param mode: use to determine which hyper we are taking the derivative w.r.t.
        """
        if mode is not None:
            if mode == "sigmaf":
                return 2 * K / theta[0]
            elif mode == "l":
                return x ** 2 * K / theta[1] ** 3
            elif mode == "sigman":
                if self.Sigmay is not None:
                    if self.mode == "kron":
                        return 2 * theta[2] * self.Sigmay
                    else:
                        return 2 * theta[2] * np.diag(self.Sigmay)
                else:
                    if self.mode == "kron":
                        return 2 * theta[2] * kt.kron_collapse_shape(x)  # broadcasting to shape of Sigmay, boradcast to shape eye(N) using Kronecker tools
                    else:
                        return 2 * theta[2] * np.eye(x.shape[0])
        else:
            raise Exception('You have to specify a mode to evaluate dcov_func')

    def spectral_density(self, theta, s):
        """
        The spectral density of the squared exponential covariance function
        :param theta: hypers
        :param s: Fourier dual of x
        """
        return np.sqrt(2*np.pi)*theta[0]**2.0*(theta[1]**2)**(self.D/2.0)*np.exp(-theta[1]**2*s**2/2.0)

    def dspectral_density(self, theta, S, s, mode=0):
        """
        Derivative of the spectral density corresponding to the squared exponential covariance function
        :param theta: hypers
        :param S: value of spectral density
        :param s: Fourier dual of x
        :param mode: use to determine which hyper we are taking the derivative w.r.t.
        """
        if mode == 0:
            return 2 * S / theta[0]
        elif mode == 1:
            return self.D*S/theta[1] - s**2*theta[1]*S

class sqexp_op(ssl.LinearOperator):
    """
    This is the linear operator representation of a covariance matrix defined on a regular grid
    """
    def __init__(self, x, theta, grid_regular=True):
        """
        :param x: the inputs (coordinates at which to evaluate the covariance function 
        :param theta: the initial hyper-parameters 
        """
        self.N = x.size
        self.x = x
        self.theta = theta
        self.grid_regular = grid_regular
        if self.grid_regular:
            # get the first row of the diff square matrix
            self.r = np.abs(np.tile(x[0], self.N) - x)
            # compute covariance function at these locations
            self.K1 = self.cov_func(self.theta, self.r)
        else:
            self.XX = abs_diff.abs_diff(self.x, self.x)
            self.K = self.cov_func(self.theta, self.XX)

        # # set basis functions required for preconditioning
        # self.M = np.array([20])
        # self.L = np.array([15])
        # from GP.tools import make_basis
        # from GP.basisfuncs import rectangular
        # self.Phi = make_basis.get_eigenvectors(self.x, self.M, self.L, rectangular.phi)
        # self.Lambda = make_basis.get_eigenvals(self.M, self.L, rectangular.Lambda)
        # self.s = np.sqrt(self.Lambda)
        # self.set_preconds()
        # set mandatory attributes for LinearOperator class
        self.shape = (self.N, self.N)
        self.dtype = np.float64

    def _matvec(self, x):
        if self.grid_regular:
            return ft.FFT_toepvec(self.K1, x)
        else:
            return self.K.dot(x)

    def _matmat(self, x):
        if self.grid_regular:
            return ft.FFT_toepmat(self.K1, x)
        else:
            return self.K.dot(x)

    def set_preconds(self):
        self.S = self.spectral_density(self.theta, self.s)
        if self.grid_regular:
            self.PhiTPhi = np.einsum('ij,ji->i', self.Phi.T, self.Phi)  # computes only the diagonal entries
            self.Z = self.PhiTPhi + theta[2] ** 2 / self.S
        else:
            self.PhiTPhi = np.dot(self.Phi.T, self.Phi)
            self.Z = self.PhiTPhi + theta[2] ** 2 * np.diag(1.0 / self.S)

    def _Mop(self, x):
        return


    def cov_func(self, theta, x):
        """
        Covariance function possibly including noise variance
        :param theta: hypers
        :param x: the inputs x[i] - x[j] (usually matrix of differences)
        :param noise: whether to add noise or not
        """
        return theta[0] ** 2.0 * np.exp(-x ** 2.0 / (2.0 * theta[1] ** 2.0))

    def dcov_func(self, theta, x, K, mode=None):
        """
        Derivates of the covariance function w.r.t. the hyperparameters
        :param theta: hypers
        :param x: the inputs x[i] - x[j] (usually matrix of differences)
        :param mode: use to determine which hyper we are taking the derivative w.r.t.
        """
        if mode == "sigmaf":
            return 2 * K / theta[0]
        elif mode == "l":
            return x ** 2 * K / theta[1] ** 3
        elif mode == "sigman":
            if self.Sigmay is not None:
                if self.mode == "kron":
                    return 2 * theta[2] * self.Sigmay
                else:
                    return 2 * theta[2] * np.diag(self.Sigmay)
            else:
                if self.mode == "kron":
                    return 2 * theta[2] * kt.kron_collapse_shape(x)  # broadcasting to shape of Sigmay, boradcast to shape eye(N) using Kronecker tools
                else:
                    return 2 * theta[2] * np.eye(x.shape[0])

    def spectral_density(self, theta, s):
        """
        The spectral density of the squared exponential covariance function
        :param theta: hypers
        :param s: Fourier dual of x
        """
        return np.sqrt(2*np.pi)*theta[0]**2.0*(theta[1]**2)**(self.D/2.0)*np.exp(-theta[1]**2*s**2/2.0)

    def dspectral_density(self, theta, S, s, mode=0):
        """
        Derivative of the spectral density corresponding to the squared exponential covariance function
        :param theta: hypers
        :param S: value of spectral density
        :param s: Fourier dual of x
        :param mode: use to determine which hyper we are taking the derivative w.r.t.
        """
        if mode == 0:
            return 2 * S / theta[0]
        elif mode == 1:
            return self.D*S/theta[1] - s**2*theta[1]*S

if __name__=="__main__":
    N = 1024
    x = np.linspace(-10, 10, N)
    sigmaf = 1.0
    l = 1.0
    sigman = 0.1
    theta = np.array([sigmaf, l, sigman])

    # get operator representation
    Kop = sqexp_op(x, theta)

    # get dense representation
    kernel = sqexp()
    xx = abs_diff.abs_diff(x, x)
    Kmat = kernel.cov_func(theta, xx, noise=False)

    # test matvec
    t = np.random.randn(N)

    res1 = Kop(t)

    res2 = Kmat.dot(t)

    print "matvec diff = ", np.abs(res1 - res2).max()

    # test matmat
    tt = np.random.randn(N, 10)

    res1 = Kop._matmat(tt)

    res2 = Kmat.dot(tt)

    print "matmat diff = ", np.abs(res1 - res2).max()





