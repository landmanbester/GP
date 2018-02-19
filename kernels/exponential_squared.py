"""
The exponential squared covariance function, its derivative and spectral density
"""

import numpy as np
from GP.tools import abs_diff
from scipy.sparse import linalg as ssl
from scipy.sparse import diags
from GP.tools import kronecker_tools as kt
from GP.tools import FFT_tools as ft
import pyfftw

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
    def __init__(self, x, theta):
        """
        :param x: the inputs (coordinates at which to evaluate the covariance function 
        :param theta: the initial hyper-parameters 
        """
        self.N = x.size
        self.x = np.reshape(x, (self.N, 1))
        self.theta = theta

        # get the first row of the diff square matrix
        self.r = np.abs(np.tile(x[0], self.N) - x)

        # compute covariance function at these locations
        K1 = self.cov_func(self.theta, self.r)

        # broadcast to circulant form
        C = np.append(K1, K1[np.arange(self.N)[1:-1][::-1]].conj())

        # set FFT objects
        self.FFT = pyfftw.builders.rfft
        self.iFFT = pyfftw.builders.irfft
        self.FFTn = pyfftw.builders.rfftn
        self.iFFTn = pyfftw.builders.irfftn

        # take the fft (only covariance info we need to store
        self.Chat = self.FFT(C)()  # this is all we really need to store for covariance function
        self.N2 = 2 * self.N - 2  # the lengt of the broadcasted vector

        # set up for inverse
        self.FFT2 = pyfftw.builders.fft
        self.iFFT2 = pyfftw.builders.ifft
        self.Chat2 = self.FFT2(C)().real
        self.Chat2[0] + 1e-13 # jitter required for inverse


        # set mandatory attributes for LinearOperator class
        self.shape = (self.N, self.N)
        self.dtype = np.float64

        # still figuring it out below this line
        # # set basis functions required for preconditioning
        # self.M = np.array([12])
        # self.L = np.array([2.5])
        # from GP.tools import make_basis
        # from GP.basisfuncs import rectangular
        # self.Phi = make_basis.get_eigenvectors(self.x, self.M, self.L, rectangular.phi)
        # self.Lambda = make_basis.get_eigenvals(self.M, self.L, rectangular.Lambda)
        # self.s = np.sqrt(self.Lambda)
        # self.set_preconds()


    def _matvec(self, x):
        x = self.FFT(x, self.N2)()
        x *= self.Chat
        return self.iFFT(x)()[0:self.N]

    def idot(self, x):
        # uses FFT to do inverse multiplication (an approximation for small data sets?)
        xhat = self.FFT2(x, self.N2)()
        y = xhat/self.Chat2
        return self.iFFT2(y)()[0:self.N].real

    def _matmat(self, x):
        x = self.FFTn(x, s=(self.N2,), axes=(0,))()
        x *= self.Chat[:, None]
        return self.iFFTn(x, s=(self.N2,), axes=(0,))()[0:self.N, :]

    def update_theta(self, theta):
        self.theta = theta
        K1 = self.cov_func(self.theta, self.r)
        C = np.append(K1, K1[np.arange(self.N)[1:-1][::-1]].conj())
        self.Chat = self.FFT(C)()

    # def set_preconds(self):
    #     self.S = self.spectral_density(self.theta, self.s)
    #     if self.grid_regular:
    #         self.PhiTPhi = np.einsum('ij,ji->i', self.Phi.T, self.Phi)  # computes only the diagonal entries
    #         self.Z = self.PhiTPhi + self.theta[2] ** 2 / self.S
    #     else:
    #         self.PhiTPhi = np.dot(self.Phi.T, self.Phi)
    #         self.Z = self.PhiTPhi + theta[2] ** 2 * np.diag(1.0 / self.S)
    #
    # def get_Lambda_Phi(self, x):
    #     return


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
        return np.sqrt(2*np.pi)*theta[0]**2.0*theta[1]**2*np.exp(-theta[1]**2*s**2/2.0)

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
    l = 0.0000001
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

    print "Starting"
    res1 = Kop(t)
    print "Done"

    res2 = Kmat.dot(t)

    print "matvec diff = ", np.abs(res1 - res2).max()

    #res1 = np.sort(Kop.idot(t))[::-1]
    res1 = Kop.idot(t)
    #res2 = np.sort(np.linalg.solve(Kmat + 1e-13*np.eye(N), t))[::-1]
    res2 = np.linalg.solve(Kmat + 1e-13 * np.eye(N), t)

    import matplotlib.pyplot as plt
    plt.figure('inv')
    plt.plot(res1, 'bx')
    plt.plot(res2, 'rx')
    plt.show()


    print "idot diff = ", np.abs(res1 - res2).max()

    # test matmat
    tt = np.random.randn(N, 10)

    res1 = Kop._matmat(tt)

    res2 = Kmat.dot(tt)

    print "matmat diff = ", np.abs(res1 - res2).max()





