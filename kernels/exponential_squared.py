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
        # shift the coordinate to origin (this is for effective preconditioning)
        xshift = x.min() + (x.max() - x.min())/2.0
        self.x = np.reshape(x - xshift, (self.N, 1))
        self.theta = theta

        # get the first row of the diff square matrix
        self.r = np.abs(np.tile(x[0], self.N) - x)

        # compute covariance function at these locations
        K1 = self.cov_func(self.theta, self.r)

        self.K1 = K1

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

        # set mandatory attributes for LinearOperator class
        self.shape = (self.N, self.N)
        self.dtype = np.float64

        # set up for inverse
        self.FFT2 = pyfftw.builders.fft
        self.iFFT2 = pyfftw.builders.ifft
        self.Chat2 = self.FFT2(C)().real + theta[-1]**2  # adding diagonal jitter is the same as adding a constant to
                                                    # the Fourier modes. Should be adding the nugget here

        # set basis functions required for preconditioning
        self.M = np.array([7])
        self.L = np.array([np.maximum(1.75*self.x.max(), 1.05*self.theta[1])])
        from GP.tools import make_basis
        from GP.basisfuncs import rectangular
        self.Phi = make_basis.get_eigenvectors(self.x, self.M, self.L, rectangular.phi)  # does not depend on theta
        self.Lambda = make_basis.get_eigenvals(self.M, self.L, rectangular.Lambda)  # do not confuse with Lambdas in K operator, no dependence on theta
        self.s = np.sqrt(self.Lambda)  # does not depend on theta
        self.S = self.spectral_density(self.theta, self.s)  # this becomes Lambdas in K operator, depends on theta

    def _matvec(self, x):
        x = self.FFT(x, self.N2)()
        x *= self.Chat
        return self.iFFT(x)()[0:self.N]

    # def idot(self, x):
    #     # uses FFT to do inverse multiplication (only works if length scale much smaller than observation length)
    #     xhat = self.FFT2(x, self.N2)()
    #     y = xhat/self.Chat2
    #     return self.iFFT2(y)()[0:self.N].real

    def _matmat(self, x):
        x = self.FFTn(x, s=(self.N2,), axes=(0,))()
        x *= self.Chat[:, None]
        return self.iFFTn(x, s=(self.N2,), axes=(0,))()[0:self.N, :]

    def update_theta(self, theta):
        self.theta = theta
        K1 = self.cov_func(self.theta, self.r)
        C = np.append(K1, K1[np.arange(self.N)[1:-1][::-1]].conj())
        self.Chat = self.FFT(C)()
        self.Chat2 = self.FFT2(C)().real + 1.0e-13
        self.S = self.spectral_density(self.theta, self.s)

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

def RRlogdet(N, sigman, pspec):
    Z = sigman**2/pspec + np.ones(N)
    logdetZ = np.sum(np.log(Z))
    logpspec = np.sum(np.log(pspec))
    return logdetZ + logpspec #+ term1

def give_roots_of_unity(N):
    n = np.arange(N)
    return np.exp(2.0j*np.pi*n/N)

if __name__=="__main__":
    N = 1024*8
    x = np.linspace(-100, 100, N)
    sigmaf = 5.0e0
    l = 1e1
    sigman = 1e-4
    theta = np.array([sigmaf, l, sigman])

    # get operator representation
    Kop = sqexp_op(x, theta)

    # get dense representation
    kernel = sqexp()
    xx = abs_diff.abs_diff(x, x)
    Kmat = kernel.cov_func(theta, xx, noise=False)

    # test matvec
    t = np.random.randn(N)

    # print "Starting"
    # res1 = Kop(t)
    # print "Done"
    #
    # res2 = Kmat.dot(t)

    # print "matvec diff = ", np.abs(res1 - res2).max()

    # res1 = np.sort(Kop.idot(t))[::-1]
    # res1 = Kop.idot(t)
    # res2 = np.sort(np.linalg.solve(Kmat + 1e-13*np.eye(N), t))[::-1]
    # res2 = np.linalg.solve(Kmat + 1e-13*np.eye(N), t)

    import matplotlib.pyplot as plt
    # plt.figure('inv')
    # plt.plot(res1, 'bx')
    # plt.plot(res2, 'rx')
    # plt.show()

    # test pspec eigenspectrum and determinant
    freqs = (np.linspace(0, 2*np.pi, N)*np.arange(N))**2
    delx = Kop.K1[1] - Kop.K1[0]
    omega = (2*np.pi*np.fft.fftfreq(N, delx))**2
    Lambda = Kop.s

    pspec = Kop.spectral_density(theta, freqs) + 1e-15
    pspec2 = Kop.spectral_density(theta, omega) + 1e-15
    eigs = np.sort(Kop.Chat2)[::-1][0:N].real*N/(2*N - 2)
    eigs2 = np.sort(Kop.Chat2 - theta[-1]**2)[::-1][0:N].real*N/(2*N - 2)
    pspec3 = Kop.spectral_density(theta, np.sort(np.sqrt(eigs2) + 1e-15))
    pspec4 = Kop.spectral_density(theta, Lambda) + 1e-15

    # test logdet
    Ky = Kmat + sigman**2*np.eye(N)
    s, logdet = np.linalg.slogdet(Ky)
    logdet *= s

    logdeteigs = np.sum(np.log(eigs))

    logdetRR = RRlogdet(N, sigman, pspec)
    logdetRR2 = RRlogdet(N, sigman, pspec2)
    logdetRR3 = RRlogdet(N, sigman, pspec3)
    logdetRR4 = RRlogdet(N, sigman, pspec4)

    logdetdiag = np.sum(np.log(np.diag(Ky)))

    teff = (x[-1]-x[0])/l
    seff = sigmaf/l
    SNR = sigmaf/sigman

    print "logdet = ", logdet, "teff = ", teff, "seff = ", seff, "SNR = ", SNR
    print "eig logdet = ", logdeteigs, np.abs(logdet - logdeteigs)/np.abs(logdet)
    print "RR logdet = ", logdetRR, np.abs(logdet - logdetRR)/np.abs(logdet)
    print "RR2 logdet = ", logdetRR2, np.abs(logdet - logdetRR2) / np.abs(logdet)
    print "RR3 logdet = ", logdetRR3, np.abs(logdet - logdetRR3) / np.abs(logdet)
    print "RR4 logdet = ", logdetRR4, np.abs(logdet - logdetRR4) / np.abs(logdet)
    print "diag logdet = ", logdetdiag, np.abs(logdet - logdetdiag)/np.abs(logdet)







