"""
The exponential squared covariance function, its derivative and spectral density
"""

import numpy as np
from GP.tools import abs_diff
from scipy.sparse import linalg as ssl
from scipy.sparse import diags
from GP.tools import kronecker_tools as kt
from GP.tools import FFT_tools as ft
try:
    import pyfftw
except:
    pass

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
    def __init__(self, x, theta, Nrem, wisdom_file=None, reset_wisdom=False):
        """
        :param x: the inputs (coordinates at which to evaluate the covariance function 
        :param theta: the initial hyper-parameters 
        """
        self.N = x.size
        self.N2 = 2 * self.N - 2  # the lengt of the broadcasted vector
        self.Nrem = Nrem
        # shift the coordinate to origin (this is for effective preconditioning)
        xshift = x.min() + (x.max() - x.min())/2.0
        self.x = np.reshape(x - xshift, (self.N, 1))
        self.theta = theta

        # get the first row of the diff square matrix
        self.r = np.abs(np.tile(x[0], self.N) - x)

        # compute covariance function at these locations
        K1 = self.cov_func(self.theta, self.r)

        # try to load in the wisdom file if it exists
        if wisdom_file is not None:
            try:
                wisdom = np.load(wisdom_file)
                pyfftw.import_wisdom(wisdom)
                self.has_wisdom = True
                print "FFT wisdom set"
            except:
                self.has_wisdom = False

        # set up a byte aligned array to store input and output of FFT on first row of Covariance matrix
        self.Chat = pyfftw.empty_aligned(self.N2, dtype='complex128')
        self.FFTC = pyfftw.FFTW(self.Chat, self.Chat, direction='FFTW_FORWARD', threads=4)

        # broadcast to circulant form
        self.Chat[:] = np.append(K1, K1[np.arange(self.N)[1:-1][::-1]].conj())

        # this should write the result to Chat automatically
        self.FFTC()

        # set up general purpose 1D FFT schemes
        self.xhat = pyfftw.empty_aligned(self.N2, dtype='complex128')
        # forward transform
        self.FFT = pyfftw.FFTW(self.xhat, self.xhat, direction='FFTW_FORWARD', threads=4)
        # backward transform
        self.iFFT = pyfftw.FFTW(self.xhat, self.xhat, direction='FFTW_BACKWARD', threads=4)

        # set up general ND purpose FFT schemes
        self.Xhat = pyfftw.empty_aligned((self.N2, self.Nrem), dtype='complex128')
        self.FFTn = pyfftw.FFTW(self.Xhat, self.Xhat, axes=(0,), direction='FFTW_FORWARD', threads=4)
        self.iFFTn = pyfftw.FFTW(self.Xhat, self.Xhat, axes=(0,), direction='FFTW_BACKWARD', threads=4)

        if reset_wisdom and wisdom_file is not None:
            # run these to learn the wisdom
            self.FFT()
            self.iFFT()
            # self.FFTn()
            # self.iFFTn()
            wisdom = pyfftw.export_wisdom()
            np.save(wisdom_file, wisdom)
            print "FFT wisdom saved"

        # set mandatory attributes for LinearOperator class
        self.shape = (self.N, self.N)
        self.dtype = np.float64

        self.has_eig_vecs = False

    def _matvec(self, x):
        self.xhat[0:self.N] = x
        self.xhat[self.N::] = 0.0
        self.FFT()
        self.xhat *= self.Chat
        self.iFFT()
        return self.xhat[0:self.N]

    # def idot(self, x):
    #     # uses FFT to do inverse multiplication (only works if length scale much smaller than observation length)
    #     xhat = self.FFT2(x, self.N2)()
    #     y = xhat/self.Chat2
    #     return self.iFFT2(y)()[0:self.N].real

    def set_eigs(self, nugget=None):
        if not self.has_eig_vecs:
            XX = abs_diff.abs_diff(self.x, self.x)
            Kmat = self.cov_func(self.theta, XX)
            self.Lambda_full, self.Q = np.linalg.eigh(Kmat)  # add jitter for stability
            self.has_eig_vecs = True
            if (self.Lambda_full < 0.0).any():  # Hermitian matrices have positive eigenvals
                I = np.argwhere(self.Lambda_full <= 0.0).squeeze()
                self.Lambda_full[I] = 0.0
                if nugget is not None:
                    self.Lambda_full += nugget
        else:
            XX = abs_diff.abs_diff(self.x, self.x)
            Kmat = self.cov_func(self.theta, XX)
            self.Lambda_full = np.linalg.eigvalsh(Kmat)
            if (self.Lambda_full < 0.0).any():  # Hermitian matrices have positive eigenvals
                I = np.argwhere(self.Lambda_full <= 0.0).squeeze()
                self.Lambda_full[I] = 0.0
                if nugget is not None:
                    self.Lambda_full += nugget

    def set_RR_eigs(self, nugget=None):
        # set basis functions required for preconditioning
        self.M = np.array([7])
        self.L = np.array([np.maximum(1.75*self.x.max(), 1.05*self.theta[1])])
        from GP.tools import make_basis
        from GP.basisfuncs import rectangular
        self.Phi = make_basis.get_eigenvectors(self.x, self.M, self.L, rectangular.phi)  # does not depend on theta
        self.Lambda = make_basis.get_eigenvals(self.M, self.L, rectangular.Lambda)  # do not confuse with Lambdas in K operator, no dependence on theta
        self.s = np.sqrt(self.Lambda)  # does not depend on theta
        self.S = self.spectral_density(self.theta, self.s)  # this becomes Lambdas in K operator, depends on theta


    def idot_full(self, x):
        # computes inverse multiply directly using eigen-decomposition
        x = self.Q.T.dot(x)
        x /= self.Lambda_full  # diag dot with inverse using vector representation
        return self.Q.dot(x)

    def _matmat(self, x):
        self.Xhat[0:self.N, :] = x
        self.Xhat[self.N::, :] = 0.0
        self.FFTn()
        self.Xhat *= self.Chat[:, None]
        self.iFFTn()
        return self.Xhat[0:self.N, :]

    def update_theta(self, theta):
        self.theta = theta
        K1 = self.cov_func(self.theta, self.r)
        self.Chat[:] = np.append(K1, K1[np.arange(self.N)[1:-1][::-1]].conj())
        self.FFTC()

        # self.S = self.spectral_density(self.theta, self.s)

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
    N = 2**8
    x = np.linspace(-100, 100, N)
    sigmaf = 1.0e0
    l = 5e0
    sigman = 1e-2
    theta = np.array([sigmaf, l, sigman])

    # get operator representation
    Kop = sqexp_op(x, theta, 20, wisdom_file='/home/landman/Projects/GP/fft_wisdom/test.wisdom.npy')

    # get dense representation
    kernel = sqexp()
    xx = abs_diff.abs_diff(x, x)
    Kmat = kernel.cov_func(theta, xx, noise=False)
    Kfunc = kernel.cov_func(theta, xx[0,1::], noise=False)

    # test matvec
    t = np.random.randn(N)

    res1 = Kop(t)

    res2 = Kmat.dot(t)

    print "matvec diff = ", np.abs(res1 - res2).max()

    t2 = np.random.randn(N)

    res1 = Kop(t2)

    res2 = Kmat.dot(t2)

    print "matvec diff = ", np.abs(res1 - res2).max()

    T = np.random.randn(N, 20)

    res1 = Kop.matmat(T)

    res2 = Kmat.dot(T)

    print "matmat diff = ", np.abs(res1 - res2).max()

    T2 = np.random.randn(N, 20)

    res1 = Kop.matmat(T2)

    res2 = Kmat.dot(T2)

    print "matmat diff = ", np.abs(res1 - res2).max()

    # testing full inverse
    nugget = 1e-3
    Kop.set_eigs(nugget=nugget)
    res1 = (Kmat + nugget*np.eye(N)).dot(t)
    tres = Kop.idot_full(res1)

    print "idot diff", np.abs(t - tres).max()

    # # set up general purpose 1D FFT schemes
    # x = pyfftw.empty_aligned(N, dtype='complex128')
    # x2 = pyfftw.empty_aligned(N, dtype='complex128')
    # xhat = pyfftw.empty_aligned(N, dtype='complex128')
    # # forward transform
    # FFT = pyfftw.FFTW(x, xhat, direction='FFTW_FORWARD', threads=4)
    # iFFT = pyfftw.FFTW(xhat, x2, direction='FFTW_BACKWARD', threads=4)
    # x[:] = t + 1.0j*t
    # FFT()
    # iFFT()
    #
    # print np.abs(x-x2).max()

    # res1 = np.sort(Kop.idot(t))[::-1]
    # res1 = Kop.idot(t)
    # res2 = np.sort(np.linalg.solve(Kmat + 1e-13*np.eye(N), t))[::-1]
    # res2 = np.linalg.solve(Kmat + 1e-13*np.eye(N), t)

    import matplotlib.pyplot as plt
    # plt.figure('inv')
    # plt.plot(res1, 'bx')
    # plt.plot(res2, 'rx')
    # plt.show()

    # # test pspec eigenspectrum and determinant
    # freqs = (np.linspace(0, 2*np.pi, N)*np.arange(N))**2
    # delx = Kop.K1[1] - Kop.K1[0]
    # omega = (2*np.pi*np.fft.fftfreq(N, delx))**2
    # Lambda = Kop.s
    #
    # pspec = Kop.spectral_density(theta, freqs) + 1e-15
    # pspec2 = Kop.spectral_density(theta, omega) + 1e-15
    # eigs = np.sort(Kop.Chat2)[::-1][0:N].real*N/(2*N - 2)
    # eigs2 = np.sort(Kop.Chat2 - theta[-1]**2)[::-1][0:N].real*N/(2*N - 2)
    # pspec3 = Kop.spectral_density(theta, np.sort(np.sqrt(eigs2) + 1e-15))
    # pspec4 = Kop.spectral_density(theta, Lambda) + 1e-15
    #
    # # test logdet
    # Ky = Kmat + sigman**2*np.eye(N)
    # s, logdet = np.linalg.slogdet(Ky)
    # logdet *= s
    #
    # logdeteigs = np.sum(np.log(eigs))
    #
    # logdetRR = RRlogdet(N, sigman, pspec)
    # logdetRR2 = RRlogdet(N, sigman, pspec2)
    # logdetRR3 = RRlogdet(N, sigman, pspec3)
    # logdetRR4 = RRlogdet(N, sigman, pspec4)
    #
    # logdetdiag = np.sum(np.log(np.diag(Ky)))
    #
    # teff = (x[-1]-x[0])/l
    # seff = sigmaf/l
    # SNR = sigmaf/sigman
    #
    # print "logdet = ", logdet, "teff = ", teff, "seff = ", seff, "SNR = ", SNR
    # print "eig logdet = ", logdeteigs, np.abs(logdet - logdeteigs)/np.abs(logdet)
    # print "RR logdet = ", logdetRR, np.abs(logdet - logdetRR)/np.abs(logdet)
    # print "RR2 logdet = ", logdetRR2, np.abs(logdet - logdetRR2) / np.abs(logdet)
    # print "RR3 logdet = ", logdetRR3, np.abs(logdet - logdetRR3) / np.abs(logdet)
    # print "RR4 logdet = ", logdetRR4, np.abs(logdet - logdetRR4) / np.abs(logdet)
    # print "diag logdet = ", logdetdiag, np.abs(logdet - logdetdiag)/np.abs(logdet)







