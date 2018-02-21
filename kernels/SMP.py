"""
The spectral mixture product kernel
"""

import numpy as np
from GP.tools import abs_diff

class SMP(object):
    def __init__(self, p=None, D=1, Sigmay=None, mode=None, A=3):
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
        self.A = int(A)


    def kernel(self, theta, x):
        """
        A single component of the spectral mixture kernel
        :param theta: array of hyperparameters theta = [w, l, mu]
        :param s: the fourier dual of the spatial coordinate
        :return: 
        """
        return theta[0]**2*np.exp(-2.0*np.pi**2*x**2*theta[1]**2)*np.cos(2.0*np.pi*x*theta[2])

    def kernels(self, thetas, x):
        """
        Computes the sum over A kernels with hypers specified in thetas (so 1D SMP)
        :param thetas: an array of arrays containing hypers for the individual kernels
        :param s: the fourier dual of spatial coordiante
        :return: 
        """
        Ki = np.zeros_like(x)
        for a in xrange(self.A):
            Ki += self.kernel(thetas[a], x)
        return Ki

    def cov_func(self, thetas, x):
        """
        Computes the product oof all kernels over D dimensions
        :param thetas: an array of arrays containing hypers for the individual kernels
        :param s: the fourier dual of spatial coordiante
        :return: 
        """
        K = np.empty(self.D, dtype=object)
        for d in xrange(self.D):
            K[d] = self.kernels(thetas[d], x[d])
        return K

if __name__=="__main__":
    N = 512
    A = 3
    D = 1
    x = np.array([np.linspace(0, 2*np.pi, N)])

    thetas = np.empty([D, A], dtype=object)
    for i in xrange(D):
        for j in xrange(A):
            thetas[i, j] = np.array([j*1.0, j*0.1, j*0.25])

    covo = SMP(D=1, A=3)

    covf = covo.cov_func(thetas, x)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x[0], covf[0])
    plt.show()
