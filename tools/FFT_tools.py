
"""
Some tools to exploit fast matrix vector products using the FFT. Inputs need to be on a regular grid. 
"""

import numpy as np

def FFT_circvec(c, x):
    """
    Computes matrix vector product y = Cx where C = circulant(c) in NlogN time
    :param c: the upper row vector of circulant matrix C 
    :param x: the vector to multiply with
    :return: y the result
    """
    Lambda = np.fft.rfft(c)
    xk = np.fft.rfft(x)
    return np.fft.irfft(Lambda*xk)


def FFT_toepvec(t, x):
    """
    Computes matrix vector product y = Kx where K is a covariance (i.e. Hermitian Toeplitz) matrix. 
    Here t is the first row/column of the covariance matrix 
    :param t: the upper row vector of the covariance matrix 
    :param x: the vector to multiply with
    :return: y the result
    """
    # broadcast to circulant
    c = np.append(t, t[np.arange(Nx)[1:-1][::-1]].conj())
    x_aug = np.append(x, np.zeros(x.size-2))
    tmp = FFT_circvec(c, x_aug)
    return tmp[0:x.size]


if __name__=="__main__":
    from GP.tools import abs_diff
    from GP.kernels import exponential_squared

    # get Toeplitz matrix
    kernel = exponential_squared.sqexp()
    Nx = 1024
    thetax = np.array([0.25, 0.25])
    x = np.linspace(-10, 10, Nx)
    xx = abs_diff.abs_diff(x, x)
    Kx = kernel.cov_func(thetax, xx, noise=False)

    # broadcast to circulant
    row1 = np.append(Kx[0, :], np.append(np.zeros(1), Kx[-1, 1::]))
    row2 = np.append(Kx[0, :], np.append(np.zeros(1), Kx[0, np.arange(Nx)[0:-1][::-1]].conj()))
    print "row diff = ", np.abs(row1 - row2).max()


    row3 = np.fft.ifft(np.fft.fft(row1, norm='ortho'), norm='ortho')

    print "FFT test diff = ", np.abs(row3 - row1).max()

    # compare eigenvalues to fft coefficients
    import scipy.linalg as scl

    row2 = np.append(Kx[0, :], Kx[0, np.arange(Nx)[1:-1][::-1]].conj())
    Kcirc = scl.circulant(row2)

    t = np.random.randn(row2.size)
    res1 = Kcirc.dot(t)
    res2 = FFT_circvec(row2, t)
    print "Circvec diff =", np.abs(res1 - res2).max()

    Lambda, Q = np.linalg.eigh(Kcirc)

    Lambda2 = np.fft.fft(row2).real

    # get the eigenvectors (they must be the roots of unity

    #Lambda = np.sort(Lambda)
    #Lambda2 = np.sort(Lambda2)

    #print "Eig diff = ", np.abs(Lambda - Lambda2).max(), np.abs(Lambda - Lambda2).min()


    # test circvec
    res1 = np.dot(Kx, t[0:Nx])

    res2 = FFT_toepvec(Kx[0, :], t[0:Nx])
    print "Toepvec diff = ", np.abs(res1 - res2).max()

    import matplotlib.pyplot as plt

    plt.figure('Toepvec Diff ')
    plt.plot(res1 - res2)
    plt.figure('Spectra 1')
    plt.plot(Lambda, 'x')
    plt.figure("Spectra 2")
    plt.plot(Lambda2, 'x')
    plt.show()




