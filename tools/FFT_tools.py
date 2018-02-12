
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
    #Lambda = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(c)))
    #xk = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))
    #return np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(xk * Lambda)))
    Lambda = np.fft.fft(np.fft.ifftshift(c), norm="ortho")
    xk = np.fft.fft(np.fft.ifftshift(x), norm="ortho")
    return np.fft.ifft(np.fft.fftshift(xk * Lambda), norm="ortho")


def FFT_toepvec(t, x):
    """
    Computes matrix vector product y = Kx where K is a covariance (i.e. Hermitian Toeplitz) matrix. 
    Here t is the first row/column of the covariance matrix 
    :param t: the upper row vector of the covariance matrix 
    :param x: the vector to multiply with
    :return: y the result
    """
    # broadcast to circulant
    c = np.append(t, np.append(np.zeros(1), t[np.arange(Nx)[0:-1][::-1]].conj()))
    x_aug = np.append(x, np.zeros(x.size))
    return FFT_circvec(c, x_aug)[0:x.size]


if __name__=="__main__":
    from GP.tools import abs_diff
    from GP.kernels import exponential_squared

    # get Toeplitz matrix
    kernel = exponential_squared.sqexp()
    Nx = 100
    thetax = np.array([0.25, 0.25])
    x = np.linspace(-1, 1, Nx)
    xx = abs_diff.abs_diff(x, x)
    Kx = kernel.cov_func(thetax, xx, noise=False)

    # broadcast to circulant
    row1 = np.append(Kx[0, :], np.append(np.zeros(1), Kx[-1, 1::]))
    row2 = np.append(Kx[0, :], np.append(np.zeros(1), Kx[0, np.arange(Nx)[0:-1][::-1]].conj()))
    print "row diff = ", np.abs(row1 - row2).max()


    row3 = np.fft.ifft(np.fft.fft(row1, norm='ortho'), norm='ortho')

    print np.abs(row3 - row1).max()

    # compare eigenvalues to fft coefficients
    import scipy.linalg as scl
    Kcirc = scl.circulant(row2)
    Lambda, Q = np.linalg.eigh(Kcirc)

    Lambda2 = np.abs(np.fft.rfft(row2))

    Lambda = np.sort(Lambda)
    Lambda2 = np.sort(Lambda2)

    print np.abs(Lambda - Lambda2).max()


    # test circvec
    t = np.sin(2*np.pi*x)
    res1 = np.dot(Kx, t)

    res2 = FFT_toepvec(Kx[0, :], x)
    print "Toepvec diff = ", np.abs(res1 - res2).max()

    # import matplotlib.pyplot as plt
    #
    # plt.figure('Spectrum')
    # plt.plot(Lambda)
    # plt.show()