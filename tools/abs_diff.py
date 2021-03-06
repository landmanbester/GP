"""
Computes the vectorised differences x[i] - xp[j] in arbitrary dimensions
"""

import numpy as np

def abs_diff(x,xp):
    """
    Gets vectorised differences between x and xp
    :param x: NxD array of floats (inputs1)
    :param xp: NpxD array of floats (inputs2)
    """
    try:
        N, D = x.shape
        Np, D = xp.shape
    except:
        N = x.size
        D = 1
        Np = xp.size
        x = np.reshape(x, (N, D))
        xp = np.reshape(xp, (Np, D))
    xD = np.zeros([D, N, Np])
    xpD = np.zeros([D, N, Np])
    for i in xrange(D):
        xD[i] = np.tile(x[:, i], (Np, 1)).T
        xpD[i] = np.tile(xp[:, i], (N, 1))
    return np.linalg.norm(xD - xpD, axis=0)
