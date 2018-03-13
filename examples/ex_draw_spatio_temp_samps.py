

import numpy as np
from GP.tools import draw_samples


if __name__=="__main__":
    Nx = 1000
    x = np.linspace(-1, 1, Nx)

    Nt = 500
    t = np.linspace(0, 10, Nt)

    meanf = np.ones([Nt, Nx], dtype=np.float64)

    theta_x = np.array([1.0, 0.1], dtype=np.float64)
    theta_t = np.array([0.5, 0.5], dtype=np.float64)

    Nsamps = 10

    samps = draw_samples.draw_spatio_temporal_samples(meanf, x, t, theta_x, theta_t, Nsamps)

    import matplotlib.pyplot as plt

    plt.figure()

    for i in xrange(Nsamps):
        plt.imshow(samps[i])
        plt.colorbar()
        plt.show()
