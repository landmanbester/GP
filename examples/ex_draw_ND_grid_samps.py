

import numpy as np
from GP.tools import draw_samples


if __name__=="__main__":
    Nt = 5
    t = np.linspace(0, 1, Nt)
    Nnu = 5
    nu = np.linspace(0, 1, Nnu)

    Nl = 50
    l = np.linspace(-0.25, 0.25, Nl)

    Nm = 50
    m = np.linspace(-0.25, 0.25, Nm)

    theta_t = np.array([0.5, 0.1], dtype=np.float64)
    theta_nu = np.array([0.1, 1.5], dtype=np.float64)
    theta_l = np.array([0.15, 0.05], dtype=np.float64)
    theta_m = np.array([0.15, 0.05], dtype=np.float64)

    x = np.array([t, nu, l, m])
    theta = np.array([theta_t, theta_nu, theta_l, theta_m])

    meanf = lambda x: np.ones([x[0].size, x[1].size, x[2].size, x[3].size], dtype=np.float64)

    Nsamps = 5

    samps = draw_samples.draw_samples_ND_grid(x, theta, Nsamps, meanf=meanf)

    import matplotlib.pyplot as plt

    plt.figure()
    for k in xrange(Nsamps):
        for i in xrange(Nt):
            for j in xrange(Nnu):
                plt.imshow(samps[k, i, j])
                plt.colorbar()
                plt.show()
