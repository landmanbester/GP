

import numpy as np
from scipy.special import gamma
from GP.tools import draw_samples

def give_mattern_pspec(s, v, sigmaf, l):
    p = v + 0.5
    lam = np.sqrt(2*v)/l
    q = sigmaf**2 * 2 * np.sqrt(np.pi) * gamma(p) * lam**(2*v)/gamma(v)
    return q/(lam**2 + s**2)**p

if __name__=="__main__":
    Nt = 350
    t = np.linspace(0, 1, Nt)

    Nnu = 250
    nu = np.linspace(0, 1, Nnu)

    theta_t = np.array([0.5, 0.1], dtype=np.float64)
    theta_nu = np.array([0.1, 1.5], dtype=np.float64)

    Nsamps = 5

    # lets prescribe a Mattern with v=3/2 for time
    sigft = 1.0
    lt = 0.25
    St_func = lambda s: give_mattern_pspec(s, 3.0/2, sigft, lt)
    # and a Mattern with v=7/2 for frequency
    sigfnu = 1.0
    lnu = 1.5
    Snu_func = lambda s: give_mattern_pspec(s, 7.0/2, sigfnu, lnu)
    samps = draw_samples.draw_time_freq_gain_samps(t, nu, St_func, Snu_func, Nsamps)

    import matplotlib.pyplot as plt

    plt.figure()
    for k in xrange(Nsamps):
        plt.imshow(samps[k])
        plt.colorbar()
        plt.show()
