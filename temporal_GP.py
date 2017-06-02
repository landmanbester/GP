#!/usr/bin/env python
"""
This class performs temporal GPR (i.e. GPR in 1D)
"""

import numpy as np
from scipy import optimize as opt

class TempoGP(object):
    def __init__(self, x, xp, covariance_function='sqexp', mode="Full", nu=2.5, basis="Rect"):
        """
        :param x: vector of inputs
        :param xp: vector of targets
        :param covariance_function: the kind of covariance function to use (currently 'sqexp' or 'mattern')
        :param mode: whether to do a full GPR or a reduced rank GPR (currently "Full" or "RR")
        :param nu: specifies the kind of Mattern function to use (must be a half integer)
        :param basis: specifies the class of basis functions to use (currently only "Rect")
        """
        # set inputs and targets
        self.x = x
        self.xp = xp

        # set covariance
        if covariance_function=="sqexp":
            from GP.kernels import exponential_squared
            # Initialise kernel
            self.kernel = exponential_squared.sqexp()
        elif covariance_function=="mattern":
            from GP.kernels import mattern
            # Initialise kernel
            self.kernel = mattern.mattern(p=int(nu))

        if mode=="Full":
            from GP.tools import abs_diff
            # Initialise absolute differences
            self.XX = abs_diff.abs_diff(x, x)
            self.XXp = abs_diff.abs_diff(x, xp)
            self.XXpp = abs_diff.abs_diff(xp, xp)
        elif mode=="RR":
            if basis=="Rect":
                from GP.basisfuncs import rectangular

            else:
                print "%s basis functions not supported yet"%basis



