#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sympy import symbols, expand, simplify, Poly, integrate
from sympy.utilities.lambdify import lambdify
from sympy.abc import s
from sympy.matrices import Matrix, eye
from sympy.integrals.transforms import inverse_laplace_transform as ilaplace
from sympy.physics.quantum import TensorProduct

def give_Phi_and_Q(nu):
    p = int(nu-0.5)
    t = symbols('t', positive=True, real=True)
    lam = symbols('lam', positive=True, real=True)
    # create polynomial expression 
    factors = Poly(expand((lam+1)**(p+1)))
    # extract coeffs
    coeffs = factors.coeffs()
    # create inverse of resolvent matrix [sI - A]
    M = eye(p+1, p+1)
    M *= s
    for i in range(p+1):
        if i < p:
            M[i, i+1] = -1
        M[p, i] += coeffs[i] * lam**(p+1-i)
    # get resolvent
    resolvent = simplify(M.inv())
    # get state transition matrix
    Phi = ilaplace(resolvent, s, t)
    # turn into callable (need sympy here since numpy's Heavyside 
    # doesn't get lambdified correctly https://github.com/sympy/sympy/issues/13176)
    Phi_func = lambdify((t, lam), Phi, modules=["numpy","sympy"])
    # get Q
    Phi_last = Phi[:, -1].reshape(p+1, 1)
    Outer = Phi_last * Phi_last.T
    # take integral
    delta = symbols('delta', positive=True, real=True)
    Q = simplify(integrate(Outer, (t, 0, delta)))
    # turn into callable
    Q_func = lambdify((delta, lam), Q, modules=["numpy", "sympy"])
    return Phi, Q

Phi, Q = give_Phi_and_Q(3.5)
