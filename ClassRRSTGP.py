#!/usr/bin/env python
"""

First attempt at constructing a reduced rank spatio temporal GP with 2D spatial sections and a time axis.
Here we are assuming each spatial slice has the same coordinates which allows efficient evaluation using 
the kronecker product.

"""