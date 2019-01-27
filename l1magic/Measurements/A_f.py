#!/usr/bin/env python3
import math
import numpy as np

def A_f(x, OMEGA, P=None):
    if P is None:
        N = len(x)
        P = N

    fx = 1/math.sqrt(N)*np.fft(x[:,P])

    b = [math.sqrt(2) * np.real(fx[OMEGA]), math.sqrt(2) * np.imag(fx[OMEGA])]
    
    return b
