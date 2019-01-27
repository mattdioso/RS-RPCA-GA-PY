#!/usr/bin/env python3
import numpy as np
import math

def At_f(b, N, OMEGA, P=None):
    if P is None:
        P = len(N)

    K = len(b)
    fx = np.zeros(N, 1)
    fx[OMEGA] = math.sqrt(2)*b[:K/2] + i*math.sqrt(2)*b[K/2 + 1: K]

    x = np.zeros[N, 1]

    x[P] = math.sqrt(N)*np.real(np.fft.ifft(fx))

    return x
