#!/usr/bin/env python3
import math
import numpy as np

def At_fhp(y, OMEGA, n):
    K = len(y)
    fx = np.zeros(n, n)

    fx[1, 1] = y[1]

    fx[OMEGA] = math.sqrt(2)*(y[1:(K+1)/2] + i*y[(K+3)/2:K])

    x = np.real(n*np.fft.ifft2(fx)).reshape(n*n, 1)

    return x
