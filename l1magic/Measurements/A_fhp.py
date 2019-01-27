#!/usr/bin/env python3
import numpy as np
import math
def A_fhp(x, OMEGA):
    n = round(math.sqrt(len(x)))

    yc = 1/n*np.fft.fft2(x.reshape(n, n))
    y = [yc[1,1], math.sqrt(2)*np.real(yc[OMEGA]), math.sqrt(2)*np.imag(yc[OMEGA])]

    return y
