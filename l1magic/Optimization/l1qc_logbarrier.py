#!/usr/bin/env python3
import numpy as np
import math
from l1qc_newton import l1qc_newton

def l1qc_logbarrier(x0, A, At, b, epsilon, lbtol=None, mu=None, cgtol=None, cgmaxiter=None):
    if lbtol is None:
        lbtol = math.exp(-3)
    if mu is None:
        mu = 10
    if cgtol is None:
        cgtol = math.exp(-8)
    if cgmaxiter is None:
        cgmaxiter = 200

    newtontol = lbtol
    newtonmaxiter = 50

    N = len(x0)
    x = x0
    u = 1.05*abs(x0) + .01*math.max(abs(x0))

    tau = (2*N+1)/np.sum(abs(xo))

    lbiter = math.ceil((np.log(2*N+1) - np.log(lbtol) - np.log(tau))/np.log(mu))

    totaliter = 0
    for ii in range(0, lbiter):
        xp, up, ntiter = l1qc_newton(x, u, A, At, b, epsilon, tau, newtontol, newtonmaxiter, cgtol, cgmaxiter)
        totaliter = totaliter + ntiter
        
        x = xp
        u = up
        tau = mu*tau

    return xp
