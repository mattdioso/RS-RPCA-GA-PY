#!/usr/bin/env python3
import numpy as np
import math
import scipy as sp
from tvdantzig_newton import tvdantzig_newton
def logbarrier_tvdantzig(x0, A, At, b, epsilon, lbtol=None, mu=None, cgtol=None, cgmaxiter=None):
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
    n = round(math.sqrt(N))

    Dv = sp.sparse.spdiags(np.concatenate(np.concatenate(-np.ones(n-1, n), np.zeros(1, n).reshape(N, 1), ), np.concatenate(np.zeros(1, n), np.ones(n-1, n).reshape(N, 1))), np.array([0, 1]), N, N)

    Dh = sp.sparse.spdiags(np.concatenate(np.concatenate(-np.ones(n, n-1), np.zeros(n, 1).reshape(N, 1), ), np.concatenate(np.zeros(n, 1), np.ones(n, n-1).reshape(N, 1))), np.array([0, n]), N, N)

    x = x0
    Dhx = Dh*x
    Dvx = Dv*x

    t = 1.05 * math.sqrt(np.square(Dhx) + np.square(Dvx)) + .01*math.max(math.sqrt(np.square(Dhx) + np.square(Dvx)))

    tau = 3*N/np.sum(math.sqrt(np.square(Dhx) + np.square(Dvx)))

    lbiter = math.ceil((np.log(3*N)-np.log(lbtol)-np.log(tau))/np.log(mu))
    totaliter = 0
    for ii in range(0, lbiter):
        xp, tp, ntiter = tvdantzig_newton(x, t, A, At, b, epsilon, tau, newtontol, newtonmaxiter, cgtol, cgmaxiter)
        totaliter = totaliter + ntiter
        tvxp = np.sum(math.sqrt(np.square(Dh*xp)) + np.square(Dv*xp))
        x = xp
        t = tp
        tau = mu * tau

    return xp

