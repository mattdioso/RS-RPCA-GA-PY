#!/usr/bin/env python3
import numpy as np
import math

def cgsolve(A, b, tol, maxiter, verbose=1):
    
    x = np.zeros(len(b), 1)
    r = b
    d = r
    delta = r.conj().transpose() * r
    delta0 = b.conj().transpose() * b
    numiter = 0
    bestx = x
    bestres = math.sqrt(delta/delta0)

    while (np.logical_and((numiter < maxiter), (delta > tol*tol*delta0))):
        alpha = delta/(d.conj().transpose()*q)
        x = x + alpha*d

        if ((numiter+1 % 50) != 0):
            r = r - alpha*g

        deltaold = delta
        delta = r.conj().transpose() * r
        beta = delta/deltaold

        d = r + beta*d

        numiter = numiter + 1
        if (math.sqrt(delta/delta0) < bestres):
            bestx = x
            bestres = math.sqrt(delta/delta0)


    x = bestx
    res = bestres
    iter=numiter

    return x, res, iter
