#!/usr/bin/env python3
import numpy as np
import math

def l1qc_newton(x0, u0, A, At, b, epsilon, tau, newtontol, newtonmaxiter, cgtol, cgmaxiter):
    alpha = 0.01
    beta = 0.5

    N = len(x0)

    Ata = A.conj().transpose() * A

    x =x0
    u =u0

    r = A*x -b

    fu1 = x - u
    fu2 = -x - u
    fe = 0.5*(r.conj().transpose() * r - epsilon**2)
    f = np.sum(u) - (1/tau)*(np.sum(np.log(-fu1)) + np.sum(np.log(-fu2)) + np.log(-fe))

    niter = 0
    done = 0
    while not done:
        atr = A.conj().transpose() * r

        ntgz = np.divide(1, fu1) - np.divide(1, fu2) + 1/fe * atr
        ntgu = -tau - np.divide(1, fu1) - np.divide(1, fu2)
        gradf = -(1/tau)*np.concatenate(ntgz, ntgu)

        sig11 = np.divide(1, np.square(fu1)) + np.divide(1, np.square(fu2))
        sig12 = np.divide(-1, np.square(fu1)) + np.divide(1, np.square(fu2))
        sigx = sig11 - np.divide(np.square(sig12), sig11)

        w1p = ntgz - np.divide(sig12, np.multiply(sig11, ntgu))
        H11p = np.diag(sigx) - (1/fe)*AtA + (1/fe)**2*atr*atr.conj().transpose()
        dx = H11p/w1p
        Adx = A*dx

        du = np.multiply(np.divide(1, sig11), ntgu) - np.multiply(np.divide(sig12, sig11), dx)

        s = 1
        xp = x+s*dx
        up = u+s*du
        rp = r+s*Adx
        coneiter = 0
        while (np.logical_or((math.max(abs(xp)-up) > 0), (rp.conj().transpose()*rp > epsilon **2 ))):
            s = beta * s
            xp = x + s*dx
            up = u +s*du
            rp = r + s*Adx
            coneiter = coneiter + 1
            if (coneiter > 32):
                xp = x
                up = u
                return xp, up, niter

        fu1p = xp - up
        fu2p = -xp - up
        fep = 0.5 * (rp.conj().transpose()*rp - epsilon**2)
        fp = np.sum(up) - (1/tau)*(np.sum(np.log(-fu1p)) + np.sum(np.log(-fu2p)) + np.log(-fep))
        flin = f + alpha * s * (gradf.conj().transpose() * np.concatenate(dx, du))
        backiter = 0
        while (fp > flin):
            s = beta*s
            xp = x+ s*dx
            up = u + s*du
            rp = r + s*Adx
            fu1p = xp - up
            fu2p = -xp -up
            fep = 0.5*(rp.conj().transpose() * rp - epsilon**2)
            fp = np.sum(up) - (1/tau)*(np.sum(np.log(-fu1p)) + np.sum(np.log(-fu2p)) + np.log(-fep))
            flin = f + alpha*s*(gradf.conj().transpose() * np.concatenate(dx, du))
            backter = backiter + 1
            if (backiter>32):
                xp =x
                up = u
                return xp, up, niter

        x = xp
        u = up
        r = rp
        fu1 = fu1p
        fu2 = fu2p
        fe = fep
        f = fp
        lambda2 = -(gradf.conj().transpose() * np.concatenate(dx, du))
        stepsize = s * np.linalg.norm(np.concatenate(dx, du))
        niter = niter + 1
        done = np.logical_or((lambda2/2 < newtontol), (niter >= newtonmaxiter))


    return xp, up, niter
