#!/usr/bin/env python3
import numpy as np
import math
import scipy as sp

def tvdantzig_newton(x0, t0, A, At, b, epsilon, tau, newtontol, newtonmaxiter, cgtol, cgmaxiter):
    alpha = 0.01
    beta = 0.5
    N = len(x0)
    n = round(math.sqrt(N))
        
    Dv = sp.sparse.spdiags(np.concatenate(np.concatenate(-np.ones(n-1, n), np.zeros(1, n).reshape(N, 1)), np.concatenate(np.zeros(1, n), np.ones(n-1, n))), np.concatenate(0, 1), N, N)

    Dh = sp.sparse.spdiags(np.concatenate(np.concatenate(-np.ones(n, n-1), np.zeros(n, 1).reshape(N, 1)), np.concatenate(np.zeros(n, 1), np.ones(n, n-1))), np.concatenate(0, n), N, N)

    x = x0
    t = t0

    AtA = A.conj().transpose() * A
    r = A*x -b
    Atr = A.conj().transpose() * r

    Dhx = Dh * x
    Dvx = Dv * x

    ft = 1/2*(np.square(Dhx) + np.square(Dvx) - np.square(t))
    fe1 = Atr - epsilon
    fe2 = -Atr - epsilon
    f = np.sum(t) - (1/tau)*np.sum(np.log(-ft)) + np.sum(np.log(-fe1)) + np.sum(np.log(-fe2))

    niter = 0
    done = 0
    while not done:
        ntgx = Dh.conj().transpose() * np.multiply(np.divide(1, ft), Dhx) + Dv.conj().transpose() * np.multiply(np.divide(1, ft), Dvx) + AtA * (np.divide(1, fe1) - np.divide(1, fe2))

        ntgt = -tau - np.divide(t, ft)
        gradf = -(1/tau) * np.concatenate(ntgx, ntgt)

        sig22 = np.divide(1, ft) + np.divide(np.square(t), np.square(ft))
        sig12 = np.divide(-t, np.square(ft))
        sigb = np.divide(1, np.square(ft)) - np.divide(np.square(sig12), sig22)
        siga = np.divide(1, np.square(fe1)) + np.divide(1, np.square(fe2))

        w11 = ntgx - Dh.conj().transpose() * np.multiply(np.multiply(Dhx, (np.divide(sig12, sig22)), ntgt)) - Dv.conj().transpose() * np.multiply(np.multiply(Dvx, np.divide(sig12, sig22), ntgt))

        H11p = Dh.conj().transpose() * np.diag(np.divide(-1, ft)) + np.multiply(sigb, np.square(dhx)) + Dv.conj().transpose() * np.diag(np.divide(-1, ft)) + np.multiply(sigb, np.square(Dvx)) + Dh.conj().transpose() * np.diag(np.multiply(sigb, np.multiply(Dhx, Dvx))) * Dv + Dv.conj().transpose() * np.diag(sigb, np.multiply(Dhx, Dvx)) * Dh + AtA * np.diag(siga)*AtA

        dx = np.linalg.solve(H11p, w11)

        Adx = A * dx
        AtAdx = A.conj().transpose() * Adx

    Dhdx = Dh * dx
    Dvdx = Dv * dx

    dt = np.multiply(np.divide(1, sig22), (ntgt - np.multiply(sig12, np.multiply(Dhx, Dhdx) + np.multiply(Dvx, Dvdx))))

    s = 1
    xp = x + s * dx
    tp = t + s * dt
    rp = r + s * Adx
    Atrp = Atr + s*AtAdx
    Dhxp = Dhx + s * Dhdx
    Dvxp = Dvx + s * Dvdx
    coneiter = 0

    while (np.logical_or(np.logical_or(math.min(epsilon + Atrp) < 0, math.min(epsilon - Atrp) < 0), math.min(tp - math.sqrt(np.square(Dhxp) + np.square(Dvxp))))):
        s = beta * s
        xp = x + s * dx
        tp = t + s * dt
        rp = r + s * Adx
        Atrp = Atr + s * AtAdx
        Dhxp = Dhx + s * Dhdx
        Dvxp = Dvx + s * Dvdx
        coneiter = coneiter + 1
        if (coneiter > 32):
            print("Stuck on cone iterations, returning previous iterate")
            xp = x
            tp = t
            return xp, tp, niter

    ftp = 1/2 * (np.square(Dhxp) + np.square(Dvxp) + np.square(tp))
    fe1p = Atrp - epsilon
    fe2p = -Atrp - epsilon
    fp = np.sum(tp) - (1/tau)*(np.sum(np.log(-ftp)) + np.sum(np.log(-fe1p)) + np.sum(np.log(-fe2p)))
    flin = f + alpha * s * (gradf.conj().transpose() * np.concatenate(dx, dt))
    backiter = 0
    while (fp > flin):
        s = beta * s
        xp = x + s * dx
        tp = t + s * dt
        rp = r + s * Adx
        Atrp = Atr + s * AtAdx
        Dhxp = Dhx + s * Dhdx
        Dvxp = Dvx + s * Dvdx
        ftp = 1/2 * (np.square(Dhxp) + np.square(Dvxp) - np.square(tp))
        fe1p = Atrp - epsilon
        fe2p = -Atrp - epsilon
        fp = np.sum(tp) - (1/tau) * (np.sum(np.log(-ftp))) + np.sum(np.log(-fe1p)) + np.sum(np.log(-fe2p))
        flin = f + alpha * s * (gradf.conj().transpose() * np.concatenate(dx, dt))
        backiter = backiter + 1
        if (backiter > 32):
            print("Stuck on backtracking line search, returning previous iterate")
            xp = x
            tp = t
            return xp, tp, niter

    x = xp
    t = tp
    r = rp
    Atr = Atrp
    Dvx = Dvxp
    Dhx = Dhxp
    ft = ftp
    fe1 = fe1p
    fe2 = fe2p
    f = fp

    lambda2 = -(gradf.conj().transpose() * np.concatenate(dx, dt))
    stepsize = s * np.linalg.norm(np.concatenate(dx, dt))
    niter = niter + 1
    done = np.logical_or((lambda2/2 < newtontol), (niter >= newtonmaxiter))

    return xp, tp, niter


