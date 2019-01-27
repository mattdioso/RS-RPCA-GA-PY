#!/usr/bin/env python3
import math
import numpy as np

def l1dantzig_pd(x0, A, At, b, epsilon, pdtol=None, pdmaxiter=None, cgtol=None, cgmaxiter=None):
    if pdtol is None:
        pdtol = math.exp(-3)

    if pdmaxiter is None:
        pdmaxiter = 50

    if cgtol is None:
        cgtol = math.exp(-8)

    if cgmaxiter is None:
        cgmaxiter = 200

    N = len(x0)

    alpha = 0.01
    beta = 0.5
    mu = 10

    gradf0 = np.concatenate(np.zeros(N, 1), np.ones(N, 1))

    x = x0
    u = 1.01*abs(x0) + math.exp(-2)

    Atr = A.conj().transpose() * (A*x - b)

    fu1 = x - u
    fu2 = -x - u
    fe1 = Atr - epsilon
    fe2 = Atr - epsilon
    lamu1 = -(1/fu1)
    lamu2 = -(1/fu2)
    lame1 = -(1/fe1)
    lame2 = -(1/fe2)

    AtAv = A.conj().transpose() * (A*(lame1-lame2))

    np.shape(np.concatenate(fu1, fu2, fe1, fe2))
    sdg = -(np.concatenate(fu1, fu2, fe1, fe2).conj().transpose()) * np.concatenate(lamu1, lamu2, lame1, lame2)
    tau = mu*(4*N)/sdg

    rdual = gradf0 + np.concatenate(lamu1-lamu2 + AtAv, -lamu1-lamu2)

    rcent = -(np.concatenate(np.multiply(lamu1,fu1), np.multiply(lamu2,fu2), np.multiply(lame1,fe1), np.multiply(lame2,fe2))) - (1/tau)

    resnorm = np.linalg.norm(np.concatenate(rdual, rcent))

    pditer = 0
    done = np.logical_or((sdg<pdtol), (pditer >= pdmaxiter))
    while not done:
        w2 = -1 -(1/tau)*(np.divide(1,fu1) + np.divide(1,fu2))
        sigl1 = np.divide(-lamu1,fu1) - np.divide(lamu2,fu2)
        sigl2 = np.divide(lamu1,fu1) - np.divide(lamu2,fu2)
        siga = -(np.divide(lame1, fe1) + np.divide(lame2,fe2))
        sigx = sigl1 - np.divide(np.square(sigl2), sigl1)
        
        #Pretty sure not doing largescale check

        w1 = -(1/tau) * (A.conj().transpose() * (A * (np.divide(1,fe2) - np.divide(1,fe1))) + np.divide(1, fu2) - np.divide(1,fu1))
        w1p = w1 - (np.divide(sigl2,sigl1))*w2
        Hp = A.conj().transpose() * (A*np.diag(siga)*A.conj().transpose()) * A + np.diag(sigx)
        dx = np.linalg.solve(Hp, w1p)

        AtAdx = A.conj().transpose() * (A*dx)

    du = np.divide(w2,sigl1) - np.divide(sigl2,sigl1)*dx

    dlamu1 = np.multiply(np.divide(-lamu1,fu1),(dx-du)) - lamu1 - (1/tau)*np.divide(1,fu1)
    dlamu2 = np.multiply(np.divide(-lamu2,fu2),(-dx-du)) - lamu2 - (1/tau)*np.divide(1,fu2)
    dlame1 = np.multiply(np.divide(-lame1,fe1),(AtAdx)) - lame1 - (1/tau)*np.divide(1,fe1)
    dlame2 = np.multiply(np.divide(-lame2,fe2),(-AtAdx)) - lame2 - (1/tau)*np.divide(1,fe2)

    AtAdv = A.conj().transpose() * (A*(dlame1-dlame2))

    iu1 = np.nonzero(dlamu1 < 0)
    iu2 = np.nonzero(dlamu2 < 0)
    ie1 = np.nonzero(dlame1 < 0)
    ie2 = np.nonzero(dlame2 < 0)
    ifu1 = np.nonzero((dx-du)<0)
    ifu2 = np.nonzero((-dx-du) > 0)
    ife1 = np.nonzero(AtAdx > 0)
    ife1 = np.nonzero(-AtAdx > 0)

    smax = math.min(1, math.min(np.concatenate(np.divide(-lamu1[iu1],dlamu1[iu1]), np.divide(-lamu2[iu2],dlamu2[iu2]), np.divide(-lame1[ie1],dlame1[ie1]), np.divide(-lame2[ie2],dlame2[ie2]), np.divide(-fu1[ifu1],(dx[ifu1]-du[ifu1])), np.divide(-fu2[ifu2],(-dx[ifu2]-du[ifu2])), np.divide(-fe1[ife1],AtAdx[ife1]), np.divide(-fe2[ife2],(-AtAdx[ife2])))))

    s= 0.99 * smax

    suffdec = 0 
    backiter = 0
    while not suffdec:
        xp = x + s*dx
        up = u + S*du
        Atrp = Atr + s*AtAdx
        AtAvp = AtAv + s*AtAdv
        fu1p = fu1 + s*(dx-du)
        fu2p = fu2 + s*(-dx-du)
        fe1p = fe1 + s*AtAdx
        fe2p = fe2 + s*(-AtAdx)
        lamu1p = lamu1 + s*dlamu1
        lamu2p = lamu2 + s*dlamu2
        lame1p = lame1 + s*dlame1
        lame2p = lame2 + s*dlame2
        rdp = gradf0 + np.concatenate(lamu1p-lamu2p + AtAvp, -lamu1p-lamu2p)
        rcp = -np.concatenate(np.multiply(lamu1p,fu1p), np.multiply(lamu2p,fu2p), np.multiply(lame1p,fe1p), np.multiply(lame2p,fe2p)) - 1/tau
        suffdec = (np.linalg.norm(np.concatenate(rdp, rcp)) <= (1-alpha*s)*resnorm)
        s = beta * s
        backiter = backiter + 1
        if (backiter > 32):
            print("Stuck backtracking, returning last iterate")
            xp = x
            return xp


    x = xp
    u = up
    Atr = Atrp
    AtAv = AtAvp
    fu1 = fu1p
    fu2 = fu2p
    fe1 = fe1p
    fe2 = fe2p
    lamu1 = lamu1p
    lamu2 = lamu2p
    lame1 = lame1p
    lame2 = lame2p

    sdg = -np.concatenate(fu1, fu2, fe1, fe2).conj().transpose() * np.concatenate(lamu1, lamu2, lame1, lame2)
    tau = mu*(4*N)/sdg

    rdual = rdp
    rcent = -np.concatenate(np.multiply(lamu1,fu1), np.multiply(lamu2,fu2), np.multiply(lame1,fe1), np.multiply(lame2,fe2)) - (1/tau)
    resnorm = np.linalg.norm(np.concatenate(rdual, rcent))

    pditer = pditer + 1
    done = np.logical_or((sdg<pdtol), (pditer >= pdmaxiter))
    
    return xp


