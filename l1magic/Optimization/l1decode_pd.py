#!/usr/bin env python3
import numpy as np
import math

def l1decode_pd(x0, A, At, y, pdtol=None, pdmaxiter=None, cgtol=None, cgmaxiter=None):
    if pdtol is None:
        pdtol = math.exp(-3)

    if pdmaxiter is None:
        pdmaxiter = 50

    if cgtol is None:
        cgtol = math.exp(-8)

    if cgmaxiter is None:
        cgmaxiter = 200

    N = len(x0)
    M = len(y)

    alpha = 0.01
    beta = 0.5
    mu = 10

    gradf0 = np.concatenate(np.zeros(N, 1), np.ones(M, 1))
    
    x = x0

    Ax = A * x

    u = (0.95) * abs(y-Ax) + (0.10)*math.max(abs(y-Ax))
    fu1 = Ax -y -u
    fu2 = -Ax + y - u

    lamu1 = -np.divide(1, fu1)
    lamu2 = -np.divide(1, fu2)

    sdg = -(np.multiply(fu1.conj().transpose(), lamu1) + np.multiply(fu2.conj().transpose(), lamu2))

    tau = mu*2*M/sdg

    rcent = np.concatenate(np.multiply(-lamu1, fu1),np.multiply(-lamu2, fu2)) - (1/tau)
    rdual = gradf0 + np.concatenate(Atv, -lamu1-lamu2)
    resnorm = np.linalg.norm(np.concatenate(rdual, rcent))

    pditer = 0
    done = np.logical_or((sdg < pdtol), (pditer >= pdmaxiter))
    while not done:
        pditer = pditer + 1

        w2 = -1 - 1/tau * (np.divide(1, fu1) + np.divide(1/fu2))

        sig1 = np.divide(-lamu1, fu1) - np.divide(-lamu2, fu2)
        sig2 = np.divide(lamu1, fu1) - np.divide(lamu2, fu2)
        sigx = sig1 - np.divide(np.square(sig2), sig1)

        ###IGNORING LARGESCALE CONFIGURATION

        w1 = -1/tau*(A.conj().transpose() * (np.divide(-1, fu1) + np.divide(1, fu2)))
        w1p = w1 - A.conj().transpose() * (np.multiply(np.divide(sig2, sig1), w2))
        H11p = A.conj().transpose() * np.diag(sigx) * A
        dx = np.linalg.solve(H11p, w1p)
        Adx = A * dx

        du = np.divide((w2 - np.multiply(sig2, Adx)), sig1)
        dlamu1 = np.multiply(np.divide(-lamu1, fu1), (Adx - du)) - lamu1 - (1/tau) * np.divide(1, fu1)
        dlamu2 = np.multiply(np.divide(lamu2, fu2), (Adx + du)) - lamu2 - (1/tau)*np.divide(1, fu2)

        Atdv = A.conj().transpose() * (dlamu1 - dlamu2)

        indl = np.nonzero(dlamu1 < 0)
        indu = np.nonzero(dlamu2 < 0)
        s = math.min(np.concatenate(1, np.divide(-lanu1[indl], dlamu1[indl]), np.divide(-lanu2[indl], dlamu2[indu])))
 
        indl = np.nonzero((Adx - du) > 0)
        indu = np.nonzero((-Adx - du) > 0)

        s = (0.99) * math.min(np.concatenate(s, np.divide(-fu1[indl], (Adx[indl] - du[indl]), np.divide(-fu2[indu], (-Adx[indu]-du[indu])))))

        backiter = 0
        xp = x + s*dx
        up = u + s*du
        Axp = Ax + s*Adx
        Atvp = Atv + s*Atdv
        lamu1p = lamu1 + s*dlamu1
        lamu2p = lamu2 + s*dlamu2
        fu1p = Axp - y - up

        fu2p = -Axp + y - up
        rdp = gradf0 + np.concatenate(Atvp, -lamu1p-lamu2p)
        rcp = np.concatenate(np.multiply(-lamu1p, fu1p), np.multiply(-lamu2p, fu2p))
        
        while (np.linalg.norm(np.concatenate(rdp, rcp)) > ((1-alpha*s)*resnorm)):
            s = beta * s
            xp = x + s * dx
            up = u + s * du
            Axp = Ax + s*Adx
            Atvp = Atv + s*Atdv
            lamu1p = lamu1 + s*dlamu1
            lamu2p = lamu2 + s*dlamu2
            fu1p = Axp - y - up
            fu2p = -Axp + y - up
            rdp = gradf0 + np.concatenate(Atvp, -lamu1p-lamu2p)
            rcp = np.concatenate(np.multiply(-lamu1p, fu1p), np.multiply(-lamu2p, fu2p)) - (1/tau)
            backiter = backiter + 1
            if (backiter > 32):
                xp = x
                return xp

        x = xp
        u = up
        Ax = Axp
        Atv = Atvp
        lamu1 = lamu1p
        lamu2 = lamu2p
        fu1 = fu1p
        fu2 = fu2p
        
        sdg = -(fu1.conj().transpose() * lamu1 + fu2.conj().transpose() * lamu2)

        tau = mu*2*M/sdg
        rcent = np.concatenate(np.multiply(-lamu1, fu1), np.multiply(-lamu2, fu2)) - (1/tau)
        rdual = rdp
        resnorm = np.linalg.norm(np.concatenate(rdual, rcent))

        done = np.logical_or((sdg < pdtol), (pditer >= pdmaxiter))

    return xp
