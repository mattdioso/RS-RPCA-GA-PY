#!/usr/bin/env python3

def l1eq_pd(x0, A, At, b, pdtol=None, pdmaxiter=None, cgtol=None, cgmaxiter=None):
    if pdtol is None:
        pdtol = math.exp(-3)

    if pdmaxiter is None:
        pdmaxiter = 50

    if cgtol is None:
        cgtol = math.max(-8)

    if cgmaxiter is None:
        cgmaxiter = 200

    N = len(x0)

    alpha = 0.01
    beta = 0.5
    mu = 10

    gradf0 = np.concatenate(np.zeros(N, 1), np.ones(N, 1))
    x = x0
    u = 1.01*(math.max(abs(x)))*np.ones(N, 1) + math.exp(-2)
    fu1 = x - u
    fu2 = -x - u

    lamu1 = np.divide(-1, fu1)
    lamu2 = np.divide(-1, fu2)

    v = -A * (lamu1-lamu2)
    Atv = A.conj().transpose() * v
    rpri = A * x -b

    sdg = -(fu1.conj().transpose() * lamu1 + fu2.conj().transpose() * lamu2)
    tau = mu*2*N/sdg

    rcent = np.concatenate(np.multiply(-lamu1, fu1), np.multiply(-lamu2, fu2)) - (1/tau)
    rdual = gradf0 + np.concatenate(lamu1 - lamu2, -lamu1-lamu2) + np.concatenate(Atv, np.zeros(N, 1))
    resnorm = np.linalg.norm(np.concatenate(rdual, rcent, rpri))

    pditer = 0

    done = np.logical_or((sdg<pdtol), (pditer >= pdmaxiter))
    while not done:
        pditer = pditer + 1
        w1 = -1/tau * (np.divide(-1, fu1) + np.divide(1, fu2)) - Atv
        w2 = -1 - 1/tau * (np.divide(-1, fu1) + np.divide(1, fu2))
        w3 = -rpri

        sig1 = np.divide(-lamu1, fu1) - np.divide(lamu2, fu2)
        sig2 = np.divide(lamu1, fu1) - np.divide(lamu2, fu2)
        sigx = sig1 - np.divide(np.square(sig2), sig1)

        H11p = -A * np.diag(np.divide(1, sigx))*A.conj().transpose()
        w1p = w3 - A*(np.divide(w1, sigx) - np.divide(np.multiply(w2, sig2), np.multiply(sigx, sig1)))
        dv = np.linalg.solve(H11p, w1p)

        dx = np.divide((w1 - np.divide(np.multiply(w2, sig2), sig1))-A.conj().transpose()*dv, sigx )
        Adx = A*dx
        Atdv = A.conj().transpose() * dv

        du = np.divide(w2 - np.multiply(sig2, dx), sig1)

        dlamu1 = np.multiply(np.divide(lamu1, fu1), (-dx+du)) - lamu1 - (1/tau) * np.divide(1, fu1)
        dlamu2 = np.multiply(np.divide(lamu2, fu2), (dx + du)) - lamu2 - 1/tau * np.divide(1, fu2)

        indp = np.nonzero(dlamu1 < 0)
        indn = np.nonzero(dlamu2 < 0)
        s = math.min(np.concatenate(1, np.divide(-lamu1[indp], dlamu1[indp]), np.concatenate(-lanu2[indn], dlamu2[indn])))

        backiter = 0
        xp = x + s * dx
        up = u + s * du
        vp = v + s * dv
        Atvp  = Atv + s*Atdv

        lamu1p = lamu1 + s*dlamu1
        lamu2p = lamu2 + s*dlamu2
        fu1p = xp - up
        fu2p = -xp - up
        rdp = gradf0 + np.concatenate(lamu1p-lamu2p, -lamu1p-lamu2p) + np.concatenate(Atvp, np.zeros(N, 1))
        rcp = np.concatenate(np.multiply(-lamu1p, fu1p), np.multiply(-lamu2p, fu2p)) - (1/tau)
        rpp = rpri + s*Adx

        while (np.linalg.norm(np.concatenate(rdp, rcp, rpp)) > (1-alpha*s)*resnorm):
            s = beta*s
            xp = x +s*dx
            up = u + s*du
            vp = v + s * dv
            Atvp = Atv + s*Atdv
            lamu1p = lamu1 + s*dlamu1
            lamu2p = lamu2 + s*dlamu2
            fu1p = xp - up
            fu2p = -xp - up
            rdp = gradf0 + np.concatenate(lamu1p - lamu2p, -lamu1p - lamu2p) + np.concatenate(Atvp, np.zeros(N, 1))
            rcp = np.concatenate(np.multiply(-lamu1p, fu1p), np.multiply(-lamu2p, fu2p)) - (1/tau)
            rpp = rpri + s*Adx
            backiter = backiter + 1
            if (backiter > 32):
                xp = x
                return xp

        x = xp
        u = up
        v = vp
        Atv = Atvp
        lamu1 = lamu1p
        lamu2 = lamu2p
        fu1 = fu1p
        fu2 = fu2p
        sdg = -(fu1.conj().transpose() * lamu1 + fu2.conj().transpose()*lamu2)
        tau = mu*2*N/sdg
        rpri = rpp
        rcent = np.concatenate(np.multiply(-lamu1, fu1), np.multiply(-lamu2, fu2)) - (1/tau)
        rdual = gradf0 + np.concatenate(lamu1 - lamu2, -lamu1 - lamu2) + np.concatenate(Atv, np.zeros(N, 1))
        resnorm = np.linalg.norm(np.concatenate(rdual, rcent, rpri))
        done = np.logical_or((sdg < pdtol), (pditer >= pdmaxiter))

    return xp
