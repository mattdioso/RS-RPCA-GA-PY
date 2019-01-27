#!/usr/bin/env python3

def LineMask(L, N):
    thc = np.linspace(0, math.pi-(math.pi-L), L)

    M = np.zeros(N)

    for l1 in range(0, L):
        if np.logical_or(thc[l1] <= math.pi/4, thc[l1] > 3*math.pi/4):
            yr = round(math.tan(thc[l1]) * list(range((-N/2+1), (N/2-1)))) + N/2+1

            for nn in range(0, N-2):
                M[yr[nn], nn+1] = 1

        else:
            xc = round(math.cot(thc[l1]) * list(range((-N/2+1), (N/2-1)))) + N/2 + 1

            for nn in range(0, N-2):
                M[nn+1, xc[nn]] = 1

    Mh = np.zeros(N)
    Mh = M

    Mh[N/2+2:N, :] = 0
    Mh[N/2+1, N/2+1:N] = 0

    M = np.fft.ifftshift(M)
    mi = np.nonzero(M)
    Mh = np.fft.ifftshift(Mh)
    mhi = np.nonzero(Mh)

    return M, Mh, mi, mhi
