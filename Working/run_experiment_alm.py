#!/usr/bin/env python3
import time
import numpy as np

def run_experiment_alm(X, filename):

    num_frames = X.shape[1]

    print('Running ALM...')
    start = time.time()
    L, S = inexact_alm_rpca(double(X))

    stop = time.time()-start

    U, Sigma, V = np.linalg.svd(L)

    L = np.int8(L)
    S = np.int8(S + 127)

    L_sequence = "Results/" + filename + ".L.alm.%03d.pgm"
    L_Video = "Results/" + filename + ".L.alm.mp4"
    S_sequence = "Results/" + filename + ".S.alm.%03d.pgm"
    S_video = "Results/" + filename + ".S.alm.mp4"

    mattoimg(L, num_rows, num_cols, L_sequence)
    mattoimg(S, num_rows, num_cols, L_sequence)
    make_video(L_sequence, L_video)
    make_video(S_sequence, S_video)
