#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Tools'))
import imgtomat
import mattotemplate


def run_experiment_rsalm(path, filename, start_frame, row_subrate, LC, UC, strelSize, bwmorphIteration):
    print("Running RS-RPCA")

    col_subrate = 1

    X, num_rows, num_cols = imgtomat(path, filename, start_frame, end_frame)

    L, S = rsalm(double(X), col_subrate, row_subrate)

    S = np.int8(S+ 127)

    T_filename = "RS-RPCA_T_" + filename

    mattotemplate(path, T_filename, start_frame, S, num_rows, num_cols, LC, UC, strelSize, bwmorphIteration)
