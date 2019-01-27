#!/usr/bin/env python3
import numpy as np
import math

def rsalm(X, col_subrate, row_subrate):
    print('   Running column ALM.... ')
    L, S = run_column_alm(X, col_subrate)

    print('   Running SVD...  ')
    U, Sigma, V = svd(L, 'econ')

    print('    Running row 11... ')
    Q_hat = run_row_11(X, U, row_subrate)

    L = U * Q_hat
    S = X - L

    return L, S

def run_column_alm(D, col_subrate):
    num_rows = D.shape[0]
    num_cols = D.shape[1]

    num_cols2 = round(num_cols * cols_subrate)
    I_columns = np.random.permutation(num_cols)
    I_columns = I_columns[0:num_cols2]

    n = max(num_rows, num_cols2)
    lamda = 1/math.sqrt(n)

    L, S = inexact_alm_rpca(D[:, I_columns], lamda)
    
    return L, S

def run_row_11(D, U, row_subrate):
    num_rows = D.shape[0]
    num_cols = D.shape[1]

    num_rows2 = round(num_rows*row_subrate)
    I_rows = np.random.permutation(num_rows)
    I_rows = I_rows[:, num_rows2]
    D = D[I_rows, :]
    U = U[I_rows, :]

    M = U.shape[1]

    A = lambda q : (U * (q.reshape(M, num_cols)).reshape(num_rows2*num_cols, 1))

    At = lambda y : (U * (y.reshape(num_rows2, num_cols)).reshape(M*num_cols, 1))

    Q_hat = (l1decode_pd(At(D[:]), A, At, D[:])).reshape(M, num_cols)

    return Q_hat


