#!/usr/bin/env python3
import matplotlib
import numpy as np

def Accuracy(path, filename, start_frame, end_frame):
    imageCount = 1
    for frame in range(start_frame, end_frame):
        temp = filename % frame
        template_filename = path + 'RS-RPCA_T_' + temp
        groundTruth_filename = path + '/GroundTruth/' + temp.replace('.PNG', '_BW.PNG')
        BW_groundTruth = matplotlib.pyplot.imread(groundTruth_filename)
        BW = matplotlib.pyplot.imread(template_filename)

        X = BW_groundTruth
        Y = BW
        NX = np.logical_not(X)
        NY = np.logical_not(Y)

        Alpha = 1
        Beta = 1

        Tversky = np.sum(np.multiply(X, Y)) / ((np.sum(np.multiply(X, Y)) + Alpha * (np.sum(np.multiply(X, NY))) + Beta * np.sum(np.multiply(NX, Y))))
        a[imageCount] = Tversky
        imageCount = imageCount + 1



    return a
