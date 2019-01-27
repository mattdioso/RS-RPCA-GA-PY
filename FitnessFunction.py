#!/usr/bin/env python3
import run_tests
import numpy as np
import math
from datetime import date
import os
import time

def FitnessFunction(x):
    
    start = time.time()

    Row_SubRate = x[1]
    LSigma = x[2]
    USigma = x[3]
    StRelSize = round(x[4])
    bwMorph = x[5]

    run_tests('G:/RPCA(GA)/Data/AT004_P016/7-06-12/', 'IMG_%04d.JPG', '0031', '0048', Row_SubRate, LSigma, USigma, StRelSize, bwMorph)
    run_tests('G:/RPCA(GA)/Data/AT006_P038/8-18-12/', 'IMG_%04d.JPG', '0664', '0670', Row_SubRate, LSigma, USigma, StRelSize, bwMorph)

    Accuracy1 = Accuracy('G:/RPCA(GA)/Data/AT004_P016/7-06-12/', 'IMG_%04d.PNG', '0031', '0048')
    Accuracy2 = Accuracy('G:/RPCA(GA)/Data/AT006_P038/8-18-12/', 'IMG_%04d.PNG', '0664', '0670')

    AccuracyBoth = [Accuracy1, Accuracy2]

    n = np.size(AccuracyBoth)
    g = g = sum(AccuracyBoth > 0.8816)
    BIPScore = 2 * g - k * math.log2(n)

    GAaccuracy = 1000 - BIPScore
    
    stop = time.time() - start

    AccuracyLog = {"parameters":x, "Time": stop, "set1":"G:/RPCA(GA)/Data/AT004_P016/7-06-12/ IMG_%04d.JPG 0031 0048", "set2":"G:/RPCA(GA)/Data/AT006_P038/8-18-12/ IMG_%04d.JPG 0664 0670", "GAaccuracy":GAaccuracy, "Accuracy1":Accuracy1, "Accuracy2": Accuracy2, "AccuracyBoth":AccuracyBoth, "BIPScore":BIPScore}

    today = str(date.today())
    name = "/Log/AccuracyDetails/" + dt + "_AccuracyDetails"

    np.save(os.getcwd()+name, AccuracyLog)

    y = [stop, GAaccuracy]

    return y
