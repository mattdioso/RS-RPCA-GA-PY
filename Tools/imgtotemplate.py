#!/usr/bin/env python3
from imgtomat import imgtomat
import numpy as np
import pandas
import cv2

def imgtotemplate(path, filename, start_frame, end_frame):
    S = imgtomat(path, filename, start_frame, end_frame)

    pathAndFileName = path + filename
#    pathAndFileName = pathAndFileName.replace('\', '\\')

    num_frames = S.shape[1]

    C = 2


    T = np.zeros(num_rows, num_cols, num_frames)

    sparse_sigma = lambda x : np.std(np.float32(S[:, x]))
    sparse_mean = lambda x : np.mean(np.float32(S[:, x]))

    for k in range(0, num_frames):
        T[:, k] = ((np.float32(S[:, k]) > (sparse_mean(k) + C*sparse_sigma(k))) | (np.float32(S[:, k]) < (sparse_mean(k) - C*sparse_sigma(k))))

    T = pandas.to_numeric(T)

    ThresholdpathAndFileName= pathAndFileName.replace("IMG_", "IMG_Th_")

    for frame in range(0, num_frames):
        current_filename = ThresholdPathAndFileName + frame
        image = T[:, frame].reshape(num_rows, num_cols)
        cv2.imwrite(current_filename, image)

    strelShape = 'disk'
    strelSize = 100

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, strelSize)
    M = np.zeros(num_rows*num_cols, num_frames)

    for imN in range(0, num_frames):
        template = T[:, imN].reshape(num_rows, num_cols)
        m1 = cv2.erode(template, 'majority', 1)
        m2 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, se)
        M[:, imN] = m2[:]

    M = pandas.to_numeric(M)

    TemplatepathAndFileName = pathAndFileName.replace("IMG_", "IMG_T_")

    for frame in range(0, num_frames):
        current_filename= TemplatePathAndFileName + frame
        image = M[:, frame].reshape(num_rows, num_cols)
        cv2.imwrite(current_filename, image)


