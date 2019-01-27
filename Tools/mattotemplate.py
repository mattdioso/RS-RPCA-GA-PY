#!/usr/bin/env python3
import numpy as np
import pandas 
import cv2

def mattotemplate(path, filename, start_frame, S, num_rows, num_cols, LC, UC, strelSize, bwmorphIteration):
    pathAndFileName = path + filename
    pathAndFileName = pathAndFileName.replace("JPG", "PNG")

    num_frames = S.shape[1]

    T = np.zeros(num_rows*num_cols, num_frames)
    sparse_sigma = lambda x : np.std(np.float32(S[:, x]))
    sparse_mean = lambda x : np.mean (np.float32(S[:, x]))

    for k in range(0, num_frames):
        T[:, k] = ((np.float32(S[:, k]) > (sparse_mean(k) + UC*sparse_sigma(k))) | (np.float32(S[:, k]) < (sparse_mean(k) - LC*sparse_sigma(k))))

    T = pandas.to_numeric(T)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, strelSize)
    M = np.zeros(num_rows*num_cols, num_frames)

    for imN in range(0, num_frames):
        template = T[:, imN].reshape(num_rows, num_cols)
        m1 = cv2.erode(template, 'majority', bwmorphIteration)
        m2 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, se)
        M[:, imN] = m2[:]

    M = pandas.to_numeric(M)

    for frame in range(0, num_frames):
        current_filename = pathAndFileName + (frame -1) + start_frame
        image = M[:, frame].reshape(num_rows, num_cols)
        cv2.imwrite(current_filename, image)
