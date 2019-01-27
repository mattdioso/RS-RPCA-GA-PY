#!/usr/bin/env python3
import cv2

def mattoimg(path, filename, X, num_rows, num_cols):

    pathAndFileName = path + filename
    pathAndFileName.replace("JPG", "PNG")

    num_frames = X.shape[1]

    if (num_rows * num_cols != X.shape[0]):
        print("mattoimg: num_rows and num_cols does not match dataset size")
        exit

    for frame in range(0, num_frames):
        current_filename = pathAndFileName + frame
        image = X[:, frame].reshape(num_rows, num_cols)
        cv2.imwrite(current_filename, image)


