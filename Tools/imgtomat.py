#!/usr/bin/env python3
import matplotlib
from pathlib import Path
from skimage.color import rgb2gray

def imgtomat(path, filename, start_frame, end_frame):

    pathAndFileName = path + filename
#    pathAndFileName = pathAndFileName.replace('\', '\\')
    first_filename = pathAndFileName + start_frame
    X = matplotlib.pyplot.imread(first_filename)
    num_rows = X.shape[0]
    num_cols = X.shape[1]

    imageCount = 0

    for frame in range(start_frame, end_frame):
        current_filename = Path(pathAndFileName + frame)
        if current_filename.is_file():
            imageCount = imageCount + 1

    X = np.zeros(num_rows * num_cols, imageCount -1)
    imageCount = 0

    for frame in range(start_frame, end_frame):
        current_filename = Path(pathAndFileName, frame)
        if current_filename.is_file():
            current_image = rgb2gray(matplotlib.pyplot.imread(pathAndFileName + frame))
            X[:, imageCount] = current_image[:]
            imageCount = imageCount + 1


