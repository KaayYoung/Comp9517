#!/usr/bin/python

import cv2
import numpy as np

image = cv2.imread('4.png', 0)

for i in range(512):
    for j in range(512):
        image[i][j] = image[i][j] * 7 / 255


cv2.imshow('new image 4', image)
cv2.waitKey(0)

