#!/usr/bin/python

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('4.png', 0)

#histogram1 = [0] * 255
histogram2 = np.zeros(255)

for i in range(512):
    for j in range(512):
        #bin_number = int(image[i][j] / 30)
        histogram2[image[i][j]] += 1

# for i in range(len(histogram2)):
#     print(histogram2[i])

plt.hist(histogram2, bins = [1, 30, 60, 90, 120, 150, 180, 210, 240])
plt.show()
