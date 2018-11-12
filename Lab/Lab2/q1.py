#!/usr/bin/python

import cv2
import numpy as np

# image = cv2.imread(noisyImg_1.jpg)


def convolve(image, kernel):
    # spatial dimension of image and kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # allocate memory for the output image, taking care to
    # "pad" the borders of the input image so the spatial size are unchanged
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                               cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")
    # from left to right, top to bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):

            # extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # same dimension with kernel
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # perform the actual convolution by taking the
            # element-wise multiplication between the ROI(Range of interest, Or the borders of an object)
            # and the kernel, then summing the matrix
            k = (roi * kernel).sum()

            # store the output value into output
            output[y - pad, x - pad] = k

            # # rescale the output image to be in the range [0, 255]
            # output = rescale_intensity(output, in_range=(0, 255))
            # # convert image back to an unsigned 8-bit integer
            # output = (output * 255).astype("uint8")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            print(output[y - pad, x - pad])

    # print the output matrix
    for y in np.arange(pad, iH + pad):
        x = 1
        print(''.join(str(output[y - pad, x - pad: x - pad + iW])))


# kernel
f = np.array([[1, 1, 1],
     [2, 2, 2],
     [-1, -1, -1]])

# Image
inImage = np.array([[3, 0, 1, 5, 0],
     [4, 3, 0, 3, 0],
     [2, 4, 1, 0, 6],
     [3, 0, 1, 5, 0]])

convolve(inImage, f)
