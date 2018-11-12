import numpy as np
import cv2
import sys


def getsize(img):
    h, w = img.shape[:2]
    return w, h


class image_feature_detector(object):
    SIFT = 0
    SURF = 1

    def __init__(self, feat_type, params=None):
        self.detector, self.norm = self.features_detector(feat_type=feat_type, params=params)

    def features_detector(self, feat_type = SIFT, params=None):

        if feat_type == self.SIFT:

            if params is None:
                nfeatures = 0
                nOctaveLayers = 3
                contrastThreshold = 0.04
                edgeThreshold = 10
                sigma = 1.6
            else:
                nfeatures = params["nfeatures"]
                nOctaveLayers = params["nOctaveLayers"]
                contrastThreshold = params["contrastThreshold"]
                edgeThreshold = params["edgeThreshold"]
                sigma = params["sigma"]

            detector = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04,
                                                       edgeThreshold=10, sigma=1.6)
            norm = cv2.NORM_L2
        elif feat_type == self.SURF:

            if params is None:
                hessianThreshold = 3000
                nOctaves = 1
                nOctaveLayers = 1
                upright = True
                extended = False
            else:
                hessianThreshold = params["hessianThreshold"]
                nOctaves = params["nOctaves"]
                nOctaveLayers = params["nOctaveLayers"]
                upright = params["upright"]
                extended = params["extended"]

            detector = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold, nOctaves=nOctaves,
                                                   nOctaveLayers=nOctaveLayers,
                                                   upright=upright,
                                                   extended=extended)
            norm = cv2.NORM_L2

        return detector, norm


if __name__ == "__main__":
  

    # SIFT
    sift_detect = image_feature_detector(feat_type=0)

    image_1 = cv2.imread("input_1.jpg", 1) # piakqiu
    gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    sift_1, nor_1 = sift_detect.features_detector(0, None)
    kp_1, des_1 = sift_1.detectAndCompute(gray_1, None) 
    output_1 = cv2.drawKeypoints(gray_1, kp_1, image_1)
    cv2.imwrite('sift_keypoints_1.jpg', output_1)

    image_2 = cv2.imread("input_2.jpg", 1) # stuttgart
    gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    sift_2, nor_2 = sift_detect.features_detector(0, None)
    kp_2, des_2 = sift_2.detectAndCompute(gray_2, None)
    output_2 = cv2.drawKeypoints(gray_2, kp_2, image_2)
    cv2.imwrite('sift_keypoints_2.jpg',output_2)


    # SURF
    surf_detect = image_feature_detector(feat_type=1)

    image_1 = cv2.imread("input_1.jpg", 1) # piakqiu
    gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    surf_1, nor_1 = surf_detect.features_detector(1, None)
    kp_1, des_1 = surf_1.detectAndCompute(gray_1, None) 
    output_1 = cv2.drawKeypoints(gray_1, kp_1, image_1)
    cv2.imwrite('surf_keypoints_1.jpg', output_1)

    image_2 = cv2.imread("input_2.jpg", 1) # stuttgart
    gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    surf_2, nor_2 = surf_detect.features_detector(1, None)
    kp_2, des_2 = surf_2.detectAndCompute(gray_2, None)
    output_2 = cv2.drawKeypoints(gray_2, kp_2, image_2)
    cv2.imwrite('surf_keypoints_2.jpg',output_2)
    
    # Implement yourself
    # Refer to opencv documentation,
    # use SIFT or SURF on the test image and show detection result.
    # 1. Read Image
    # 2. Convert the image to greyscale
    # 3. Initialize an SIFT detector
    # 4. Feature detection and visualize the detected features


    # The advantage for SURF is that the convolution with box filters
    # may be done easily with the help of integral images

    # detector is based on DoG, descriptor is based on a histogram 
    # of gradient orientations
