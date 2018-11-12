import cv2
import argparse
from sys import argv
import random
import numpy as np


# Prepocessing images: gray image and Gaussian 
def image_extract_feature(img_name):

    gray_img = cv2.cvtColor(img_name, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray_img, None)
    # (21, 21) is the width and height of the kernel which should be positive and odd
    # The goal is to remove noise
    gray_img = cv2.GaussianBlur(gray_img, (21, 21), 0)

    return gray_img, kp, des


# Match two images
def image_matches(img_1, img_2, contain, ok, n_matches=10):

    # Have copy because we need to draw
    # rectangles and tracing lines on original imags
    copy_1 = img_1.copy()
    copy_2 = img_2.copy()

    # Find keypoints and destinations on gray image
    gray_1, gray_kp1, gray_des1 = image_extract_feature(img_1)
    gray_2, gray_kp2, gray_des2 = image_extract_feature(img_2)

    global centers_1, centers_2, centers_3

    i = 0
    
    if ok:
        for box in contain:
            # get the corners of box
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            # center value for tracing
            center = (int((p1[0] + p2[0]) / 2), (int((p1[1] + p2[1]) / 2)))
            center = np.array(center)
            if i == 0:
                centers_1.append(center)
                center = center.reshape(-1, 1, 2)
            if i == 1:
                centers_2.append(center)
                center = center.reshape(-1, 1, 2)
            if i == 2:
                centers_3.append(center)
                center = center.reshape(-1, 1, 2)
            
            # Draw tracing lines for the three objects
            cv2.polylines(copy_1, np.int32([centers_1]), False, (220, 130, 50), 2)
            cv2.polylines(copy_1, np.int32([centers_2]), False, (193, 19, 130), 2)
            cv2.polylines(copy_1, np.int32([centers_3]), False, (24, 105, 230), 2)

            # If first time come into, we randomly choose a color for this rectangle
            # Later on rectangles heritate colors
            if first[i]:
                box_color[i] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                cv2.rectangle(copy_1, p1, p2, (box_color[i]), 2, 1)
                # first Variable set for coming in this loop once only
                first[i] = False
                i += 1
            else:
                cv2.rectangle(copy_1, p1, p2, (box_color[i]), 2, 1)
                i += 1
    else:
        cv2.putText(copy_1, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    
    # Apply ratio test
    # BFMatcher, it takes two optional params. First one is normType, it specifies the 
    # distance measurement to be used.boolean  cv.NORM_L2 is good for SIFT or SURF
    # For ORB, cv2.NORM_HAMMING should be used
    # second param is boolean variable, crossCheck return good match if it is true
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(gray_des1, gray_des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img_matches = cv2.drawMatchesKnn(copy_1, gray_kp1, copy_2, gray_kp2, good, None, flags=2)

    return img_matches


# Select ROI
def ROI_select(img):

    container = []
    obj_1 = cv2.selectROI(img)
    container.append(obj_1)
    obj_2 = cv2.selectROI(img)
    container.append(obj_2)
    obj_3 = cv2.selectROI(img)
    container.append(obj_3)

    return container


def track_object(video):
    cam = cv2.VideoCapture(video)

    if cam.isOpened():
        print('Open')
    else:
        print('not opened')
        exit()

    # Template image as a object to compare with
    # Won't change after initialize
    background = None

    # Define multitracker
    tracker = cv2.MultiTracker_create()
    
    while 1:
        ret, frame = cam.read()
        
        if not ret:
            print ("Cannot capture ret of frame")
            exit()

        copy_f = frame.copy()

        if background is None:
            background = frame

            # Only select once
            obj_contain = ROI_select(copy_f)

            # Initialize trackers, add three KCF trackers
            # bool add(String trackerType, Mat Image, Rec2D boundingbox)
            success = tracker.add(cv2.TrackerKCF_create(), frame, obj_contain[0])
            success = tracker.add(cv2.TrackerKCF_create(), frame, obj_contain[1])
            success = tracker.add(cv2.TrackerKCF_create(), frame, obj_contain[2])

            matched = image_matches(frame, background, obj_contain, success, 10)
            matched = cv2.resize(matched, (1200, 400), interpolation=cv2.INTER_CUBIC)
            
        # Find matched points, draw tracing lines and rectangles

        cv2.imshow("Match result", matched)

        # update: (x, y)-coordinates of existing objects
        # Register new objects
        # Deregister old objects
        success, obj_contain = tracker.update(frame)

        matched = image_matches(frame, background, obj_contain, success, 10)
        matched = cv2.resize(matched, (1200, 400), interpolation=cv2.INTER_CUBIC)
        # waitKey(1) will display a frame for 1 ms
        key = cv2.waitKey(1) & 0xFF

        # Press q for quitting
        if key == ord('q'):
            break

    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', "--input")
    # input video
    inputVideo = argv[1]
    
    # obj_contain: contain boxes
    # first: control 3 selected objects whether first time go into loop
    # box_color: each box has different color
    global obj_contain, first, box_color
    # Below global variables are for tracing(trajectories)
    global centers_1, centers_2, centers_3
    
    centers_1 = []	
    centers_2 = []
    centers_3 = []
    obj_contain = []
    first = [True, True, True]
    box_color = [None, None, None]

    # orb descriptor
    # Oriented FAST ()and rotated BRIEF
    orb = cv2.ORB_create(400)
    track_object(inputVideo)
