import argparse
import random
import cv2
import numpy as np
from sys import argv

parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--output')


# A funciton to randomly define centers
def random_centers():

    ini_centers = []
    
    for x in range(k):
        ini_centers.append(random.randint(0, 256))
    return ini_centers


def get_sum_of_distance(h, w, pre_dis):

    means = []
    ini_centers = [-1, -1]
    first_cluster = []
    second_cluster = []

    if pre_dis is None:
        ini_centers = random_centers()
    else:
        ini_centers[0] = pre_dis[0]
        ini_centers[1] = pre_dis[1]
    for h1 in range(h):
        for w1 in range(w):

            a = np.square(my_input[h1][w1] - ini_centers[0])
            b = np.square(my_input[h1][w1] - ini_centers[1])
            # If a > b, the current pixel should belong to the second cluster
            if a >= b:
                second_cluster.append(my_input[h1][w1])
            else:
                first_cluster.append(my_input[h1][w1])
    # calculate the mean for both clusters
    mean1 = np.mean(first_cluster)
    mean2 = np.mean(second_cluster)
    means.append(mean1)
    means.append(mean2)
    return means


def k_means(image):

    pre_distance = None
    arr_distance = get_sum_of_distance(iH, iW, pre_distance)
    pre_distance = [-1, -1]
    while (pre_distance[0] != arr_distance[0] or pre_distance[1] != pre_distance[1]) or pre_distance is None:
        pre_distance = arr_distance[:]
        arr_distance = get_sum_of_distance(iH, iW, pre_distance)
    avg = np.mean(arr_distance)

    # simply threshold 
    for h1 in range(iH):
        for w1 in range(iW):
            # 255 is background color
            if image[h1][w1] > avg:
                output_binary[h1][w1] = 255
            else:
                output_binary[h1][w1] = 0
    

if __name__ == "__main__":
    # Get the length of arguments
    total = len(argv)
    my_input = cv2.imread(argv[total - 2], 0)
    # Use Gaussian Blur before 
    my_input = cv2.GaussianBlur(my_input, (5, 5), 0)
    (iH, iW) = my_input.shape[:2]
    output_binary = np.zeros((iH, iW), dtype="float32")
    
    # k is the number of clusters
    k = 2
    k_means(my_input)
    kernel = np.ones((5, 5), np.uint8)
    output_binary_filter = cv2.morphologyEx(output_binary, cv2.MORPH_OPEN, kernel)
    #output_binary_filter = cv2.medianBlur(output_binary, 3)
    cv2.imwrite(argv[total - 1], output_binary_filter)
