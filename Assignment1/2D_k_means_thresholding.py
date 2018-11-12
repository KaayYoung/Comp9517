import argparse
import random
import cv2
import numpy as np
from sys import argv

parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--size')
parser.add_argument('--output')


# randomly set up centers
def set_random():
    ini_centers = []
    for unit in range(4):
        ini_centers.append(random.randint(0, 256))
        print(str(ini_centers[unit]))
    return ini_centers

# calculate local neighbourhood averages
def cal_nei_avg(n, h1, w1):

    avg_area = 0
    for col in range(h1, n + h1):
        for row in range(w1, n + w1):
            avg_area += new_matrix[col][row]
    avg_area = avg_area / (n * n)

    return avg_area


def two_d_kmeans(n):

    pixel_value = [0, 0, 0, 0]
    
    clus_1 = []
    clus_2 = []
    for row in range(iH):
        for col in range(iW):
            
            # ini_centers[0] and ini_centers[1] is for cluster 1
            pixel_value[0] = np.square(three_matrix[row][col][0] - ini_centers[0])
            pixel_value[1] = np.square(three_matrix[row][col][1] - ini_centers[1])
            
            # ini_centers[2] and ini_centers[3] is for cluster 2
            pixel_value[2] = np.square(three_matrix[row][col][0] - ini_centers[2])
            pixel_value[3] = np.square(three_matrix[row][col][1] - ini_centers[3])

            result_1 = np.sqrt(pixel_value[0] + pixel_value[1])
            result_2 = np.sqrt(pixel_value[2] + pixel_value[3])

            if result_1 > result_2:
                clus_2.append([three_matrix[row][col][0], three_matrix[row][col][1]])
            else:
                clus_1.append([three_matrix[row][col][0], three_matrix[row][col][1]])

    clus_1 = np.array(clus_1)
    clus_2 = np.array(clus_2)

    means_1 = np.mean(clus_1, axis=0)
    means_2 = np.mean(clus_2, axis=0)

    # these variables are storing current values, aiming to compare with previous values
    now_val_1 = means_1[0]
    now_val_2 = means_1[1]
    now_val_3 = means_2[0]
    now_val_4 = means_2[1]


    if now_val_1 == pre[0] and now_val_2 == pre[1] and now_val_3 == pre[2] and now_val_4 == pre[3]:
        global output_binary
        avg = (now_val_1 + now_val_3)/2
        for h1 in range(iH):
            for w1 in range(iW):

                if my_input[h1][w1] > avg:
                    output_binary[h1][w1] = 255
                else:
                    output_binary[h1][w1] = 0

        output_binary = np.array(output_binary)
        cv2.imwrite(argv[total - 1], output_binary)
    else:
        pre[0] = ini_centers[0] = means_1[0]
        pre[1] = ini_centers[1] = means_1[1]
        pre[2] = ini_centers[2] = means_2[0]
        pre[3] = ini_centers[3] = means_2[1]
        # If each of them are not equal, do recursion
        two_d_kmeans(n)


if __name__ == "__main__":

    # global variables
    total = len(argv)
    my_input = cv2.imread(argv[total - 3], 0)
    size = argv[total - 2]
    size = int(size)

    (iH, iW) = my_input.shape[:2]
    output_binary = np.zeros((iH, iW), dtype="float32")

    pre = [0, 0, 0, 0]
    inten_mean_1 = 0
    inten_mean_2 = 0

    # new matrix for calculating neighbour
    new_matrix = [[0 for x in range(size + iW)] for y in range(size + iH)]
    for a in range(iH):
        for b in range(iW):
            new_matrix[a][b] = my_input[a][b]

    # 3D matrix
    #three_matrix = [[[0 for x in range(iW)] for y in range(iH)] for z in range(2)]
    three_matrix = np.zeros((iH, iW, 2))
    for a in range(iH):
        for b in range(iW):
            three_matrix[a][b][0] = my_input[a][b]
            three_matrix[a][b][1] = cal_nei_avg(size, a, b)

    # run functions
    ini_centers = set_random()
    two_d_kmeans(size)
