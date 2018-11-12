import argparse
from sys import argv
import cv2
import numpy as np
import random

# define arguments,  main function is at the bottom
parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--size')
parser.add_argument('--output')


# A function to find the subset of an element i
# pseudocode is from disjoint-set wikipedia
def find_parent(parent, i):
    while parent[i] != -1:
        i = parent[i]
    return i


# A function to do union of two subsets
# pseudocode is from disjoint-set wikipedia
def union(parent, a, b):

    a_root = find_parent(parent, a)
    b_root = find_parent(parent, b)

    if a_root == b_root:
        return
    if a_root > b_root:
        parent[a_root] = b_root
    elif a_root < b_root:
        parent[a_root] = b_root


def two_pass(image):

    # label_id indicates how many components
    label_id = 1
    linked = [-1]

    # first pass
    for h1 in range(iH):
        for w1 in range(iW):
            if image[h1][w1] != 255:
                # both pixels to the north and west have same value as the current one
                if h1 > 0 and image[h1][w1] == image[h1 - 1][w1] and w1 > 0 and image[h1][w1] == image[h1][w1 - 1]:
                    if label_matrix[h1 - 1][w1] != label_matrix[h1][w1 - 1]:
                        union(linked, label_matrix[h1 - 1][w1], label_matrix[h1][w1 - 1])
                    label_matrix[h1][w1] = min(label_matrix[h1 - 1][w1], label_matrix[h1][w1 - 1])
                # the pixel to the left has the value to the current one
                elif w1 > 0 and image[h1][w1] == image[h1][w1 - 1]:
                    label_matrix[h1][w1] = label_matrix[h1][w1 - 1]
                # the pixel to the left has different value but the one to the north has the same value
                elif h1 > 0 and image[h1][w1] == image[h1 - 1][w1]:
                    label_matrix[h1][w1] = label_matrix[h1 - 1][w1]
                else:
                    label_matrix[h1][w1] = label_id
                    linked.append(-1)
                    label_id += 1

    # second pass, aiming to find the parent of current pixel
    for h1 in range(iH):
        for w1 in range(iW):
            if image[h1][w1] != 255:
                label_matrix[h1][w1] = find_parent(linked, label_matrix[h1][w1])


def get_num_area():
    
    # get the number of areas  
    num_area = 0
    for h1 in range(iH):
        for w1 in range(iW):
            if label_matrix[h1][w1] not in label_area:
                label_area[label_matrix[h1][w1]] = 1
            else:
                label_area[label_matrix[h1][w1]] += 1

    # then get the the number of areas whose size is bigger than n   
    for key in label_area:
        if label_area[key] > input_size:
            num_area += 1
    print(num_area)


def print_output():

    for h1 in range(iH):
        for w1 in range(iW):
            if label_matrix[h1][w1] == 255:
                output_image[h1][w1] = [255, 255, 255]
            else:
                if label_matrix[h1][w1] not in output_colors:
                    output_colors[label_matrix[h1][w1]] = [random.randint(0, 255), random.randint(0, 255),
                                                           random.randint(0, 255)]
                output_image[h1][w1] = output_colors[label_matrix[h1][w1]]


if __name__ == "__main__":
    # Get the lenght of arguments
    total = len(argv)
    # The third argument should be the input image
    binary_input = cv2.imread(argv[total - 3], 0)
    # The fourth argument should be the minimum number of pixels in the component
    input_size = int(argv[total - 2])
    (iH, iW) = binary_input.shape[:2]

    # simple threshold
    for x in range(iH):
        for y in range(iW):
            if 0 < binary_input[x][y] < 127:
                binary_input[x][y] = 0
            elif 126 < binary_input[x][y] < 256:
                binary_input[x][y] = 255

    # Create matrix for labelling, initialize values are background
    label_matrix = [[255 for col in range(iW)] for row in range(iH)]
    # initialize output image
    output_image = np.zeros((iH, iW, 3), dtype="float32")


    # run two_pass algorithm
    two_pass(binary_input)

    label_area = {}
    
    get_num_area()
    
    output_colors = {}
    print_output()
    # Write the output image into the fifth argument 
    cv2.imwrite(argv[total - 1], output_image)
