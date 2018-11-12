# Adapted from :
# https://gis.stackexchange.com/questions/152853/image-segmentation-of-rgb-image-by-k-means-clustering-in-python

import numpy as np
import sklearn.cluster
import matplotlib.pyplot as plt

from PIL import Image

img_path = "9517_lab6/tm1_1_1.png"
size = 100, 100
num_clusters = 2

img = Image.open(img_path)
img.thumbnail(size)

img_mat = np.array(img)

red = img_mat[:, :, 0]
green = img_mat[:, :, 1]
blue = img_mat[:, :, 2]
original_shape = red.shape # so we can reshape the labels later

samples = np.column_stack([red.flatten(), green.flatten(), blue.flatten()])

# K-Means
clf = sklearn.cluster.KMeans(n_clusters=num_clusters)
labels = clf.fit_predict(samples).reshape(original_shape)

plt.imshow(labels)
plt.show()

# Mean-Shift
ms = sklearn.cluster.MeanShift(cluster_all=False)
ms_labels = ms.fit_predict(samples).reshape(original_shape)

plt.imshow(ms_labels)
plt.show()
