import time as time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

# Generate data
img = cv2.imread('Belha Ciao.jpg')
# convert to greyscale
lena = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Downsample the image by a factor of 4
# lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]

# reshape to appropriate dimension
X = np.reshape(lena, (-1, 1))


# Define the structure A of the data. Pixels connected to their neighbors.
connectivity = grid_to_graph(*lena.shape)


# Compute clustering
print("Compute structured hierarchical clustering...")
st = time.time()
# number of regions (clusters)
n_clusters = 5
# compute Agglomerative Clustering Algorithm with ward method
ward = AgglomerativeClustering(n_clusters=n_clusters,
                               linkage='ward', connectivity=connectivity).fit(X)

# reshape the labels
label = np.reshape(ward.labels_, lena.shape)

print("Elapsed time: ", time.time() - st)
print("Number of pixels: ", label.shape)
print("Number of clusters: ", np.unique(label).size)


# Plot the results on an image
plt.figure(figsize=(5, 5))
plt.imshow(lena, cmap=plt.cm.gray)
for l in range(n_clusters):
    plt.contour(label == l, contours=1,
                colors=[plt.cm.Spectral(l / float(n_clusters)), ])
plt.xticks(())
plt.yticks(())
plt.show()
