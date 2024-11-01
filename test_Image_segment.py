import cv2
import numpy as np
import matplotlib.pyplot as plt
import AHC1 as hc

# read the image
image = cv2.imread("table.jpg")
print(image.shape)
print('************')

# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)

# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = image.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)

print(pixel_values.shape)

# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# number of clusters (K)
k = 3
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
hc_labels = hc.hierarchical_clustering(pixel_values, k)


# convert back to 8 bit values
centers = np.uint8(centers)
print(centers.shape)

# flatten the labels array
labels = labels.flatten()
print(labels.shape)

# convert all pixels to the color of the centroids
segmented_image = centers[labels.flatten()]
print(segmented_image.shape)

# reshape back to the original image dimension
segmented_image = segmented_image.reshape(image.shape)
# show the image
plt.imshow(segmented_image)
plt.show()
