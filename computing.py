import numpy as np
import cv2
from sklearn.feature_extraction.image import grid_to_graph
import AHC1 as AggloCluster


def main_computing(file_path):
    # number of clusters
    k = 3

    # read the image
    image = cv2.imread(file_path)

    # convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # convert to greyscale
    grey_pic = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    X = np.reshape(grey_pic, (-1, 1))

    # Define the structure A of the data. Pixels connected to their neighbors.
    connectivity = grid_to_graph(*grey_pic.shape)

    # Compute clustering
    print("Compute structured hierarchical clustering...")
    # number of regions
    hc_result, hc_eval = AggloCluster.hierarchical_clustering(X, k, connectivity)
    # reshape labels structure to appropriate shape
    hc_label = np.reshape(hc_result, grey_pic.shape)
    hc_label_reshape = hc_label.reshape((-1, 1))

    # Compute clustering
    print("Compute structured K-Means clustering...")
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # number of clusters (K)
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels2 = hc_label_reshape.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels2.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)

    # show the image
    # plt.imshow(segmented_image)
    # plt.show()

    return segmented_image

# individual running
# main_computing('Belha Ciao.jpg')
