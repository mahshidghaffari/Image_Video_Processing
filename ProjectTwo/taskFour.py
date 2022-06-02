import glob

import cv2
import numpy as np


#  getting the weights of all images
def getWeights(data, eig_vec, mean):
    weights = []
    for i in range(data.shape[0]):
        img = np.squeeze(np.asarray(data[i]), axis=0)
        weight = np.dot(img - mean, eig_vec[i])
        weights.append(weight)
    return weights


def getEigFace(eigVec, wights, mean):
    eigFaces = eigVec * wights
    eigFaces = eigFaces.sum()
    newFace = eigFaces + mean
    return newFace.astype(np.uint8)


# reading data
# kid pictures
var_one = []
for pic in glob.glob("asset/faces/1/*.jpg"):
    var_one.append(cv2.imread(pic))

# man pictures
var_two = []
for pic in glob.glob("asset/faces/2/*.jpg"):
    var_two.append(cv2.imread(pic))

#  old-woman pictures
var_three = []
for pic in glob.glob("asset/faces/3/*.jpg"):
    var_three.append(cv2.imread(pic))

# Task 4.1 -  Find the eigen faces for each image
# convert all image to row
data_a = np.matrix([img.flatten() for img in var_one])
data_b = np.matrix([img.flatten() for img in var_two])
data_c = np.matrix([img.flatten() for img in var_three])

# calculate mean and eigen vector for each batch of images
mean_a, eigVec_a = cv2.PCACompute(data_a, mean=None)
mean_b, eigVec_b = cv2.PCACompute(data_b, mean=None)
mean_c, eigVec_c = cv2.PCACompute(data_c, mean=None)

cv2.imshow("Task 4.1 => Image 1", mean_a.astype(np.uint8).reshape((1280, 853, 3)))
cv2.imshow("Task 4.1 => Image 2", mean_b.astype(np.uint8).reshape((1280, 853, 3)))
cv2.imshow("Task 4.1 => Image 3", mean_c.astype(np.uint8).reshape((960, 1280, 3)))

# Task 4.2 - Reconstruct each face
weights_a = getWeights(data_a, eigVec_a, mean_a)
weights_b = getWeights(data_b, eigVec_b, mean_b)
weights_c = getWeights(data_c, eigVec_c, mean_c)

newFace_a = getEigFace(eigVec_a, weights_a, mean_a)
newFace_b = getEigFace(eigVec_b, weights_b, mean_b)
newFace_c = getEigFace(eigVec_c, weights_c, mean_c)

cv2.imshow("Task 4.2 => Image 1", newFace_a.reshape((1280, 853, 3)))
cv2.imshow("Task 4.2 => Image 2", newFace_b.reshape((1280, 853, 3)))
cv2.imshow("Task 4.2 => Image 3", newFace_c.reshape((960, 1280, 3)))


# Task 4.3 Reconstruct each face using ’wrong’ PCA weights:
ws_diff = getWeights(data_b, eigVec_b, mean_a)
newFace_diff = getEigFace(eigVec_b, ws_diff, mean_a)
img = newFace_diff.reshape(1280, 853, 3)
cv2.imshow("Task 4.3 => Image diff", newFace_c.reshape((960, 1280, 3)))

cv2.waitKey(0)
cv2.destroyAllWindows()
