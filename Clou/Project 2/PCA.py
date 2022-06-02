import glob
import cv2 as c
import numpy as np


def getMatrixData(variations):
    matrix = np.matrix([var.flatten() for var in variations])
    return matrix


def computePCA(data):
    mean, vectors = c.PCACompute(data, mean= None)
    # getting array of length of the # of eigenvectors, 1:1 correspondance
    weights = getWeights(data, vectors, mean)
    eigenFace = vectors * weights
    # we want the weighted sum of all eigenvectors
    eigenFace = eigenFace.sum()
    reFace = mean + eigenFace
    r = reFace.astype(np.uint8)
    return r


# similar procedure that when we recontruct one face
# face A and face B
def combineFacesPCA(dA, dB):
    meanA, vA = c.PCACompute(dA, mean=None)
    meanB, vB = c.PCACompute(dB, mean=None)
    # get the weights by using the mean of A and the data and eigenvectors of B
    weights = getWeights(dB, vB, meanA)
    eigenF = vB * weights
    new = meanA + eigenF.sum()
    return new.astype(np.uint8)


def getDifference(face, mean):

    # face = np.squeeze(np.asarray(face), axis= 0)
    # mean = np.squeeze(np.asarray(mean), axis= 0)
    diff = face - mean
    return diff


# weights are computed by computing the difference between the
# variation and the mean face and then multiplied by their corresponding eigenV
def getWeights(d, v, m):
    w = []
    for i in range(d.shape[0]):
        diff = getDifference(d[i], m)
        w.append(np.dot(diff, v[i]))
    return w



# saving all variations/face into a list of images
dis_F1 = []
for img in glob.glob("images project 2/womanVariation/*.jpeg"):
    dis_F1.append(c.imread(img))


dis_F2 = []
for img in glob.glob("images project 2/boyVariations/*.jpeg"):
    dis_F2.append(c.imread(img))

dis_F3 = []
for img in glob.glob("images project 2/manVars/*.jpeg"):
    dis_F3.append(c.imread(img))

# get the variations matrices
dF1 = getMatrixData(dis_F1)
dF2 = getMatrixData(dis_F2)
dF3 = getMatrixData(dis_F3)

mean_1, vectors = c.PCACompute(dF1, mean= None)
mean_2, vectors = c.PCACompute(dF2, mean= None)
mean_3, vectors = c.PCACompute(dF3, mean= None)
c.imshow("Task 4.1-woman", mean_1.astype(np.uint8).reshape(1090, 1050, 3))
c.imshow("Task 4.1-boy", mean_2.astype(np.uint8).reshape(1556, 1600, 3))
c.imshow("Task 4.1-man", mean_3.astype(np.uint8).reshape(1090, 1050, 3))

# get the eigenvectors, values
womanPCA = computePCA(dF1).reshape(1090, 1050, 3)
boyPCA = computePCA(dF2).reshape(1556, 1600, 3)
manPCA = computePCA(dF3).reshape(1090, 1050, 3)

c.imshow("Task 4.2-woman", womanPCA.reshape(1090, 1050, 3))
c.imshow("Task 4.2-boy", boyPCA.reshape(1556, 1600, 3))
c.imshow("Task 4.2-man", manPCA.reshape(1090, 1050, 3))
#
# womanPCA = c.resize(womanPCA, (400, 400))
womanManPCA = combineFacesPCA(dF1, dF3).reshape(1090, 1050, 3)
c.imshow("Task 4.3", womanManPCA.reshape(1090, 1050, 3))


c.waitKey(0)
c.destroyAllWindows()
