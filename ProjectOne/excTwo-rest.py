import math

import cv2;
from matplotlib import pyplot as plt
import convertor
import numpy as np

img = cv2.imread("assets/2-lowContrast.png")

# Question Two.Two
# https://www.geeksforgeeks.org/negative-transformation-of-an-image-using-python-and-opencv/
# method: Steps for negative transformation
# 1- Read an image
# 2- Get height and width of the image
# 3- Each pixel contains 3 channels. So, take a pixel value and collect 3 channels in 3 different variables.
# 4- Negate 3 pixels values from 255 and store them again in pixel used before.
# 5- Do it for all pixel values present in image.


# Read an image
img = cv2.imread('assets/2-lowContrast.png', 1)
plt.imshow(img)
plt.show()

# Histogram plotting of the image
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)

    # Limit X - axis to 256
    plt.xlim([0, 256])

plt.show()

# power law pointwise transform
# get height and width of the image
height, width, _ = img.shape

for i in range(0, height - 1):
    for j in range(0, width - 1):
        pixel = img[i, j]

        pixel[0] = 255 - pixel[0]
        pixel[1] = 255 - pixel[1]
        pixel[2] = 255 - pixel[2]
        img[i, j] = pixel

# Display the negative transformed image
plt.imshow(img)
plt.show()

# Histogram plotting of the
# negative transformed image
color = ('b', 'g', 'r')

for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])

plt.show()

S = np.power(img, 2)
cv2.imshow('gamma', S)
cv2.waitKey(0)
cv2.destroyAllWindows()