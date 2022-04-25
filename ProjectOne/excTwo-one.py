import cv2;
from matplotlib import pyplot as plt

img = cv2.imread("assets/2-lowContrast.png")

# Question one
# https://www.geeksforgeeks.org/opencv-python-program-analyze-image-using-histogram/


# calculate frequency of pixels in range 0-255
histg = cv2.calcHist([img], [0], None, [256], [0, 256])

# show the plotting graph of an image
plt.plot(histg)
# alternative way to find histogram of an image
# plt.hist(img.ravel(), 256, [0, 256])

plt.show()

# cv2.imshow('RGB', img)
