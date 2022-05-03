import cv2;
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread("assets/2-lowContrast.png")

# Question Two

# Read an image
imgLowContrast = cv2.imread("assets/2-lowContrast.png")
imgHighContrast = cv2.imread("assets/2-highContrast.png")
cv2.imshow('Original Low contrast image', imgLowContrast)
cv2.imshow('Original High contrast image', imgHighContrast)
imgGrayLow = cv2.cvtColor(imgLowContrast, cv2.COLOR_BGR2GRAY)
imgGrayHig = cv2.cvtColor(imgHighContrast, cv2.COLOR_BGR2GRAY)

# Task Two
# Negative pointwise transformation
imgNegLow = np.abs(255 - imgGrayLow)
imgNegHig = np.abs(255 - imgGrayHig)

cv2.imshow('Negative transformed of low contrast', imgNegLow)
cv2.imshow('Negative transformed of High contrast', imgNegHig)

# task three
# histogram of transformed image
plt.hist(imgNegLow.ravel(), 256, [0, 256])
plt.title("Low")
plt.show()

plt.hist(imgNegHig.ravel(), 256, [0, 256])
plt.title("High")
plt.show()

# task Four
# # power law pointwise transform
imgPowLow2 = np.power(imgGrayLow, 2)
imgPowHig2 = np.power(imgGrayHig, 2)
imgPowLow5 = np.power(imgGrayLow, 0.5)
imgPowHig5 = np.power(imgGrayHig, 0.5)

cv2.imshow('Power Law  low contrast 2', imgPowLow2)
cv2.imshow('Power Law  High contrast 2', imgPowHig2)
cv2.imshow('Power Law  low contrast 0.5', imgPowLow5)
cv2.imshow('Power Law  High contrast 0.5', imgPowHig5)
cv2.waitKey(0)
cv2.destroyAllWindows()