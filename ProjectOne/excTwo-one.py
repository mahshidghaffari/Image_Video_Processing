import cv2
from matplotlib import pyplot as plt

imgLowContrast = cv2.imread("assets/2-lowContrast.png")
imgHighContrast = cv2.imread("assets/2-highContrast.png")

# Task one
# show the plotting graph of an image
cv2.imshow('Original Low contrast image', imgLowContrast)
cv2.imshow('Original High contrast image', imgHighContrast)

# calculate frequency of pixels in range 0-255
plt.hist(imgHighContrast.ravel(), 256, [0, 256])
plt.title("High")
plt.show()

plt.hist(imgLowContrast.ravel(), 256, [0, 256])
plt.title("Low")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
