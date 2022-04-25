import cv2;
from matplotlib import pyplot as plt

from ProjectOne.UsefulFuc import convertor
import numpy as np


def im2double(im):
    # Get the data type of the input image
    return im.astype(np.float) / info.max  # Divide all values by the largest possible value in the datatype


#  periodic noise
image = cv2.imread("assets/3-a.jpg")
imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
row, col = imgGray.shape
Y = np.linspace(0, 1, col)
info = np.iinfo(imgGray.dtype)
noisy = imgGray.astype(np.float) / info.max + np.multiply(np.sin(10 * np.pi * Y), 0.2)

# plot
plt.subplot(121)
plt.imshow(noisy, cmap='gray')
plt.title('Magnitude Spectrum Original')
plt.subplot(122)
plt.imshow(noisy, cmap='gray')
plt.title('noisy Compare')
plt.show()
# cv2.imshow("x", noisy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
