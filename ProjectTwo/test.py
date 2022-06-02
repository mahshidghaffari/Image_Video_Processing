import numpy as np

import cv2

img = cv2.imread('asset/faces/original1.jpg')
# cv2.imshow("Image1", img)

dimensions = img.shape
print(dimensions)

cv2.waitKey(0)
cv2.destroyAllWindows()
