
import cv2
import numpy as np

sourceA = cv2.imread('assets/3-a.jpg', 1)
imgA = sourceA.astype(np.float32)

sourceB = cv2.imread('assets/3-b.jpg', 1)
imgB = sourceB.astype(np.float32)

#--- the following holds the square root of the sum of squares of the image dimensions ---
#--- this is done so that the entire width/height of the original image is used to express the complete circular range of the resulting polar image ---
valueA = np.sqrt(((imgA.shape[0] / 2.0) ** 2.0) + ((imgA.shape[1] / 2.0) ** 2.0))
polar_image_A = cv2.linearPolar(imgA, (imgA.shape[0] / 2, imgA.shape[1] / 2), valueA, cv2.WARP_FILL_OUTLIERS)
polar_image_A = polar_image_A.astype(np.uint8)


valueB = np.sqrt(((imgB.shape[0] / 2.0) ** 2.0) + ((imgB.shape[1] / 2.0) ** 2.0))
polar_image_B = cv2.linearPolar(imgB, (imgB.shape[0] / 2, imgB.shape[1] / 2), valueB, cv2.WARP_FILL_OUTLIERS)
polar_image_B = polar_image_B.astype(np.uint8)

cv2.imshow(" Image A", sourceA)
cv2.imshow("Image B", sourceB)
cv2.imshow("Polar Image A", polar_image_A)
cv2.imshow("Polar Image B", polar_image_B)

cv2.waitKey(0)
cv2.destroyAllWindows()