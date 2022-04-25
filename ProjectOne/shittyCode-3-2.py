import cv2
import numpy as np
import matplotlib.pyplot as plt


# 2. Loading and plotting the original image
from ProjectOne import unifyColor

img = cv2.imread("assets/3-b.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis("off")
plt.title("Original Image")
plt.show()


# 3. Converting Image to GrayScale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
gray2 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
plt.figure(figsize=(10,10))
plt.imshow(gray2,cmap="gray")
plt.axis("off")
plt.title("Grayscale Image")
plt.show()



# # 4. Getting edged image
#
# edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
# plt.figure(figsize=(10,10))
# plt.imshow(edges,cmap="gray")
# plt.axis("off")
# plt.title("Edged Image")
# plt.show()
#
# # 5. Turning Images into Cartoons
#
# color = cv2.bilateralFilter(img, 9, 250, 250)
# cartoon = cv2.bitwise_and(color, color, mask=edges)
# plt.figure(figsize=(10,10))
# plt.imshow(cartoon,cmap="gray")
# plt.axis("off")
# plt.title("Cartoon Image")
# plt.show()
