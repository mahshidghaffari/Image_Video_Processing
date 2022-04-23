import cv2
import numpy as np

BGRImage = cv2.imread("assets/3-b.jpg")

# edges = cv2.Canny(BGRImage,100,200)
#retrieving the edges for cartoon effect
#by using thresholding technique

grayScaleImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2GRAY)
smoothGrayScale = cv2.medianBlur(grayScaleImage, 9)
getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255,
  cv2.ADAPTIVE_THRESH_MEAN_C,
  cv2.THRESH_BINARY, 19, 19)
colorImage = cv2.bilateralFilter(BGRImage, 9, 300, 300)

cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)
# s , t  = sobel_filters(BGRImage)
cv2.imshow("original image", BGRImage)
cv2.imshow("x" , cartoonImage)
cv2.waitKey(0)
cv2.destroyAllWindows()