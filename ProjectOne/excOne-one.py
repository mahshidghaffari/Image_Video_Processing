import cv2;
from ProjectOne.UsefulFuc import convertor
import numpy as np

# redding image using opencv2 library
BGRImage = cv2.imread("assets/1-1b.jpg.jpg")

# Task One
RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB)
HSVImage = cv2.cvtColor(RGBImage, cv2.COLOR_RGB2HSV)

cv2.imshow('RGB', RGBImage)
cv2.imshow('HSV', BGRImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
