import cv2;
from ProjectOne.UsefulFuc import convertor
import numpy as np

# reading Image
BGRImage = cv2.imread("assets/1-1b.jpg")
RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB)

# Task two
# using Convertor class to convert RGB to HSI
HSIImage = convertor.RGB_TO_HSI(BGRImage)

bgr = np.float32(BGRImage) / 255
blue = bgr[:, :, 0]
green = bgr[:, :, 1]
red = bgr[:, :, 2]
intensity = np.divide(blue + red + green, 3)
value = np.maximum(np.maximum(red, green), blue)

cv2.imshow('RGB', RGBImage)
cv2.imshow('HSI', HSIImage)
cv2.imshow('intensity', intensity)
cv2.imshow('value', value)

cv2.waitKey(0)
cv2.destroyAllWindows()
