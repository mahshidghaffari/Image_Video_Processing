import cv2;
from ProjectOne.UsefulFuc import convertor
import numpy as np

BGRImage = cv2.imread("assets/1-1a.jpg")

RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB)

# Color Space
# Part One
HSVImage = cv2.cvtColor(RGBImage, cv2.COLOR_RGB2HSV)

# part two
HSIImage = convertor.RGB_TO_HSI(BGRImage)

bgr = np.float32(BGRImage) / 255
blue = bgr[:, :, 0]
green = bgr[:, :, 1]
red = bgr[:, :, 2]
# calculate Intensity
intensity = np.divide(blue + red + green, 3)
# calculate Value
value = np.maximum(np.maximum(red, green), blue)

cv2.imshow('RGB', RGBImage)
cv2.imshow('HSV', HSVImage)
cv2.imshow('HSI', HSIImage)

cv2.waitKey(0)
cv2.destroyAllWindows()
