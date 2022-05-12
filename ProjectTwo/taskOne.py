import cv2;
from ProjectOne.UsefulFuc import convertor
import numpy as np


# lab 5- lecture 5

# redding image using opencv2 library
BGRImage = cv2.imread("asset/bird.jpg")
bgr = np.float32(BGRImage) / 255
blue = bgr[:, :, 0]
green = bgr[:, :, 1]
red = bgr[:, :, 2]

# Task One
n2, n1 = BGRImage.shape[:2]

[k1, k2] = np.mgrid[0:n2, 0:n1]
print([k1, k2])

[u, v] = np.mgrid[-n2 / 2:n2 / 2, -n1 / 2:n1 / 2]
u = 2 * u / n2
v = 2 * v / n1

F = np.fft.fft2(BGRImage)

a = 0.13
b = 0

# Blurring function.
H = np.sinc((u * a + v * b)) * np.exp(-1j * np.pi * (u * a + v * b))
# print(F.shape)
# G = F * H
# G = G * blue * green * red
