import cv2;
import matplotlib.pyplot as plt
from skimage.util import random_noise

import numpy as np

# redding image using opencv2 library & constructor
BGRImage = cv2.imread("asset/bird.jpg")
RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB).astype(np.double)
grey = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2GRAY).astype(np.double)

a = 15
b = 15

# Degrading the image and adding noise:
# --------------------------------------------------------------------------------
# Task 1.1.1:  motion blurring
img = BGRImage / 255
n2, n1 = img.shape[:2]
[u, v] = np.mgrid[-round(n2 / 2):round(n2 / 2), -round(n1 / 2):round(n1 / 2)]
u = 2 * u / n2
v = 2 * v / n1

# Blurring function.
H = np.sinc((u * a + v * b)) * np.exp(-1j * np.pi * (u * a + v * b))
F = np.fft.fftshift(np.fft.fft2(img))

Fb = np.fft.fftshift(np.fft.fft2(img[:, :, 0]))
Fg = np.fft.fftshift(np.fft.fft2(img[:, :, 1]))
Fr = np.fft.fftshift(np.fft.fft2(img[:, :, 2]))

Gb = np.multiply(Fb, H)
Gg = np.multiply(Fg, H)
Gr = np.multiply(Fr, H)

gb = np.abs(np.fft.ifft2(Gb))
gg = np.abs(np.fft.ifft2(Gg))
gr = np.abs(np.fft.ifft2(Gr))
blurry = cv2.merge((gb, gg, gr))

# --------------------------------------------------------------------------------
# Task 1.1.2: Gaussian noise
blu_gau_img = random_noise(blurry, 'gaussian', mean=0, var=0.03)
# --------------------------------------------------------------------------------
# Task 1.1.3:

cv2.imshow("Motion Blurry Image", blurry)
cv2.imshow("Motion Blur and Gaussian noise", blu_gau_img)

# Removing noise:
# --------------------------------------------------------------------------------
# Task 1.2.1:  inverse filtering Blurry image
# ref: https://stackoverflow.com/questions/7894094/am-i-using-numpy-to-calculate-the-inverse-filter-correctly

reFb_1 = np.fft.fftshift(np.fft.fft2(blurry[:, :, 0]))
reFg_1 = np.fft.fftshift(np.fft.fft2(blurry[:, :, 1]))
reFr_1 = np.fft.fftshift(np.fft.fft2(blurry[:, :, 2]))


reGb_1 = np.divide(reFb_1, reFb_1/Fb)
reGg_1 = np.divide(reFg_1, reFg_1/Fg)
reGr_1 = np.divide(reFr_1, reFr_1/Fr)

deGb_1 = np.abs(np.fft.ifft2(reGb_1))
deGg_1 = np.abs(np.fft.ifft2(reGg_1))
deGr_1 = np.abs(np.fft.ifft2(reGr_1))
denoise_blurry_gauss = cv2.merge((deGb_1, deGg_1, deGr_1))
cv2.imshow("Blurry Inverse", denoise_blurry_gauss)
# --------------------------------------------------------------------------------
# Task 1.2.2:  inverse filtering Blurry image and Noisy
blu_gau_img = cv2.normalize(blu_gau_img, None)
reFb_2 = np.fft.fftshift(np.fft.fft2(blu_gau_img[:, :, 0])) / Fb
reFg_2 = np.fft.fftshift(np.fft.fft2(blu_gau_img[:, :, 1])) / Fg
reFr_2 = np.fft.fftshift(np.fft.fft2(blu_gau_img[:, :, 2])) / Fr

reGb_2 = np.divide(reFb_2, H)
reGg_2 = np.divide(reFg_2, H)
reGr_2 = np.divide(reFr_2, H)

deGb_2 = np.abs(np.fft.ifft2(reGb_2))
deGg_2 = np.abs(np.fft.ifft2(reGg_2))
deGr_2 = np.abs(np.fft.ifft2(reGr_2))
denoise_blurry_gauss = cv2.merge((deGb_2, deGg_2, deGr_2))
cv2.imshow("Blurry and Gaussian Inverse", denoise_blurry_gauss)

# --------------------------------------------------------------------------------
# Task 1.2.3:  MMSE Filter for additive noise
gau_img = random_noise(img, 'gaussian', mean=0, var=0.03)

oG_1 = np.fft.fftshift(np.fft.fft2(gau_img[:, :, 0]))
oG_2 = np.fft.fftshift(np.fft.fft2(gau_img[:, :, 1]))
oG_3 = np.fft.fftshift(np.fft.fft2(gau_img[:, :, 2]))

nn = img - gau_img

sG_b = abs(np.fft.fftshift(np.fft.fft2(nn[:, :, 0]))) ** 2
sG_g = abs(np.fft.fftshift(np.fft.fft2(nn[:, :, 1]))) ** 2
sG_r = abs(np.fft.fftshift(np.fft.fft2(nn[:, :, 2]))) ** 2

xG_b = abs(Fb) ** 2
xG_g = abs(Fg) ** 2
xG_r = abs(Fr) ** 2

dh1 = np.abs(1) ** 2 + (sG_b / xG_b)
dh2 = np.abs(1) ** 2 + (sG_g / xG_g)
dh3 = np.abs(1) ** 2 + (sG_r / xG_r)

Hw1 = np.conj(1) / dh1
Hw2 = np.conj(1) / dh2
Hw3 = np.conj(1) / dh3

R1 = Hw1 * Gb
R2 = Hw2 * Gg
R3 = Hw3 * Gr
a1 = np.abs(np.fft.ifft2(R1))
a2 = np.abs(np.fft.ifft2(R2))
a3 = np.abs(np.fft.ifft2(R3))
MMSEFilter_additive_noise = cv2.merge((a1, a2, a3))
cv2.imshow("MMSE filter for additive noise ", MMSEFilter_additive_noise)

# --------------------------------------------------------------------------------
# Task 1.2.4:  MMSE Filter for motion blur and additive noise

neImg = img - blu_gau_img

snnBb = abs(np.fft.fftshift(np.fft.fft2(neImg[:, :, 0]))) ** 2
snnBg = abs(np.fft.fftshift(np.fft.fft2(neImg[:, :, 1]))) ** 2
snnBr = abs(np.fft.fftshift(np.fft.fft2(neImg[:, :, 2]))) ** 2
# mean of each channel
k_1 = np.mean((reFb_2 / snnBb))
k_2 = np.mean((reFg_2 / snnBg))
k_3 = np.mean((reFr_2 / snnBr))

IB_b = oG_1 / Fb
IB_g = oG_2 / Fg
IB_r = oG_3 / Fr

dhB_b = np.abs(IB_b) ** 2 + (k_1)
dhB_g = np.abs(IB_g) ** 2 + (k_2)
dhB_r = np.abs(IB_r) ** 2 + (k_3)

HwB_b = np.conj(IB_b) / dhB_b
HwB_g = np.conj(IB_g) / dhB_g
HwB_r = np.conj(IB_r) / dhB_r

RB_b = HwB_b * Gb
RB_g = HwB_g * Gg
RB_r = HwB_r * Gr

aB_b = np.abs(np.fft.ifft2(RB_b))
aB_g = np.abs(np.fft.ifft2(RB_g))
aB_r = np.abs(np.fft.ifft2(RB_r))

MMSEFilter_blur_noise = cv2.merge((aB_b, aB_g, aB_r))
cv2.imshow("MMSE filter for additive noise and motion blur", MMSEFilter_blur_noise)

cv2.waitKey(0)
cv2.destroyAllWindows()
