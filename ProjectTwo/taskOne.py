import cv2;
import imageio
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

import numpy as np

# redding image using opencv2 library & constructor
BGRImage = cv2.imread("asset/bird.jpg")
RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB).astype(np.double)
grey = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2GRAY).astype(np.double)
fig, ax = plt.subplots(3, 2, sharex='col', sharey='row')
fig.suptitle('TasK One')
a = 0.1
b = 0.09

# Degrading the image and adding noise:
# --------------------------------------------------------------------------------
# Task One:  motion blurring
n2, n1 = BGRImage.shape[:2]
[u, v] = np.mgrid[-n2 / 2:n2 / 2, -n1 / 2:n1 / 2]
u = 2 * u / n2
v = 2 * v / n1
F = np.fft.fft2(RGBImage)
blueF = F[:, :, 0]
greenF = F[:, :, 1]
redF = F[:, :, 2]
# Blurring function.
H = np.sinc((u * a + v * b)) * np.exp(-1j * np.pi * (u * a + v * b))
G = F
G[:, :, 0] = np.multiply(blueF, H)
G[:, :, 1] = np.multiply(greenF, H)
G[:, :, 2] = np.multiply(redF, H)
Gp = np.fft.ifft2(G)
blurry = abs(Gp) / 255

# --------------------------------------------------------------------------------
# Task Two: Gaussian noise
mean = 0
var = 0.08
std = var ** 0.5
gaus_noise = np.random.normal(mean, std, blurry.shape)
gaus_noise = gaus_noise.reshape(blurry.shape)
noise_img = blurry + gaus_noise
noise_img = noise_img * 255

# --------------------------------------------------------------------------------
# Task Three:
ax[0, 0].set_title("Motion Blurring")
ax[0, 0].imshow(blurry)
ax[0, 1].set_title("Gaussian Noise")
ax[0, 1].imshow(noise_img.astype(np.uint8))

# Removing noise:
# --------------------------------------------------------------------------------
# Task One:  inverse filtering Blurry image
# ref: https://stackoverflow.com/questions/7894094/am-i-using-numpy-to-calculate-the-inverse-filter-correctly

G_inverse_blurry = np.fft.fft2(blurry)
G_inverse_blurry[:, :, 0] = np.divide(blueF, H)
G_inverse_blurry[:, :, 1] = np.divide(greenF, H)
G_inverse_blurry[:, :, 2] = np.divide(redF, H)
Gp_inverse_blurry = np.fft.ifft2(G_inverse_blurry)
blurry_inverse = abs(Gp_inverse_blurry) / 255
ax[1, 0].set_title("Inverse filtering Blurry")
ax[1, 0].imshow(blurry_inverse)

# --------------------------------------------------------------------------------
# Task two:  inverse filtering Blurry image and Noisy

G_inverse_blurry_noisy = np.fft.fft2(noise_img)
G_inverse_blurry_noisy[:, :, 0] = np.divide(noise_img[:, :, 0], H) - np.divide(blueF, H)
G_inverse_blurry_noisy[:, :, 1] = np.divide(noise_img[:, :, 1], H) - np.divide(greenF, H)
G_inverse_blurry_noisy[:, :, 2] = np.divide(noise_img[:, :, 2], H) - np.divide(redF, H)
noisy_blurry_inverse = abs(np.fft.ifft2(G_inverse_blurry_noisy)) / 255
noisy_blurry_inverse = (blurry_inverse * 255).astype(np.uint8)
ax[1, 1].set_title("Inverse filtering Blurry and Noisy")
ax[1, 1].imshow(noisy_blurry_inverse)

# --------------------------------------------------------------------------------
# Task two:  MMSE Filter

Hp2 = H * np.conj(H)
test = Gp * np.conj(Gp)
Sn = gaus_noise * np.conj(gaus_noise)
Sf = F + np.conj(F)

one = np.multiply(H, Hp2)
G3 = F
G3[:, :, 0] = np.multiply(blueF, one)
G3[:, :, 1] = np.multiply(greenF, one)
G3[:, :, 2] = np.multiply(redF, one)
one = np.fft.ifft2(G3)
two = np.divide(Sn, Sf)
print(one.shape)
print(two.shape)
print(test.shape)

Hw = np.divide(Gp * np.conj(Gp), one + 0)
Hw = np.abs(np.fft.ifft2(Hw)) / 255

ax[2, 0].set_title("MMSE Filter")
ax[2, 0].imshow(Hw)

plt.show()
