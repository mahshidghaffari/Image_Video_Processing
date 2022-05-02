import cv2;
from matplotlib import pyplot as plt
from scipy import fftpack, ndimage
import numpy as np

image = cv2.imread("assets/3-a.jpg")

#  Task One adding periodic noise
imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
row, col = imgGray.shape
x = np.linspace(30, 40, row)
y = np.linspace(30, 40, col)
[X, Y] = np.meshgrid(y, x)
info = np.max(imgGray)
noisy = imgGray.astype(float) / info + np.multiply(np.sin(10 * np.pi * Y), 0.2)
# cv2.imshow("Sin noisy image", noisy)

# Task Two
# 2D fourier transform
fftOrigin = np.fft.fftshift(np.fft.fft2(imgGray))
magSpecOri = 20 * np.log(np.abs(fftOrigin))
deOrigin = abs(magSpecOri) / 255

fftNoisy = np.fft.fftshift(np.fft.fft2(noisy))
magSpecNoisy = 20 * np.log(np.abs(fftNoisy))
deNoisy = abs(magSpecNoisy) / 255

plt.subplot(121)
plt.imshow(deOrigin, cmap='gray')
plt.title('2D power spectrum Origin')
plt.subplot(122)
plt.imshow(deNoisy, cmap='gray')
plt.title('2D power spectrum Noisy')
plt.show()

# 1D fourier transform
# origin1D = deOrigin[int(row / 2), :]
rowN, colN = noisy.shape
# noisy1D = deNoisy[int(row / 2), :]
#
# plt.subplot(121)
# plt.imshow(noisy1D, cmap='gray')
# plt.title('1D power spectrum Origin')
# plt.subplot(122)
# plt.imshow(noisy1D, cmap='gray')
# plt.title('1D power spectrum Noisy')
# plt.show()

# 3D fourier transform
n3D, m3D = np.meshgrid(np.arange(row), np.arange(col))
fig = plt.figure(figsize=(14, 9))
ax = plt.axes(projection='3d')
ax.plot_surface(n3D, m3D, imgGray.T, cmap=plt.cm.coolwarm, linewidth=0)
plt.title("3D origin")
plt.show()


n3D, m3D = np.meshgrid(np.arange(rowN), np.arange(colN))
fig = plt.figure(figsize=(14, 9))
ax = plt.axes(projection='3d')
ax.plot_surface(n3D, m3D, noisy.T, cmap=plt.cm.coolwarm, linewidth=0)
plt.title("3D noisy")
plt.show()



# denoising image
im_fft = fftpack.fft2(noisy)
keep_fraction = 0.1
im_fft2 = im_fft.copy()
r, c = im_fft2.shape
im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
im_new = fftpack.ifft2(im_fft2).real

im_blur = ndimage.gaussian_filter(noisy, 4)

plt.figure()
plt.imshow(im_new, plt.cm.gray)
plt.title('Reconstructed Image')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
