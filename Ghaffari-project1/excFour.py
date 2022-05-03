import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt

# ADD padding
img = Image.open("assets/1-1a.jpg")
imgPadding = padding = np.pad(img, ((400, 0), (0, 0), (0, 0)),mode='constant')

# FFT 2D Original
ftImg = cv2.imread("assets/3-a.jpg")
ftImgGray = cv2.cvtColor(ftImg, cv2.COLOR_BGR2GRAY)
ftImg = np.fft.fft2(ftImgGray)
ftImg = np.fft.fftshift(ftImg)
magnitudeSpectrum = 20 * np.log(np.abs(ftImg))

# FFT 2D Padding

ftImgGray = cv2.cvtColor(padding, cv2.COLOR_BGR2GRAY)
ftImg = np.fft.fft2(ftImgGray)
ftImg = np.fft.fftshift(ftImg)
magnitudeSpectrumPadding = 20 * np.log(np.abs(ftImg))

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.subplot(122)
plt.imshow(imgPadding, cmap='gray')
plt.title('additional Padding')
plt.show()


plt.subplot(121)
plt.imshow(magnitudeSpectrum, cmap='gray')
plt.title('Magnitude Spectrum Original')
plt.subplot(122)
plt.imshow(magnitudeSpectrumPadding, cmap='gray')
plt.title('Magnitude Spectrum Padding')
plt.show()
