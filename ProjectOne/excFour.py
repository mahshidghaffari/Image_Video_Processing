import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
# converte paddinget kar nemikone
# ADD padding using pillow library
# ref: https://www.geeksforgeeks.org/add-padding-to-the-image-with-python-pillow/
img = Image.open("assets/1-1a.jpg")
top, right, left, bottom = 100, 100, 100, 100

width, height = img.size

newWidth = width + right + left
newHeight = height + top + bottom

imgPadding = Image.new(img.mode, (newWidth, newHeight), (255, 255, 255))
imgPadding.paste(img, (left, top))

# FFT 2D Original
ftImg = cv2.imread("assets/3-a.jpg")
ftImgGray = cv2.cvtColor(ftImg, cv2.COLOR_BGR2GRAY)
ftImg = np.fft.fft2(ftImgGray)
ftImg = np.fft.fftshift(ftImg)
magnitudeSpectrum = 20 * np.log(np.abs(ftImg))
# magnitudeSpectrum.save('result/output.jpg')

# FFT 2D Padding
ftImgGray = cv2.cvtColor(imgPadding, cv2.COLOR_BGR2GRAY)
ftImg = np.fft.fft2(ftImgGray)
ftImg = np.fft.fftshift(ftImg)
magnitudeSpectrumPadding = 20 * np.log(np.abs(ftImg))
# magnitudeSpectrum.save('result/output.jpg')



plt.subplot(121)
plt.imshow(magnitudeSpectrum, cmap='gray')
plt.title('Magnitude Spectrum Original')
plt.subplot(122)
plt.imshow(magnitudeSpectrumPadding, cmap='gray')
plt.title('Magnitude Spectrum Padding')
plt.show()
