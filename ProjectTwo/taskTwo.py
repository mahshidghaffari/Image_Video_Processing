import cv2;
from matplotlib import pyplot as plt
from scipy import fftpack

from ProjectOne.UsefulFuc import convertor
import numpy as np
from numpy import r_

# lab 6- lecture 6 DCT/ the famous zig zag
# water marking p.626
# redding image using opencv2 library
BGRimg = cv2.imread("asset/images/two.png")
img = cv2.cvtColor(BGRimg, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original Low contrast image', img)

plt.title("8x8 DCTs of the image")
blockSize = 1000
imgShape = img.shape
cv2.imshow(img[pos:8,pos:8])
#
# # --------------------------------------------------------------------------------
# # Task 2.1
# # ref https://stackoverflow.com/questions/15978468/using-the-scipy-dct-function-to-create-a-2d-dct-ii
#
# dct = np.zeros(imgShape)
# for i in r_[:imgShape[0]:blockSize]:
#     for j in r_[:imgShape[1]:blockSize]:
#         dct2 = fftpack.dct(fftpack.dct((img[i:i + blockSize, j:j + blockSize]).T, norm='ortho').T, norm='ortho')
#         dct[i:i + blockSize, j:j + blockSize] = dct2
#
# ax[1, 0].set_title("dct")
# ax[1, 0].imshow(dct)
cv2.waitKey(0)
cv2.destroyAllWindows()