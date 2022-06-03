import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops
from skimage.measure import label


def open(image, kernel):
    eroded = cv2.erode(image, kernel)
    dilated = cv2.dilate(eroded, kernel)
    return dilated


def close(image, kernel):
    dilated = cv2.dilate(image, kernel)
    eroded = cv2.erode(dilated, kernel)
    return eroded


def countOrange(img, x, y, z):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (x, x))
    opening = open(img, k)
    labeling = label(opening)
    regions = regionprops(label(labeling))
    masks = []
    box = []
    indices = []
    for num, x in enumerate(regions):
        area = x.area
        convex_area = x.convex_area
        if (num != 0 and (area > 10) and (convex_area / area < y)
                and (convex_area / area > z)):
            masks.append(regions[num].convex_image)
            box.append(regions[num].bbox)
            indices.append(num)
    return len(masks)


# Task 3.1
# ############################################
# reading data
img_1 = cv2.imread("asset/oranges.jpg")
img_2 = cv2.imread("asset/orangetree.jpg")
img_3 = cv2.imread("asset/jar.jpg")

RGB_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
RGB_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
RGB_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2RGB)

grey_1 = cv2.cvtColor(RGB_1, cv2.COLOR_RGB2GRAY)
grey_2 = cv2.cvtColor(RGB_2, cv2.COLOR_RGB2GRAY)
grey_3 = cv2.cvtColor(RGB_3, cv2.COLOR_RGB2GRAY)

# getting black or white by threshold
(x_1, y_1) = cv2.threshold(grey_1, 127, 255, cv2.THRESH_BINARY)
(x_2, y_2) = cv2.threshold(grey_2, 127, 255, cv2.THRESH_BINARY)
(x_3, y_3) = cv2.threshold(grey_3, 127, 255, cv2.THRESH_BINARY)

orange_num1 = countOrange(y_1, 10, 1.09, 0.9)
orange_num2 = countOrange(y_2, 40, 1.5, 1.1)
print("the number of the oranges in orange image is: ", orange_num1)
print("the number of the oranges in orange tree image is: ", orange_num2)

# Task 3.2
# ############################################
size = np.linspace(1, 100, 100)
intensity = np.zeros(len(size))

# filling intensity matrix regarding surface area
for i in range(len(size)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i + 1, i + 1))
    intensity[i] = np.sum(open(y_3, kernel))
    print("loding", i, ":", len(size))

print("finishing intensity")
diffRate = np.gradient(intensity)

plt.plot(size, diffRate, '-bo')
plt.title('difference in surface area / radius of SE')
plt.show()

plt.plot(size, intensity)
plt.title('sum of intensities ')
plt.show()
