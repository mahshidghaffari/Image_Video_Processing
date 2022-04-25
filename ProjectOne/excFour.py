import cv2;
from ProjectOne.UsefulFuc import convertor
import numpy as np

BGRImage = cv2.imread("assets/1-1a.jpg")

RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB)

