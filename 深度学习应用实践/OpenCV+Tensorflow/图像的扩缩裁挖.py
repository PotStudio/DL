import tensorflow as tf
import cv2
import numpy as np

dst = cv2.imread("lena.jpg")

cv2.imshow("dst", dst)
cv2.waitKey(0)
