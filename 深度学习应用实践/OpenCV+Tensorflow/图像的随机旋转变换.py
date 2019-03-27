import numpy as np
import cv2

img = cv2.imread("lena.jpg")
rows, cols, depth = img.shape
# 参数三：scale 缩放的倍数
# 获得仿射变换矩阵
img_changed = cv2.getRotationMatrix2D((rows / 2, cols / 2), 45, 1)


res = cv2.warpAffine(img, img_changed, (rows, cols))
cv2.imshow("res", res)
cv2.waitKey(0)