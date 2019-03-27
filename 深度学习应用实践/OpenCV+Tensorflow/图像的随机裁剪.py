import cv2
import random
import numpy as np

img = cv2.imread("lena.jpg")
width, height, depth = img.shape
img_width_box = int(width * 0.2)
img_height_box = int(width * 0.2)
print(img_height_box, img_width_box, depth)

for i in range(9):
    start_pointX = int(random.uniform(0, img_width_box))
    start_pointy = int(random.uniform(0, img_height_box))
    copy_img = img[start_pointX:200, start_pointy:200]
    cv2.imshow("test", copy_img)
    cv2.waitKey(0)
