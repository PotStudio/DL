import cv2
import numpy as np
img = np.array(np.zeros((300, 300)), dtype=np.uint8)

cv2.imshow("img", img)
cv2.waitKey(0)
