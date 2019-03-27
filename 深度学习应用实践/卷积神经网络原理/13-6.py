import tensorflow as tf
import cv2
import numpy as np

img = cv2.imread("images/little_bird.jpg", 1)
img = np.array(img, dtype=np.float32)
x_image = tf.reshape(img, [1, 411, 658, 3])
filter = tf.Variable(tf.ones([7, 7, 3, 1]))

conv = tf.nn.conv2d(x_image, filter, strides=[1, 2, 2, 1], padding="SAME")


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(conv)
    print(res)
    res_image = sess.run((tf.reshape(res, [206, 329]))/128+1)
    print(res_image.shape)
cv2.imshow("little_bird", res_image.astype("uint8"))
cv2.waitKey(0)