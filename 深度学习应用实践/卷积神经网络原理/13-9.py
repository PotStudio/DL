import tensorflow as tf
import cv2
import numpy as np

img = cv2.imread("images/little_bird.jpg")
print(img.shape)

img = np.array(img, dtype=np.float32)

x_images = tf.reshape(img, [1, 411, 658, 3])

filter = tf.Variable(tf.ones([7, 7, 3, 1]))
conv = tf.nn.conv2d(x_images, filter, strides=[1, 2, 2, 1], padding="SAME")
max_pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
print(max_pool.shape)
res_reshape = tf.reshape(max_pool, [103, 165])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(res_reshape)/128+1

cv2.imshow("res", res.astype("uint8"))
cv2.waitKey(0)
