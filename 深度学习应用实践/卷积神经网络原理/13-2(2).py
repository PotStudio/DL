import tensorflow as tf
import cv2

image = cv2.imread("images//little_bird.jpg", 1)
print(image.shape)
image_reshape = tf.reshape(image, shape=[1, image.shape[0], image.shape[1], image.shape[2]])
input = tf.Variable(tf.cast(image_reshape, tf.float32))
filter = tf.Variable(tf.ones([3, 3, 3, 1]))
conv = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding="SAME")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pre = sess.run(conv)
    cv2.imshow("pre", pre[0])
    cv2.waitKey(0)
