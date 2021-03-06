import tensorflow as tf


# tf.random_normal()服从正太分布的数值
input = tf.Variable(tf.random_normal([1, 3, 3, 1]))
filter = tf.Variable(tf.ones([1, 1, 1, 1]))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    conv = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding="SAME")

    print(sess.run(conv))
