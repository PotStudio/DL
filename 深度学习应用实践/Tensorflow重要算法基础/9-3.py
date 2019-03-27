import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

xs = np.random.randint(low=46, high=99, size=100)
ys = 1.7 * xs

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(0.1)
b = tf.Variable(0.1)

y_ = tf.multiply(w, x) + b

cost = tf.reduce_sum(tf.pow((y - y_), 2))


train_step = tf.train.GradientDescentOptimizer(0.02).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(10):
        sess.run(train_step, feed_dict={x: xs, y: ys})


