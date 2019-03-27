import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

threshold = 1.0e-2
# numpy.random.randn(d0, d1, …, dn)是从标准正态分布中返回一个或多个样本值。
# numpy.random.rand(d0, d1, …, dn)的随机样本位于[0, 1)中。
x1_data = np.random.randn(100).astype(np.float32)
x2_data = np.random.randn(100).astype(np.float32)
y_data = x1_data * 3 + x2_data * 4 + 8

weight1 = tf.Variable(1.)
weight2 = tf.Variable(1.)
bias = tf.Variable(1.)

x1_ = tf.placeholder(tf.float32)
x2_ = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)

y_model = x1_ * weight1 + x2_ * weight2 + bias

loss = tf.reduce_mean(tf.pow((y_model - y_), 2))

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
flag = 1
while flag:
    for (x1, x2, y) in zip(x1_data, x2_data, y_data):
        sess.run(train_op, feed_dict={x1_: x1, x2_: x2, y_: y})
    if sess.run(loss, feed_dict={x1_: x1, x2_: x2, y_: y}) < threshold:
        flag = 0
print(weight1.eval(sess), weight2.eval(sess), bias.eval(sess))
