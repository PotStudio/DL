import tensorflow as tf
import numpy as np

import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_data = tf.placeholder(tf.float32, [None, 784])
y_data = tf.placeholder(tf.float32, [None, 10])

weight = tf.Variable(tf.ones([784, 10]))
bias = tf.Variable(tf.ones([10]))

y_model = tf.nn.relu(tf.matmul(x_data, weight) + bias)

loss = -tf.reduce_sum(y_data * tf.log(y_model))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
# tf.argmax 0：按列计算，1：行计算
currect_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(y_data, 1))

# tf.cast张量数据类型转换
accuracy = tf.reduce_mean(tf.cast(currect_prediction, "float"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000000):
        batch_x, batch_y = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x_data: batch_x, y_data: batch_y})
        if i % 50 == 0:
            print(sess.run(accuracy, feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels}))
