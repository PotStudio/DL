# 卷积神经网络的程序编写
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x_data = tf.placeholder(tf.float32, [None, 784])
y_data = tf.placeholder(tf.float32, [None, 10])
# 图片数量, 图片高度, 图片宽度, 图像通道数
x_images = tf.reshape(x_data, (-1, 28, 28, 1))
# 卷积核的高度，卷积核的宽度，图像通道数，卷积核个数
# 卷积层
w_conv = tf.Variable(tf.ones([5, 5, 1, 32]))
b_conv = tf.Variable(tf.ones([32]))
relu_layout = tf.nn.relu(tf.nn.conv2d(x_images, w_conv, strides=[1, 1, 1, 1], padding="SAME") + b_conv)

# 1、[batch, height, width, channels] 2、[1, height, width, 1]
# 3、[1, stride,stride, 1]
h_pool = tf.nn.max_pool(relu_layout, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
# 全连接层
w_fc = tf.Variable(tf.ones([14 * 14 * 32, 1024]))
b_fc = tf.Variable(tf.ones([1024]))

h_pool_flat = tf.reshape(h_pool, [-1, 14 * 14 * 32])
# relu
h_fc = tf.nn.relu(tf.matmul(h_pool_flat, w_fc) + b_fc)
# 全连接
w_fc2 = tf.Variable(tf.ones([1024, 10]))
b_fc2 = tf.Variable(tf.ones([10]))
# softmax
y_model = tf.nn.softmax(tf.matmul(h_fc, w_fc2) + b_fc2)

loss = -tf.reduce_sum(y_data * tf.log(y_model))

train_step = tf.train.AdadeltaOptimizer(0.0001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_data, 1), tf.argmax(y_model, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        batch_x, batch_y = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x_data: batch_x, y_data: batch_y})
        print(sess.run(accuracy, feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels}))
