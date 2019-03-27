import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import time

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])

x_images = tf.reshape(x, [-1, 28, 28, 1])
# tf.truncated_normal 产生正太分布数据，
# 与tf.random_normal不同的是，截取的是两个连准差之内的部分，
# 即横轴区间（μ-2σ，μ+2σ）内的面积为95.449974%。以内的，更接近均值，超出的重新选择，
# tf.random_normal一般认为是在横轴区间（μ-3σ，μ+3σ）内的面积为99.730020%。 以内的
# 卷积层1
filter1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6]))
bias1 = tf.Variable(tf.truncated_normal([6]))
conv1 = tf.nn.conv2d(x_images, filter1, strides=[1, 1, 1, 1], padding="SAME")
layer1 = tf.nn.sigmoid(conv1 + bias1)
print(layer1.shape)
# 下采样（池化层）
maxPool2 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
# (28-2)/2+1 = 14
print(maxPool2.shape)
# 卷积层2
filter2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16]))
bias2 = tf.Variable(tf.truncated_normal([16]))
conv2 = tf.nn.conv2d(maxPool2, filter2, strides=[1, 1, 1, 1], padding="SAME")
layer2 = tf.nn.sigmoid(conv2 + bias2)
# 下采样层2
maxPool3 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
# （14-2）/2+1 = 7
print(maxPool3.shape)
# 最后一个卷积层，卷积后为1×1,120个通道
filter3 = tf.Variable(tf.truncated_normal([5, 5, 16, 120]))
bias3 = tf.Variable(tf.truncated_normal([120]))
conv3 = tf.nn.conv2d(maxPool3, filter3, strides=[1, 1, 1, 1], padding="SAME")
layer3 = tf.nn.sigmoid(conv3 + bias3)
print(layer3)
# 全连接层

fc1_w = tf.Variable(tf.truncated_normal([7 * 7 * 120, 80]))
fc1_b = tf.Variable(tf.truncated_normal([80]))
layer3_reshape = tf.reshape(layer3, [-1, 7 * 7 * 120])
fc1_layer = tf.nn.sigmoid(tf.matmul(layer3_reshape, fc1_w) + fc1_b)

fc2_w = tf.Variable(tf.truncated_normal([80, 10]))
fc2_b = tf.Variable(tf.truncated_normal([10]))
fc2_layer = tf.nn.softmax(tf.matmul(fc1_layer, fc2_w) + fc2_b)

loss = -tf.reduce_sum(y_ * tf.log(fc2_layer))

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(fc2_layer, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

start_time = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        data_xs, data_ys = mnist.train.next_batch(100)
        # print(data_ys.shape)
        sess.run(train_step, feed_dict={x: data_xs, y_: data_ys})
        if i % 2 == 0:
            print(sess.run(accuracy, feed_dict={x: data_xs, y_: data_ys}))
            end_time = time.time()
            print("time is ", (end_time - start_time))
            start_time = end_time
    print("accuracy is ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
sess.close()
