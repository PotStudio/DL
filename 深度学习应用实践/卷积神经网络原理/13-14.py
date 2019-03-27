import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import time
import matplotlib.pyplot as plt

mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)


def weight_Variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_Variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def conv2d(data, filter):
    return tf.nn.conv2d(data, filter, strides=[1, 1, 1, 1], padding="SAME")


def max_poll2x2(data):
    return tf.nn.max_pool(data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_ = tf.reshape(x, [-1, 28, 28, 1])

conv1_w = weight_Variable([5, 5, 1, 32])
conv1_b = bias_Variable([32])
layer1 = tf.nn.relu(conv2d(x_, conv1_w) + conv1_b)

max_pool1 = max_poll2x2(layer1)
# (28-2)/2+1=14

conv2_w = weight_Variable([5, 5, 32, 64])
conv2_b = bias_Variable([64])
layer2 = tf.nn.relu(conv2d(max_pool1, conv2_w) + conv2_b)

max_pool2 = max_poll2x2(layer2)
# (14-2)/2+1=7
fc1_w = weight_Variable([7 * 7 * 64, 1024])
fc1_b = bias_Variable([1024])
max_pool2_reshape = tf.reshape(max_pool2, [-1, 7 * 7 * 64])
layer3 = tf.nn.relu(tf.matmul(max_pool2_reshape, fc1_w) + fc1_b)

fc2_w = weight_Variable(([1024, 128]))
fc2_b = bias_Variable([128])
layer4 = tf.nn.relu(tf.matmul(layer3, fc2_w) + fc2_b)

fc3_w = weight_Variable(([128, 10]))
fc3_b = bias_Variable([10])
result = tf.nn.softmax(tf.matmul(layer4, fc3_w) + fc3_b)

loss = -tf.reduce_sum(y_ * tf.log(result))

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

currect_prediction = tf.equal(tf.argmax(result, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(currect_prediction, tf.float32))
c = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    for i in range(1000):
        batch_xs, batch_ys = mnist_data.train.next_batch(200)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 2 == 0:
            end_time = time.time()
            train_accuracy = sess.run(tf.cast(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}), tf.float32))
            c.append(train_accuracy)
            print("step %d, accuracy is %f" % (i, train_accuracy))
            print("time is", end_time - start_time)
            start_time = end_time
    print(sess.run(accuracy, feed_dict={x: mnist_data.test.images, y_: mnist_data.test.labels}))
sess.close()
plt.plot(c)
# tight_layout会自动调整子图参数，使之填充整个图像区域。
plt.tight_layout()
plt.savefig("cnn-tf-cifar10-11.png", dpi=200)
plt.show()