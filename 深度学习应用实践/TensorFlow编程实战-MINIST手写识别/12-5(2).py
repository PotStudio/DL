import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x_data = tf.placeholder(tf.float32, [None, 784])
y_data = tf.placeholder(tf.float32, [None, 10])

x_data_reshape = tf.reshape(x_data, [-1, 28, 28, 1])

#  conv_1
weight = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[32]))
relu_layout = tf.nn.relu(tf.nn.conv2d(x_data_reshape, weight, strides=[1, 1, 1, 1], padding="SAME") + bias)
pool_layout = tf.nn.max_pool(relu_layout, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
# conv_2
weight2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu_layout2 = tf.nn.relu(tf.nn.conv2d(pool_layout, weight2, strides=[1, 1, 1, 1], padding="SAME") + bias2)
pool_layout2 = tf.nn.max_pool(relu_layout2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
# full_connect1
weight3 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
bias3 = tf.Variable(tf.constant(0.1, shape=[1024]))
pool_layout2_reshape = tf.reshape(pool_layout2, shape=[-1, 7 * 7 * 64])
relu_layout3 = tf.nn.relu(tf.matmul(pool_layout2_reshape, weight3) + bias3)

# full_connect2
weight4 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
bias4 = tf.Variable(tf.constant(0.1, shape=[10]))
pred = tf.nn.softmax(tf.matmul(relu_layout3, weight4) + bias4)

loss = -tf.reduce_sum(y_data * tf.log(pred))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
currect_prediction = tf.equal(tf.argmax(y_data, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(currect_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        batch_x, batch_y = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x_data: batch_x, y_data: batch_y})
        print(sess.run(accuracy, feed_dict={x_data: batch_x, y_data: batch_y}))
    print(sess.run(accuracy, feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels}))
