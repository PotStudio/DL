import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

images = tf.placeholder(tf.float32, [None, 784])
labels = tf.placeholder(tf.float32, [None, 10])

images_reshape = tf.reshape(images, shape=[-1, 28, 28, 1])


def weight_varible(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))



def bias_varible(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(data, weight):
    return tf.nn.conv2d(data, weight, strides=[1, 1, 1, 1], padding="SAME")


def max_pool(data):
    return tf.nn.max_pool(data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 卷积层1
w0 = weight_varible([5, 5, 1, 32])
b0 = bias_varible([32])
conv = conv2d(images_reshape, w0) + b0
relu = tf.nn.relu(conv)
pool_result = max_pool(relu)
print(pool_result)

# 卷积层2
w1 = weight_varible([5, 5, 32, 64])
b1 = bias_varible([64])
conv1 = conv2d(pool_result, w1) + b1
relu1 = tf.nn.relu(conv1)
pool1 = max_pool(relu1)
print(pool1)
# 全连接1
w_fc1 = weight_varible([7 * 7 * 64, 1024])
b_fc1 = bias_varible([1024])

pool1_reshape = tf.reshape(pool1, [-1, 7 * 7 * 64])
fc1_result = tf.nn.relu(tf.matmul(pool1_reshape, w_fc1) + b_fc1)
# 全连接2
w_fc2 = weight_varible([1024, 10])
b_fc2 = bias_varible([10])
fc2_result = tf.nn.softmax(tf.matmul(fc1_result, w_fc2) + b_fc2)

loss = -tf.reduce_sum(labels * tf.log(fc2_result))
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
currect_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(fc2_result, 1))
accuracy = tf.reduce_mean(tf.cast(currect_prediction, "float"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        batch_images, batch_labels = mnist.train.next_batch(50)
        print(type(batch_images))
        sess.run(train_step, feed_dict={images: batch_images, labels: batch_labels})
        # test_images, test_labels = mnist.test.next_batch(10)
        print(sess.run(accuracy, feed_dict={images: batch_images, labels: batch_labels}))
    test_images, test_lables = mnist.test.next_batch(10)
    print(test_images.shape)
    test_acc = sess.run(accuracy, feed_dict={images: test_images, labels: test_lables})
    print("测试准确率为", test_acc)