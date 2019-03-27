import tensorflow as tf

data = tf.constant([
    [[3.0, 2.0, 3.0, 4.0],
     [2.0, 6.0, 2.0, 4.0],
     [1.0, 2.0, 1.0, 5.0],
     [4.0, 3.0, 2.0, 1.0]]
])

data = tf.reshape(data, shape=[1, 4, 4, 1])
maxPoling = tf.nn.max_pool(data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(maxPoling))
