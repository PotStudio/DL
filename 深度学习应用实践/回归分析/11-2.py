import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

x_data = np.random.randn(100).astype(np.float32)
y_data = x_data * 0.3 + 0.1

weight = tf.Variable(0.5)
bias = tf.Variable(0.0)
x_ = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)
y_model = weight * x_ + bias

loss = tf.pow((y_model - y_), 2)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for _ in range(100):
    for (x, y) in zip(x_data, y_data):
        sess.run(train_op, feed_dict={x_: x, y_: y})
    print("weight：", weight.eval(sess), "|bias:", bias.eval(sess))

plt.plot(x_data, y_data, "ro", label="Original data")
plt.plot(x_data, sess.run(weight) * x_data + sess.run(bias), label="Fitted")
plt.legend()
plt.show()
