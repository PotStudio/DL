import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

threshold = 1.0e-2
x1_data = np.random.randn(100).astype(np.float32)
x2_data = np.random.randn(100).astype(np.float32)

y_data = x1_data * 3 + x2_data * 2 + 0.35

weight1 = tf.Variable(1.)
weight2 = tf.Variable(1.)
bias = tf.Variable(1.)

x1_ = tf.placeholder(tf.float32)
x2_ = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)
y_model = x1_ * weight1 + x2_ * weight2 + bias

loss = tf.reduce_mean(tf.pow((y_model - y_), 2))
train__op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

flag = 1
while flag:
    for (x1, x2, y) in zip(x1_data, x2_data, y_data):
        sess.run(train__op, feed_dict={x1_: x1, x2_: x2, y_: y})
    current_loss = sess.run(loss, feed_dict={x1_: x1_data, x2_: x2_data, y_: y_data})
    print(current_loss)
    if current_loss < threshold:
        flag = 0

fig = plt.figure()
ax = Axes3D(fig)
# meshgrid画网格
x, y = np.meshgrid(x1_data, x2_data)
# np.multiply()和*都是矩阵袁术相乘
z = sess.run(weight1) * (x) + sess.run(weight2) * (y) + sess.run(bias)
ax.contourf(x, y, z, zdir="z", offset=-1, cmap=plt.cm.hot)
ax.set_zlim(-1, 1)
plt.show()
