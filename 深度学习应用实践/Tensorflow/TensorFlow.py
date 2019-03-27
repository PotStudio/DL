import tensorflow as tf
import numpy as np

inputX = np.random.rand(100)
# np.multiply()函数
# 数组和矩阵对应位置相乘，输出与相乘数组/矩阵的大小一致
inputY = np.multiply(3, inputX) + 1
x = tf.placeholder(tf.float32)
weight = tf.Variable(0.25)
bias = tf.Variable(0.25)

y = tf.multiply(weight, x) + bias
y_ = tf.placeholder(tf.float32)

# x = tf.constant([[1, 1, 1], [1, 1, 1]])
#   tf.reduce_sum(x)  # 6
loss = tf.reduce_sum(tf.pow((y - y_), 2))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for _ in range(1000):
    sess.run(train_step, feed_dict={x: inputX, y_: inputY})
    if _ % 20 == 0:
        print("W 的 值 为 ：", weight.eval(session=sess), "bias的值为：", bias.eval(session=sess))

