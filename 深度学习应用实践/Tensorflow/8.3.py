import tensorflow as tf
import numpy as np

# 生成一个3000行1列的随机矩阵
inputX = np.random.rand(3000, 1)

# 三个参数 loc mean, scale, size
noise = np.random.normal(0, 0.05, inputX.shape)
# 随机生成一个y=4*x+1的线性曲线，对随机生成的曲线增加一个偏差为0.05满足正太分布的噪声
outputY = inputX * 4 + 1 + noise

weight1 = tf.Variable(np.random.rand(inputX.shape[1], 4))
bias1 = tf.Variable(np.random.rand(inputX.shape[1], 4))
x1 = tf.placeholder(tf.float64, [None, 1])
y1_ = tf.matmul(x1, weight1) + bias1

y = tf.placeholder(tf.float64, [None, 1])
# tf.square()是对a里的每一个元素求平方
loss = tf.reduce_mean(tf.reduce_sum(tf.square((y1_ - y)), reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.25).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(1000):
        sess.run(train, feed_dict={x1: inputX, y: outputY})
    print(weight1.eval(sess))
    print("------------")
    print(bias1.eval(sess))
    print("结果是")
    x_data = np.matrix([[1.], [2.], [3.]])
    print(sess.run(y1_, feed_dict={x1: x_data}))

# # 生成一个2行3列的矩阵，第一行为[1, 2, 3]第二行自动补[3, 3, 3]
# tf.constant([1, 2, 3], shape=[2, 3])
# # 生成正态分布随机数
# tf.random_normal(shape, mean=)
