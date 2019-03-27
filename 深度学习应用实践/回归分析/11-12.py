import tensorflow as tf
import numpy as np

houses = 100
features = 2
# 设计模型为y=2*x1+3*x2
x_data = np.zeros([houses, 2])
for house in range(houses):
    # 函数原型：  numpy.random.uniform(low,high,size)
    #
    # 功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high
    # np.round 将数组舍入到给定的小数位数
    x_data[house, 0] = np.round(np.random.uniform(50., 250))
    x_data[house, 1] = np.round(np.random.uniform(3., 7.))

weights = np.array([[2.], [3.]])

y_data = np.dot(x_data, weights)

x_data_ = tf.placeholder(tf.float32, [1, 2])
y_data_ = tf.placeholder(tf.float32, [1, 1])

weights_ = tf.Variable(np.ones([2, 1]), dtype=tf.float32)
y_model = tf.matmul(x_data_, weights_)
loss = tf.reduce_mean(tf.pow((y_model - y_data_), 2))

train_op = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(500):
    for x, y in zip(x_data, y_data):
        z1 = x.reshape(1, 2)
        z2 = y.reshape(1, 1)
        sess.run(train_op, feed_dict={x_data_: z1, y_data_: z2})
print(weights_.eval(sess))
