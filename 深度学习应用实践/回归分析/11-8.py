import tensorflow as tf

matrix1 = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
matrix2 = tf.constant([1, 1, 1, 1, 1, 1], shape=[2, 3])

# multiply和*相同
result1 = tf.multiply(matrix1, matrix2)

sess = tf.Session()
print(sess.run(result1))
