import tensorflow as tf
import numpy as np

writer = tf.python_io.TFRecordWriter("tranArray.tfrecords")
for _ in range(100):
    randomArray = np.random.random((1, 3))
    array_raw = randomArray.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        "lable": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
        "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[array_raw]))
    }))
    writer.write(example.SerializeToString())
writer.close()

