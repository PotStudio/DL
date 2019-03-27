import os
import tensorflow as tf
from PIL import Image

path ="jpg"
filenames = os.listdir(path)
writer = tf.python_io.TFRecordWriter("train.tfrecords")

for name in os.listdir(path):
    class_path = path + os.sep + name
    for img_name in os.listdir(path):
        img_path = class_path + os.sep+img_name
        print(img_path)
