import tensorflow as tf
import cv2
img_add_list = []
img_lable_list = []
with open("train_list.csv", "r") as fid:
    for image in fid.readlines():
        # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
        img_add_list.append(image.strip().split(",")[0])
        img_lable_list.append(image.strip().split(",")[1])
    img = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file("jpg\\image0.jpg"),channels=1),dtype=tf.float32)
    print(img)