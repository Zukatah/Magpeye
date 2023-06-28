import tensorflow as tf
import os

def get_label(file_path):
    label = int(tf.strings.split(file_path, os.path.sep)[-2])
    labelOneHot = tf.one_hot(label, 4)
    return labelOneHot

def process_image(file_path):
    label = get_label(file_path)
    img3d = tf.io.read_file(file_path)
    img3d = tf.io.decode_raw(img3d, tf.uint8)
    return img3d[128:], label

def process_image2(file_path):
    label = get_label(file_path)
    img3d = tf.io.read_file(file_path)
    img3d = tf.io.decode_raw(img3d, tf.uint8)
    return img3d[128:], label, file_path

def scale(image, label):
    return image/255, label

def scale2(image, label, file_path):
    return image/255, label, file_path

def reshape(image, label):
    return tf.reshape(image, [5,480,270, 1]), tf.reshape(label, [4])

def reshape2(image, label, file_path):
    return tf.reshape(image, [5,480,270, 1]), tf.reshape(label, [4]), file_path

lf_ds_train_raw = tf.data.Dataset.list_files('Pictures3D/train/*/*', shuffle=True)
lf_ds_val_raw = tf.data.Dataset.list_files('Pictures3D/val/*/*', shuffle=True)

'''
# Wir könnten auch so mit take und skip aus einem ds mehrere ds bilden (train und val bspw.)
lf_ds_len = len(lf_ds)
train_size = int(lf_ds_len * 0.95)
lf_ds_train = lf_ds.take(train_size)
lf_ds_val = lf_ds.skip(train_size)
'''

print("len(lf_ds_train_raw)", len(lf_ds_train_raw), "len(lf_ds_val_raw)", len(lf_ds_val_raw))

lf_ds_train = lf_ds_train_raw.map(process_image).map(scale).map(reshape).batch(8)
lf_ds_val_filename = lf_ds_val_raw.map(process_image2).map(scale2).map(reshape2).batch(8)
lf_ds_val = lf_ds_val_raw.map(process_image).map(scale).map(reshape).batch(8)