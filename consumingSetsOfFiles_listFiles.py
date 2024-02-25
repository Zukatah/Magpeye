import tensorflow as tf
import os
from globalConstants import TRAINING_EXAMPLE_DEPTH, TRAINING_EXAMPLE_HEIGHT, TRAINING_EXAMPLE_WIDTH, TRAINING_BATCH_SIZE

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
    return tf.reshape(image, [TRAINING_EXAMPLE_DEPTH,TRAINING_EXAMPLE_HEIGHT,TRAINING_EXAMPLE_WIDTH, 1]), tf.reshape(label, [4])

def reshape2(image, label, file_path):
    return tf.reshape(image, [TRAINING_EXAMPLE_DEPTH,TRAINING_EXAMPLE_HEIGHT,TRAINING_EXAMPLE_WIDTH, 1]), tf.reshape(label, [4]), file_path

lf_ds_train_raw = tf.data.Dataset.list_files('Pictures3D/train/*/*', shuffle=True)
lf_ds_val_raw = tf.data.Dataset.list_files('Pictures3D/val/*/*', shuffle=True)

print("len(lf_ds_train_raw)", len(lf_ds_train_raw), "len(lf_ds_val_raw)", len(lf_ds_val_raw))

lf_ds_train = lf_ds_train_raw.map(process_image).map(scale).map(reshape).batch(TRAINING_BATCH_SIZE)
lf_ds_val_filename = lf_ds_val_raw.map(process_image2).map(scale2).map(reshape2).batch(TRAINING_BATCH_SIZE)
lf_ds_val = lf_ds_val_raw.map(process_image).map(scale).map(reshape).batch(TRAINING_BATCH_SIZE)