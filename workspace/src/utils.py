import tensorflow as tf
import json


def load_image(i):
    byte_img = tf.io.read_file(i)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding="utf-8") as f:
        label = json.load(f)

    return [label['class']], label['bbox']
