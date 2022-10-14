from tensorflow.python.keras.models import load_model
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
from src.utils import load_image, load_labels

model = load_model("facetracker.h5")

test_images = tf.data.Dataset.list_files(
    'data\\augmented\\test\\images\\*.jpg')
test_images = test_images.map(load_image)

test_labels = tf.data.Dataset.list_files(
    'data\\augmented\\test\\labels\\*.json')
test_labels = test_labels.map(lambda x: tf.py_function(
    load_labels, [x], [tf.uint8, tf.float16]))

test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(3000)
test = test.batch(8)
test = test.prefetch(4)

test_data = test.as_numpy_iterator()
test_sample = test_data.next()
yhat = model.predict(test_sample[0])

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx in range(4):
    sample_image = test_sample[0][idx]
    sample_coords = yhat[1][idx]

    if yhat[0][idx] > 0.9:
        cv2.rectangle(sample_image,
                      tuple(np.multiply(sample_coords[:2], [
                            120, 120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [
                            120, 120]).astype(int)),
                      (128, 0, 0), 2)

    ax[idx].imshow(sample_image)

plt.show()
