import tensorflow as tf
import matplotlib.pyplot as plt
import json
import numpy as np
from customModel import FaceDetection, localization_loss, build_model
from src.utils import load_image, load_labels

# ToDo:
# Load images and labels for train and val augmented data
# Concatinate image and labels into 1 train and val

train_images = tf.data.Dataset.list_files(
    'data\\augmented\\train\\images\\*.jpg', shuffle=False)
train_images = train_images.map(load_image)

val_images = tf.data.Dataset.list_files(
    'data\\augmented\\val\\images\\*.jpg', shuffle=False)
val_images = val_images.map(load_image)


train_labels = tf.data.Dataset.list_files(
    'data\\augmented\\train\\labels\\*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(
    load_labels, [x], [tf.uint8, tf.float16]))

val_labels = tf.data.Dataset.list_files(
    'data\\augmented\\val\\labels\\*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(
    load_labels, [x], [tf.uint8, tf.float16]))


train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(10000)
train = train.batch(8)
train = train.prefetch(4)

val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(3000)
val = val.batch(8)
val = val.prefetch(4)

batch_per_epoch = len(train)
decay = (1./0.75-1) / batch_per_epoch

facetracker = build_model()
model = FaceDetection(facetracker)
opt = tf.keras.optimizers.Adam(learning_rate=1e-4, decay=decay)
classloss = tf.keras.losses.BinaryCrossentropy()
regloss = localization_loss
model.compile(opt, classloss, regloss)

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=10, validation_data=val,
                 callbacks=[tensorboard_callback])

facetracker.save('facetracker.h5')

fig, ax = plt.subplots(ncols=3, figsize=(20, 5))

ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
ax[0].title.set_text('Loss')
ax[0].legend()

ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
ax[1].plot(hist.history['val_class_loss'],
           color='orange', label='val class loss')
ax[1].title.set_text('Classification Loss')
ax[1].legend()

ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
ax[2].plot(hist.history['val_regress_loss'],
           color='orange', label='val regress loss')
ax[2].title.set_text('Regression Loss')
ax[2].legend()

plt.show()
