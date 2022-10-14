import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_image(i):
    byte_img = tf.io.read_file(i)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (224, 224))
    img = img / 255
    return img

images = tf.data.Dataset.list_files('data/plain/train/images/*.jpg')
images = images.map(load_image)
image_generator = images.batch(4).as_numpy_iterator()
plot_images = image_generator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()
