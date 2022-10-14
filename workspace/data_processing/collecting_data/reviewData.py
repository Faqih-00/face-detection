import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_image

images = tf.data.Dataset.list_files('data/plain/train/images/*.jpg')
images = images.map(load_image)
image_generator = images.batch(4).as_numpy_iterator()
plot_images = image_generator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()
