import numpy as np
import tensorflow as tf
import pathlib

from tensorflow import keras

filename=input("Introduceti numele fisierului:")

batch_size = 32
height = 200
width = 200

class_names=['biotite', 'bornite', 'chrysocolla', 'malachite', 'muscovite', 'pyrite', 'quartz']

model=keras.models.load_model("trained_model")

img = tf.keras.utils.load_img(
    filename, target_size=(height, width))
images = tf.keras.utils.img_to_array(img)
images = tf.expand_dims(images, 0)

predictions = model.predict(images)
score = tf.nn.softmax(predictions[0])

print("This is an image of {} with a {:.2f} % confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
