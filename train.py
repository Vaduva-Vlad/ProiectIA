import matplotlib.pyplot as pyplot
import PIL
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import pathlib
import numpy

dataset_directory=pathlib.Path('minet')
image_count = len(list(dataset_directory.glob('*/*.jpg')))

batch_size=32
height=200
width=200

training_dataset = tf.keras.utils.image_dataset_from_directory(
dataset_directory,
validation_split=0.2,
seed=123,
subset="training",
batch_size=batch_size,
image_size=(height,width)
)

validation_dataset=tf.keras.utils.image_dataset_from_directory(
dataset_directory,
validation_split=0.2,
subset="training",
seed=123,
batch_size=batch_size,
image_size=(height,width)
)

class_names=training_dataset.class_names

AUTOTUNE=tf.data.AUTOTUNE
training_dataset=training_dataset.cache().shuffle(999).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

normalization_lyr = layers.Rescaling(1./255)
normalized_dataset=training_dataset.map(lambda x,y:(normalization_lyr(x),y))

num_of_classes=len(class_names)
data_augmentation=keras.Sequential(
    [layers.RandomFlip("horizontal",input_shape=(height,width,3)),layers.RandomRotation(0.1),layers.RandomZoom(0.1)])

model=Sequential([
data_augmentation,
layers.Rescaling(1./255),
layers.Conv2D(16,3,padding="same", activation="relu"),
layers.MaxPooling2D(),
layers.Conv2D(32,3,padding="same",activation="relu"),
layers.MaxPooling2D(),
layers.Conv2D(64,3,padding="same",activation="relu"),
layers.MaxPooling2D(),
layers.Dropout(0.2),
layers.Flatten(),
layers.Dense(128,activation="relu"),
layers.Dense(num_of_classes,name="outputs")
])

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
model.summary()

epochs=40
history=model.fit(training_dataset,validation_data=validation_dataset,epochs=epochs)

accuracy=history.history["accuracy"]
val_accuracy=history.history["val_accuracy"]

loss=history.history["loss"]
val_loss=history.history["val_loss"]

pyplot.figure(figsize=(9,9))
pyplot.subplot(1,2,1)
pyplot.plot(range(epochs),accuracy,label="Training Accuracy")
pyplot.plot(range(epochs),val_accuracy,label="Validation Accuracy")
pyplot.legend(loc="lower right")
pyplot.title("Training and Validation Accuracy")

pyplot.subplot(1,2,1)
pyplot.plot(range(epochs),accuracy,label="Training Loss")
pyplot.plot(range(epochs),val_accuracy,label="Validation Loss")
pyplot.legend(loc="upper right")
pyplot.title("Training and Validation Loss")
pyplot.show()

model.save("trained_model")