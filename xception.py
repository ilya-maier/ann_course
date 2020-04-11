#!pip install deeplearning2020
import tensorflow as tf
import matplotlib.pyplot as plt
from deeplearning2020 import helpers
from deeplearning2020.datasets import ImageWoof
from tensorflow.python.keras.applications import xception
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, \
    Dropout
from tensorflow.python.keras.models import Model


# a cnn model based on Xception (transfer learning) for imagewoof dataset
# -> achieves more than 95%

def preprocess(image, label):
    resized_image = tf.image.resize(image, (300, 300))
    return resized_image, label


batch_size = 32

train_data, test_data, classes = ImageWoof.load_data()
train_data = train_data.map(preprocess).batch(batch_size).prefetch(1)
test_data = test_data.map(preprocess).batch(batch_size).prefetch(1)

base_model = xception.Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(300, 300, 3)
)
model = GlobalAveragePooling2D()(base_model.output)
model = Dropout(0.5)(model)
model = Dense(512, activation='relu')(model)
model = Dropout(0.5)(model)
model = Dense(256, activation='relu')(model)
model = Dropout(0.5)(model)
output = Dense(10, activation='softmax')(model)

model = Model(base_model.input, output)
# model.summary()

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer="Adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
history = model.fit(
    train_data,
    epochs=5,
    validation_data=test_data
)
helpers.plot_history("After 5 epochs:", history)
plt.show()

for layer in base_model.layers[:20]:
    layer.trainable = True

model.compile(
    optimizer="Adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data
)
helpers.plot_history("After 15 epochs:", history)
plt.show()
