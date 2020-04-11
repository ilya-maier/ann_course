import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.core import Dense, Flatten
from tensorflow.python.keras.models import Model

tf.compat.v1.disable_eager_execution()


# a simple cnn model for imagenette dataset

def preprocess(image, label):
    resized_imaged = tf.image.resize(image / 255, [300, 300])
    return resized_imaged, label


data, info = tfds.load(
    "imagenette/320px",
    as_supervised=True,
    with_info=True
)
train_data = data["train"]
test_data = data["validation"]

batch_size = 32

train_data = train_data.shuffle(1000)

train_data = train_data.map(preprocess).batch(batch_size).prefetch(1)
test_data = test_data.map(preprocess).batch(batch_size).prefetch(1)

learning_rate = 0.001
dense_neurons = 256
n_filters = 256
kernel = (3, 3)
n_classes = info.features['label'].num_classes
activation = 'elu'

input_layer = Input(shape=(300, 300, 3))

model = Conv2D(
    filters=n_filters,
    kernel_size=(3, 3),
    activation=activation
)(input_layer)
model = MaxPooling2D((2, 2))(model)

model = Conv2D(
    filters=n_filters,
    kernel_size=kernel,
    activation=activation
)(model)
model = MaxPooling2D((2, 2))(model)

model = Conv2D(
    filters=n_filters * 2,
    kernel_size=kernel,
    activation=activation
)(model)
model = MaxPooling2D((2, 2))(model)

model = Conv2D(
    filters=n_filters * 2,
    kernel_size=kernel,
    activation=activation
)(model)
model = MaxPooling2D((2, 2))(model)

model = Conv2D(
    filters=n_filters * 2,
    kernel_size=kernel,
    activation=activation
)(model)
model = MaxPooling2D((2, 2))(model)

model = Conv2D(
    filters=n_filters * 2,
    kernel_size=kernel,
    activation=activation,
    padding='same'
)(model)
model = MaxPooling2D((2, 2))(model)

model = Flatten()(model)
model = Dense(
    dense_neurons,
    activation=activation
)(model)

model = Dense(
    dense_neurons / 2,
    activation='tanh'
)(model)

output = Dense(
    n_classes,
    activation="softmax"
)(model)

cnn_model = Model(input_layer, output)

cnn_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.SGD(lr=learning_rate, momentum=0.9),
    metrics=["accuracy"]
)
cnn_model.summary()
history = cnn_model.fit(
    train_data,
    epochs=12,
    validation_data=test_data
)
