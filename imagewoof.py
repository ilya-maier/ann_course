#!pip install deeplearning2020
import tensorflow as tf
from deeplearning2020 import helpers
from deeplearning2020.datasets import ImageWoof
from tensorflow import keras
from tensorflow.python.keras.layers import BatchNormalization, GaussianNoise
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.core import Dense, Flatten, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


# a ccn model for a more complex imagewoof dataset, which is a bit deeper:
# contains batch norm, dropout and data augmentation

def preprocess(image, label):
    resized_imaged = tf.image.resize(image, [300, 300])
    return resized_imaged, label


batch_size = 32
train_data, test_data, classes = ImageWoof.load_data()

x, y, x_test, y_test = helpers.dataset_to_ndarray(train_data, test_data)


def create_model(
        noise=True,
        first_kernel_size=(7, 7),
        n_filters=64,
        n_covul_layers=5,
        activation='swish',
        dense_neurons=1024,
        dropout=0.5,
        lr=0.0001):
    kernel = (3, 3)
    n_classes = len(classes)

    input_layer = Input(shape=(300, 300, 3))
    if noise:
        input_layer = GaussianNoise(0.1)(input_layer)

    model = BatchNormalization(axis=[1, 2])(input_layer)

    model = Conv2D(
        filters=n_filters,
        kernel_size=first_kernel_size,
        activation=activation
    )(model)
    model = BatchNormalization(axis=[1, 2])(model)
    model = MaxPooling2D((2, 2))(model)

    for i in range(2, n_covul_layers):
        model = Conv2D(
            filters=n_filters * i,
            kernel_size=kernel,
            activation=activation
        )(model)
        model = Conv2D(
            filters=n_filters * i,
            kernel_size=kernel,
            activation=activation,
            padding='same'
        )(model)
        model = BatchNormalization(axis=[1, 2])(model)
        model = MaxPooling2D((2, 2))(model)

    model = Conv2D(
        filters=n_filters * (n_covul_layers + 1),
        kernel_size=kernel,
        activation=activation,
        padding='same'
    )(model)
    model = Conv2D(
        filters=n_filters * (n_covul_layers + 1),
        kernel_size=kernel,
        activation=activation,
        padding='same'
    )(model)
    model = BatchNormalization(axis=[1, 2])(model)
    model = MaxPooling2D((2, 2))(model)

    model = Flatten()(model)
    model = Dense(
        dense_neurons,
        activation=activation
    )(model)
    model = BatchNormalization()(model)
    model = Dropout(dropout)(model)

    model = Dense(
        dense_neurons / 2,
        activation=activation
    )(model)
    model = BatchNormalization()(model)
    model = Dropout(dropout)(model)

    output = Dense(
        n_classes,
        activation="softmax"
    )(model)

    model = Model(input_layer, output)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(lr=lr),
        metrics=["accuracy"]
    )

    return model


cnn_model = create_model()
cnn_model.summary()

generator = ImageDataGenerator(
    rotation_range=0.3,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True
)
train_generator = generator.flow(x, y)
test_generator = generator.flow(x_test, y_test)

x_augmented, y_augmented = train_generator.next()
helpers.plot_images_with_labels((x_augmented[:9] + 1) / 2,
                                y_augmented[:9].astype('int'), classes)

for t in range(4):
    history = cnn_model.fit(
        train_generator,
        epochs=10,
        validation_data=test_generator
    )
    helpers.plot_history(f"After {t * 10} epochs:", history)
