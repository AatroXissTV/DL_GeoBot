# model.py
# created 11/02/2022 at 17:47 by Antoine 'AatroXiss' BEAUDESSON
# last modified 11/02/2022 at 17:47 by Antoine 'AatroXiss' BEAUDESSON

""" datasets.py:
    - *
"""

__author__ = "Antoine 'AatroXiss' BEAUDESSON"
__copyright__ = "Copyright 2021, Antoine 'AatroXiss' BEAUDESSON"
__credits__ = ["Antoine 'AatroXiss' BEAUDESSON"]
__license__ = ""
__version__ = "0.0.12"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports

# third party imports
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras import layers, models

# local application imports

# other imports

# constants


def cnn_model():
    """
    """

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
              input_shape=(768, 331, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # noqa
                  metrics=['accuracy'])

    return model


def cnn_model_2():
    """
    """

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(768, 331, 3)))

    # 1st convolutional layer
    model.add(layers.Conv2D(25, (5, 5), activation='relu',
                            strides=(1, 1), padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding='same'))
    # 2nd convolutional layer
    model.add(layers.Conv2D(50, (5, 5), activation='relu',
                            strides=(2, 2), padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    # 3rd convolutional layer
    model.add(layers.Conv2D(70, (3, 3), activation='relu',
                            strides=(2, 2), padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), padding='valid'))
    model.add(layers.BatchNormalization())
    # ANN block
    model.add(layers.Flatten())
    model.add(layers.Dense(units=100, activation='relu'))
    model.add(layers.Dense(units=100, activation='relu'))
    model.add(layers.Dropout(0.25))
    # output layer
    model.add(layers.Dense(units=10, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # noqa
                  metrics=['accuracy'])


def evaluate_model_val_loss(history, epochs):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(15, 8))
    plt.plot(loss, label='Train loss')
    plt.plot(val_loss, '--', label='Val loss')
    plt.title('Training and validation loss')
    plt.xticks(np.arange(0, epochs))
    plt.yticks(np.arange(0, 0.7, 0.05))
    plt.grid()
    plt.legend()
    plt.show()


def evaluate_model_val_acc(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.figure(figsize=(15, 8))
    plt.plot(acc, label='Train accuracy')
    plt.plot(val_acc, '--', label='Val accuracy')
    plt.title('Training and validation accuracy')
    plt.xticks(np.arange(0, epochs))
    plt.yticks(np.arange(0, 1, 0.05))
    plt.grid()
    plt.legend()
    plt.show()
