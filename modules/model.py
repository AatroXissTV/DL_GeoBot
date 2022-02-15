# model.py
# created 14/02/2022 at 16:52 by Antoine 'AatroXiss' BEAUDESSON
# last modified 14/02/2022 at 16:52 by Antoine 'AatroXiss' BEAUDESSON

""" model.py:
    - *
"""

__author__ = "Antoine 'AatroXiss' BEAUDESSON"
__copyright__ = "Copyright 2021, Antoine 'AatroXiss' BEAUDESSON"
__credits__ = ["Antoine 'AatroXiss' BEAUDESSON"]
__license__ = ""
__version__ = "0.0.15"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports

# third party imports
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
import tensorflow as tf

# local application imports
from modules.dataset_management import (
    data_augmentation
)

# other imports

# constants


def cnn_model(class_names, img_height, img_width):
    """
    """

    num_classes = len(class_names)
    model = Sequential(
        [
            data_augmentation(img_height, img_width),
            layers.Rescaling(1./255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ]
    )

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # noqa
                  metrics=['accuracy'])

    return model


def visualize_val_acc(epochs, history):
    """
    """

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_range = range(epochs)
    print(epochs_range)

    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.show()
    input("Press Enter to continue...")


def visualize_val_loss(epochs, history):
    """
    """

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    input("Press Enter to continue...")
