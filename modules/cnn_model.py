# cnn_model.py
# created 24/02/2022 at 11:36 by Antoine 'AatroXiss' BEAUDESSON
# last modified 24/02/2022 at 11:36 by Antoine 'AatroXiss' BEAUDESSON

""" cnn_model.py:
    - *
"""

__author__ = "Antoine 'AatroXiss' BEAUDESSON"
__copyright__ = "Copyright 2021, Antoine 'AatroXiss' BEAUDESSON"
__credits__ = ["Antoine 'AatroXiss' BEAUDESSON"]
__license__ = ""
__version__ = "0.1.1"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports

# third party imports
from keras.models import Sequential
from keras import layers
import tensorflow as tf

# local application imports
from modules.data import (
    data_augmentation
)


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
