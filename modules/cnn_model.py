# cnn_model.py
# created 24/02/2022 at 11:36 by Antoine 'AatroXiss' BEAUDESSON
# last modified 21/03/2022 at 01:42 by Antoine 'AatroXiss' BEAUDESSON

""" cnn_model.py:
    - *
"""

__author__ = "Antoine 'AatroXiss' BEAUDESSON"
__copyright__ = "Copyright 2021, Antoine 'AatroXiss' BEAUDESSON"
__credits__ = ["Antoine 'AatroXiss' BEAUDESSON"]
__license__ = ""
__version__ = "0.1.3"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports

# third party imports
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras import layers
import tensorflow as tf

# local application imports
from modules.data import (
    data_augmentation
)


def cnn_model(class_names, img_height, img_width):
    model = Sequential(
    [
        data_augmentation(img_height, img_width),
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),  # or 255
        
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(len(class_names)),

    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # noqa
                  metrics=['accuracy'])

    model.summary()
    
    return model

def get_callbacks():
    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                patience=10, 
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.0001)
    callbacks = [earlystop, learning_rate_reduction]
    return callbacks