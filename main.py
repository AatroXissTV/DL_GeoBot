# main.py
# created 03/02/2022 at 12:02 by Antoine 'AatroXiss' BEAUDESSON
# last modified 10/02/2022 at 15:02 by Antoine 'AatroXiss' BEAUDESSON

""" main.py:
    - *
"""

__author__ = "Antoine 'AatroXiss' BEAUDESSON"
__copyright__ = "Copyright 2021, Antoine 'AatroXiss' BEAUDESSON"
__credits__ = ["Antoine 'AatroXiss' BEAUDESSON"]
__license__ = ""
__version__ = "0.0.10"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports
import os

# third party imports
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.keras.callbacks import EarlyStopping

# local application imports

# other imports

# constants
TRAIN_PATH_FR = 'dataset/train/France'
TRAIN_PATH_US = 'dataset/train/United States'
TRAIN_PATH = 'dataset/train/all'


def loading_dataset(path):
    """
    This function loads the dataset and returns it
    as a list.
    """
    filenames = os.listdir(path)
    return filenames


def construct_dataset(filenames_fr, filenames_us):
    """
    This function constructs the dataset with
    the filenames.
    """

    label = []
    filename = []
    for f in filenames_fr:
        filename.append(f)
        label.append('fr')

    # Get the same number of files for 'fr' and 'us'
    number_of_fr_files = len(filename)
    i = 0
    while i < number_of_fr_files:
        filename.append(filenames_us[i])
        label.append('us')
        i += 1

    df = pd.DataFrame({'filename': filename, 'label': label})
    return df


def check_dataset(path, df):
    """
    This function allows you to check if
    the images in the dataset are correctly loaded
    """

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(load_img(path+'/'+df['filename'][i]))

    plt.show()


def cnn_architecture():
    """
    Define a simple CNN model with 13 convolutional layers
    using a pile of Conv2D, MaxPooling2D layers
    """

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3),
              activation='relu',
              input_shape=(256, 256, 3)))
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
    model.add(layers.Dense(512,
                           activation='relu',
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=adam.Adam(learning_rate=5e-4),
                  loss='binary_crossentropy',
                  metrics='acc')

    # model.summary()
    return model


def image_data_generator(df, path):
    gen = ImageDataGenerator(rescale=1./255,
                             rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')
    data = gen.flow_from_dataframe(
        dataframe=df,
        directory=path,
        x_col='filename',
        y_col='label',
        class_mode='binary',
        target_size=(256, 256),
        seed=42,
        batch_size=32,
    )
    return data


def main():
    filenames_fr = loading_dataset(TRAIN_PATH_FR)
    filenames_us = loading_dataset(TRAIN_PATH_US)
    df = construct_dataset(filenames_fr, filenames_us)

    # check df
    # check_dataset(TRAIN_PATH, df)

    # split datas into train, test and validation
    train, test_val = train_test_split(
        df,
        test_size=0.5,
        stratify=df['label'],
        random_state=17
    )
    test, val = train_test_split(
        test_val,
        test_size=0.5,
        stratify=test_val['label'],
        random_state=17
    )

    # Generating Artificial Images
    train_data = image_data_generator(train, TRAIN_PATH)
    print(len(train_data))
    val_data = image_data_generator(val, TRAIN_PATH)

    # Training the model
    model = cnn_architecture()
    history = model.fit(train_data,
                        validation_data=val_data,
                        epochs=60,
                        batch_size=32,
                        steps_per_epoch=len(train_data),
                        validation_steps=len(val_data),
                        callbacks=[EarlyStopping(monitor='val_acc',
                                                 min_delta=0.001,
                                                 patience=5,
                                                 verbose=1)])

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(15, 8))
    plt.plot(loss, label='Train loss')
    plt.plot(val_loss, '--', label='Val loss')
    plt.title('Training and validation loss')
    plt.xticks(np.arange(0, 10))
    plt.yticks(np.arange(0, 0.7, 0.05))
    plt.grid()
    plt.legend()
    plt.show()

    loss = history.history['acc']
    val_loss = history.history['val_acc']

    plt.figure(figsize=(15, 8))
    plt.plot(loss, label='Train acc')
    plt.plot(val_loss, '--', label='Val acc')
    plt.title('Training and validation accuracy')
    plt.yticks(np.arange(0.5, 1, 0.05))
    plt.xticks(np.arange(0, 61))
    plt.grid()
    plt.legend()
    plt.show()


main()
