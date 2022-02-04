# main.py
# created 03/02/2022 at 12:02 by Antoine 'AatroXiss' BEAUDESSON
# last modified 03/02/2022 at 12:02 by Antoine 'AatroXiss' BEAUDESSON

""" main.py:
    - *
"""

__author__ = "Antoine 'AatroXiss' BEAUDESSON"
__copyright__ = "Copyright 2021, Antoine 'AatroXiss' BEAUDESSON"
__credits__ = ["Antoine 'AatroXiss' BEAUDESSON"]
__license__ = ""
__version__ = "0.0.5"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports
import os

# third party imports
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
import tensorflow as tf

# local application imports

# other imports

# constants

labels = [
    "France",
    "United States",
]
img_size = 224


def get_data(dataset_dir):
    data = []
    for label in labels:
        path = os.path.join(dataset_dir, label)
        class_num = labels.index(label)
        for img_path in os.listdir(path):
            try:
                img_arr = cv2.imread(
                    os.path.join(path, img_path))[..., ::-1]  # BGR to RGB
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(f"Error while loading {img_path}: {e}")
    return np.array(data)


def data_preprocessing(data):
    x = []
    y = []

    for feature, label in data:
        x.append(feature)
        y.append(label)

    return x, y


def data_normalization(x, y):
    x = np.array(x) / 255.0
    x.reshape(-1, img_size, img_size, 1)
    y = np.array(y)

    return x, y


def data_augmentation(x):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    return datagen.fit(x)


def model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Conv2D(
        32, 3, padding="same", activation="relu", input_shape=(224, 224, 3)))
    model.add(MaxPool2D())

    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())

    model.add(Conv2D(64, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    model.summary()
    opt = Adam(lr=0.000001)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        epochs=500,
                        validation_data=(x_test, y_test))

    return history


def main():
    train = get_data("dataset/train")
    test = get_data("dataset/test")
    x_train, y_train = data_preprocessing(train)
    x_test, y_test = data_preprocessing(test)
    x_train, y_train = data_normalization(x_train, y_train)
    x_test, y_test = data_normalization(x_test, y_test)
    data_augmentation(x_train)
    history = model(x_train, y_train, x_test, y_test)
    print(history)


main()
