# main.py
# created 03/02/2022 at 12:02 by Antoine 'AatroXiss' BEAUDESSON
# last modified 04/02/2022 at 12:42 by Antoine 'AatroXiss' BEAUDESSON

""" main.py:
    - *
"""

__author__ = "Antoine 'AatroXiss' BEAUDESSON"
__copyright__ = "Copyright 2021, Antoine 'AatroXiss' BEAUDESSON"
__credits__ = ["Antoine 'AatroXiss' BEAUDESSON"]
__license__ = ""
__version__ = "0.0.7"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports
import os

# third party imports
import pandas as pd
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from sklearn.model_selection import train_test_split

# local application imports

# other imports

# constants
TRAIN_PATH_FR = 'dataset/train/France'
TRAIN_PATH_US = 'dataset/train/United States'
TEST_PATH = 'dataset/test'


def loading_dataset(path):
    """
    This funtion loads the dataset and returns it as a
    a list.
    """
    filenames = os.listdir(path)
    return filenames


def construct_dataset(train_files_fr, train_files_us):
    """
    This function constructs the dataset.
    """
    label = []
    dataset = []
    for filename in train_files_fr:
        dataset.append(filename)
        label.append('France')

    # Get the same number of us files as fr files
    number_of_france_files = len(dataset)
    i = 0
    while i < number_of_france_files:
        dataset.append(train_files_us[i])
        label.append('United States')
        i += 1

    df = pd.DataFrame({'filename': dataset, 'label': label})
    return df


def cnn_architecture():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3))) # noqa
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
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics='acc')
    model.summary()


def main():
    train_files_fr = loading_dataset(TRAIN_PATH_FR)
    train_files_us = loading_dataset(TRAIN_PATH_US)
    df = construct_dataset(train_files_fr, train_files_us)
    cnn_architecture()

    # split datas in train and test and validation
    train, test_val = train_test_split(df, test_size=0.5,
                                       stratify=df['label'],
                                       random_state=17)


main()
