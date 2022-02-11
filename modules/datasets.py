# datasets.py
# created 11/02/2022 at 17:41 by Antoine 'AatroXiss' BEAUDESSON
# last modified 11/02/2022 at 17:41 by Antoine 'AatroXiss' BEAUDESSON

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
import os
from random import randint

# third party imports
from keras.preprocessing.image import load_img, ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd

# local application imports

# other imports

# constants
PATH_ALL = 'dataset/train/all'


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

    # get the same number of images for each class
    number_of_images = len(filenames_fr)
    i = 0
    while i < number_of_images:
        filename.append(filenames_us[i])
        label.append('us')
        i += 1

    df = pd.DataFrame({'filename': filename, 'label': label})

    return df


def select_random_image(df):
    """
    This function selects a random image from the dataset.
    """
    # select a random image from the dataset
    f = randint(0, len(df))
    image_path = PATH_ALL + '/' + df['filename'][f]
    image_label = df['label'][f]

    return image_path, image_label


def check_dataset(df):

    plt.figure(figsize=(192, 108))

    # display 25 random images from the dataset
    i = 0
    while i < 25:
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        # select a random image from the dataset
        image_path, image_label = select_random_image(df)

        plt.imshow(load_img(image_path))
        plt.xlabel(image_label)

        # update i
        i += 1
    plt.show()
    print("Done")


def image_data_gen(df, path):
    gen = ImageDataGenerator(rescale=1. / 255)
    data = gen.flow_from_dataframe(dataframe=df,
                                   directory=path,
                                   x_col='filename',
                                   y_col='label',
                                   target_size=(768, 331),
                                   class_mode='binary',
                                   seed=42,
                                   shuffle=True)
    return data
