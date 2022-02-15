# dataset_management.py
# created 14/02/2022 at 11:07 by Antoine 'AatroXiss' BEAUDESSON
# last modified 14/02/2022 at 11:07 by Antoine 'AatroXiss' BEAUDESSON

""" main.py:
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
import pathlib

# third party imports
import tensorflow as tf
import matplotlib.pyplot as plt

# local application imports

# other imports

# constants


def explore_data(path_dataset):
    """
    This function explores the given directory and
    returns the number of jpg files in it.

    :arg path_dataset: the path to the dataset
    :return: the path of the directory
    """
    data_dir = pathlib.Path(path_dataset)
    image_count = len(list(data_dir.glob("*/*.jpg")))
    print(f'There is {image_count} images')

    return data_dir


def load_data(data_dir, subset, batch_size, img_height, img_width):
    """
    This function loads the data from the given directory.
    It uses the image_dataset_from_directory function and splits the dataset
    into a training (80%) and validation (20%) set.

    :arg data_dir: the path to the dataset
    :arg subset: the name of the subset
    :arg batch_size: the batch size
    :arg img_height: the height of the images
    :arg img_width: the width of the images
    :return: the dataset
    """
    ds = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                     validation_split=0.2,
                                                     subset=subset,
                                                     seed=123,
                                                     image_size=(img_height,
                                                                 img_width),
                                                     batch_size=batch_size)
    return ds


def data_augmentation(img_height, img_width):
    """
    """

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal",
                                       input_shape=(img_height, img_width, 3)),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )

    return data_augmentation


def visualize_data(ds, class_names):
    """
    This function visualizes the 9 first images in the dataset.

    :arg ds: the dataset
    :arg class_names: the class names
    """
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()
    input("Press Enter to continue...")


def visualize_data_augmentation(ds, img_height, img_width):
    """
    """
    plt.figure(figsize=(10, 10))
    for images, _ in ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(img_height, img_width)(images)
            plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")
    plt.show()
    input("Press Enter to continue...")
