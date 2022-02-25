# load_data.py
# created 25/02/2022 at 12:02 by Antoine 'AatroXiss' BEAUDESSON
# last modified 25/02/2022 at 12:10 by Antoine 'AatroXiss' BEAUDESSON

""" data.py:
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
from pathlib import Path

# third party imports

# local application imports
from modules.data import (
    load_data_from_dir,
    check_datasets_class_names,
)

# other imports & constant


def explore_ds(path, dir_name):
    """
    This function explores the given directory
    and display the number of jpg files in it.
    if the directory has no images, throw an error.

    :arg path: path of the directory
    :return: the complete path
    """

    data_dir = Path(path)
    image_count = len(list(data_dir.glob("*/*.jpg")))

    # Check if the directory has images
    if image_count == 0:
        raise ValueError("The given directory has no images")
    else:
        print(f"The {dir_name} directory contains {image_count} images")
    return data_dir


def load_data(train_dir, test_dir, img_height, img_width, batch_size):
    """
    This function is used to load the different datasets
    needed for the application.
    It uses the given directories to load the images.

    :arg train_dir: the path to the training dataset
    :arg test_dir: the path to the testing dataset
    :return: train_ds, val_ds, test_ds, class_names
    """
    train_ds = load_data_from_dir(
        train_dir,
        0.2,
        'training',
        img_height=img_height,
        img_width=img_width,
        batch_size=batch_size,
    )
    val_ds = load_data_from_dir(
        train_dir,
        0.2,
        'validation',
        img_height=img_height,
        img_width=img_width,
        batch_size=batch_size,
    )
    test_ds = load_data_from_dir(
        test_dir,
        0.2,
        'training',
        img_height=img_height,
        img_width=img_width,
        batch_size=batch_size,
    )

    class_names = check_datasets_class_names(train_ds, test_ds)

    return train_ds, val_ds, test_ds, class_names
