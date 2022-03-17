# data.py
# created 24/02/2022 at 11:02 by Antoine 'AatroXiss' BEAUDESSON
# last modified 24/02/2022 at 11:10 by Antoine 'AatroXiss' BEAUDESSON

""" data.py:
    - *
"""

__author__ = "Antoine 'AatroXiss' BEAUDESSON"
__copyright__ = "Copyright 2021, Antoine 'AatroXiss' BEAUDESSON"
__credits__ = ["Antoine 'AatroXiss' BEAUDESSON"]
__license__ = ""
__version__ = "0.1.2"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports

# third party imports
import tensorflow as tf

# local application imports

# other imports & constants


def data_augmentation(img_height, img_width):

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal",
                                       input_shape=(img_height, img_width, 3)),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )

    return data_augmentation


def load_data_from_dir(ds_dir, validation_split, subset,
                       img_height, img_width, batch_size):
    """
    This function loads the data from the given directory.
    It uses the image_dataset_from_directory function and splits the dataset
    into a training (80%) and validation (20%) set.
    :arg data_dir: the path to the dataset
    :arg validation_split: the % of the dataset to be used for validation
    :arg subset: the name of the subset
    :arg batch_size: the batch size
    :arg img_height: the height of the images
    :arg img_width: the width of the images
    :return: the dataset
    """

    ds = tf.keras.utils.image_dataset_from_directory(ds_dir,
                                                     validation_split=validation_split,  # noqa
                                                     subset=subset,
                                                     seed=123,
                                                     image_size=(img_height,
                                                                 img_width),
                                                     batch_size=batch_size)
    return ds


def check_datasets_class_names(train_ds, test_ds):
    """
    This function checks if the class_names between the training and testing
    datasets are the same.
    :arg train_ds: the training dataset
    :arg test_ds: the testing dataset
    :return: None
    """

    class_names_train = train_ds.class_names
    class_names_test = test_ds.class_names

    if class_names_train != class_names_test:
        raise ValueError("The training and testing datasets have different class names")  # noqa: E501
    else:
        pass

    return class_names_train
