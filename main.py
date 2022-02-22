# main.py
# created 03/02/2022 at 12:02 by Antoine 'AatroXiss' BEAUDESSON
# last modified 14/02/2022 at 11:07 by Antoine 'AatroXiss' BEAUDESSON

""" main.py:
    - *
"""

__author__ = "Antoine 'AatroXiss' BEAUDESSON"
__copyright__ = "Copyright 2021, Antoine 'AatroXiss' BEAUDESSON"
__credits__ = ["Antoine 'AatroXiss' BEAUDESSON"]
__license__ = ""
__version__ = "0.0.17"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports

# third party imports

# local application imports
from modules.dataset_management import (
    explore_data,
    load_data,
)
from modules.model import (

    use_pretrained_model,
    train_model,
)

# other imports

# constants
PATH_TRAIN_DATASET = 'dataset/train/'
IMG_TEST_PATH_US = 'dataset/test/us/canvas_1629257624.jpg'
IMG_TEST_PATH_FR = 'dataset/test/fr/canvas_1629257785.jpg'

VAL_PATH = IMG_TEST_PATH_FR

BATCH_SIZE = 32
IMG_HEIGHT = 150
IMG_WIDTH = 300
EPOCHS = 1


def main():
    """
    """

    train_dir = explore_data(PATH_TRAIN_DATASET)
    train_ds = load_data(train_dir, 'training',
                         BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)
    val_ds = load_data(train_dir, 'validation',
                       BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)

    class_names = train_ds.class_names

    # Ask for the user which model he wants to use
    user_input = input('Use pretrained model? (0/1)')
    if user_input == '0':
        use_pretrained_model(class_names, IMG_HEIGHT, IMG_WIDTH, VAL_PATH)
    elif user_input == '1':
        train_model(
            class_names,
            IMG_HEIGHT,
            IMG_WIDTH,
            train_ds,
            val_ds,
            EPOCHS,
            VAL_PATH
        )
    else:
        print("Please enter a valid input (0/1)")


if __name__ == "__main__":
    main()
