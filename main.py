# main.py
# created 03/02/2022 at 12:02 by Antoine 'AatroXiss' BEAUDESSON
# last modified 17/03/2022 at 14:02 by Antoine 'AatroXiss' BEAUDESSON

""" main.py:
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

# local application imports
from modules.tensorflow_check import (
    check_tensorflow_version,
    check_tensorflow_gpu,
)
from modules.load_data import (
    load_data,
    explore_ds,
)
from modules.model import (
    train_model,
    evaluate_model,
)
from modules.cnn_model import (
    get_callbacks
)

# other imports & constants
TRAIN_PATH = "resources/cleaned_dataset"
TEST_PATH = "resources/test_dataset"

IMG_HEIGHT = int(662 / 2)
IMG_WIDTH = int(1536 / 2)
BATCH_SIZE = 32
EPOCHS = 100


def main():
    """
    This function is the entry point of the program.
    GeoBot is a CNN that can detect which country a picture is from.
    """

    # check tensorflow version & GPU availability
    check_tensorflow_version()
    check_tensorflow_gpu()
    input("Press Enter to continue...\n")

    # explore directories
    print("Exploring directories...")
    train_dir = explore_ds(TRAIN_PATH, "train")
    test_dir = explore_ds(TEST_PATH, "test")
    input("Press Enter to continue...\n")

    # load data
    print("Loading data...")
    train_ds, val_ds, test_ds, class_names = load_data(train_dir, test_dir,
                                                       IMG_HEIGHT, IMG_WIDTH,
                                                       BATCH_SIZE)
    print("Data loaded.")
    input("Press Enter to continue...\n")

    # print get callbacks
    print("Getting callbacks...")
    callbacks = get_callbacks()

    # Train the model
    print("Training the model...")
    model = train_model(train_ds, val_ds,
                        IMG_HEIGHT, IMG_WIDTH,
                        class_names, EPOCHS,
                        callbacks)
    input("Press Enter to continue...\n")

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, test_ds)
    # Ask the user if he wants to save the model
    save_model = input("Do you want to save the model? (y/n) ")
    if save_model == "y":
        model.save("model.h5")
        print("Model saved.")
    input("Press Enter to continue...\n")  # wait for user input


if __name__ == "__main__":
    main()
