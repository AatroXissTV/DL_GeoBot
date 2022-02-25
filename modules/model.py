# model.py
# created 24/02/2022 at 11:36 by Antoine 'AatroXiss' BEAUDESSON
# last modified 24/02/2022 at 11:36 by Antoine 'AatroXiss' BEAUDESSON

""" model.py:
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

# third party imports

# local application imports
from modules.cnn_model import (
    cnn_model,
)
from modules.visualize_results import (
    visualize_val_acc,
    visualize_val_loss,
)

# other imports & constants


def train_model(train_ds, val_ds,
                img_height, img_width,
                class_names,
                epochs):
    """
    This function is used to train the model
    """

    # build the model
    model = cnn_model(class_names, img_height, img_width)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # visualize the results
    visualize_val_acc(epochs, history)
    visualize_val_loss(epochs, history)

    return model


def evaluate_model(model, test_ds):
    """
    This function is responsible to display the score
    of the model.

    :arg model: the model used to make the predictions:
    :arg test_ds: the dataset used to evaluate the model
    """
    score = model.evaluate(test_ds, verbose=2)
    print(
        "This CNN model has an accuracy of {:.2f}% on the test set."
        .format(100 * score[1])
    )
    input("Press Enter to continue...")
