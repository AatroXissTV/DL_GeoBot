# visualize_results.py
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
import matplotlib.pyplot as plt

# local application imports

# other imports

# constants


def visualize_val_acc(epochs, history):
    """
    """

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_range = range(epochs)
    print(epochs_range)

    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.show()
    input("Press Enter to continue...")


def visualize_val_loss(epochs, history):
    """
    """

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    input("Press Enter to continue...")
