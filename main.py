# main.py
# created 03/02/2022 at 12:02 by Antoine 'AatroXiss' BEAUDESSON
# last modified 11/02/2022 at 17:01 by Antoine 'AatroXiss' BEAUDESSON

""" main.py:
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

# third party imports
from sklearn.model_selection import train_test_split

# local application imports
from modules.datasets import (
    loading_dataset,
    construct_dataset,
    image_data_gen
)
from modules.model import (
    cnn_model,
    evaluate_model_val_acc,
    evaluate_model_val_loss,
)

# other imports

# constants
PATH_FR = 'dataset/train/France'
PATH_US = 'dataset/train/United States'
PATH_ALL = 'dataset/train/all'

EPOCHS = 10


def main():
    filenames_fr = loading_dataset(PATH_FR)
    filenames_us = loading_dataset(PATH_US)
    df = construct_dataset(filenames_fr, filenames_us)

    # check_dataset(df)
    train, val = train_test_split(df, test_size=0.5,
                                  random_state=17, stratify=df['label'])

    # image augmentation
    train_datagen = image_data_gen(train, PATH_ALL)
    val_datagen = image_data_gen(val, PATH_ALL)

    model = cnn_model()
    history = model.fit(train_datagen,
                        validation_data=val_datagen,
                        epochs=EPOCHS,
                        batch_size=32,
                        steps_per_epoch=len(train_datagen),
                        validation_steps=len(val_datagen))
    evaluate_model_val_loss(history, EPOCHS)
    evaluate_model_val_acc(history, EPOCHS)


main()
