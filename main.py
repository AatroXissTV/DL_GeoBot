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
__version__ = "0.0.15"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports

# third party imports
from keras import layers
import numpy as np

# local application imports
from modules.dataset_management import (
    explore_data,
    load_data,
)
from modules.model import (
    cnn_model,
    visualize_val_acc,
    visualize_val_loss
)

# other imports

# constants
PATH_TRAIN_DATASET = 'dataset/train/'

BATCH_SIZE = 32
IMG_HEIGHT = 331
IMG_WIDTH = 768
EPOCHS = 15


def main():
    """
    """
    train_dir = explore_data(PATH_TRAIN_DATASET)

    train_ds = load_data(train_dir, 'training',
                         BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)
    test_ds = load_data(train_dir, 'validation',
                        BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)

    # get image_batch and label_batch
    for image_batch, label_batch in train_ds:
        print(image_batch.shape, label_batch.shape)
        break

    class_names = train_ds.class_names

    # visualize the data
    # visualize_data(train_ds, class_names)

    # Performance
    # AUTOTUNE = tf.data.AUTOTUNE
    # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # standardize the data
    normalization_layer = layers.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, label_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    print(np.min(first_image), np.max(first_image))

    # Build model
    model = cnn_model(class_names, IMG_HEIGHT, IMG_WIDTH)
    history = model.fit(train_ds,
                        validation_data=test_ds,
                        epochs=EPOCHS)

    # visualize the results
    visualize_val_acc(EPOCHS, history)
    visualize_val_loss(EPOCHS, history)
    input("Press Enter to continue...")


if __name__ == "__main__":
    main()
