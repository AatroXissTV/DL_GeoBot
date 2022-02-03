# main.py
# created 03/02/2022 at 12:02 by Antoine 'AatroXiss' BEAUDESSON
# last modified 03/02/2022 at 12:02 by Antoine 'AatroXiss' BEAUDESSON

""" main.py:
    - *
"""

__author__ = "Antoine 'AatroXiss' BEAUDESSON"
__copyright__ = "Copyright 2021, Antoine 'AatroXiss' BEAUDESSON"
__credits__ = ["Antoine 'AatroXiss' BEAUDESSON"]
__license__ = ""
__version__ = "0.0.2"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports
import os

# third party imports
import matplotlib.pyplot as plt
import cv2
import numpy as np
import seaborn as sns

# local application imports

# other imports

# constants

labels = [
    "France",
    "United States",
]
img_size = 224


def get_data(dataset_dir):
    data = []
    for label in labels:
        path = os.path.join(dataset_dir, label)
        class_num = labels.index(label)
        for img_path in os.listdir(path):
            try:
                img_arr = cv2.imread(
                    os.path.join(path, img_path))[..., ::-1]  # BGR to RGB
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(f"Error while loading {img_path}: {e}")
    return np.array(data)


train = get_data("dataset/train")

train_list = []
for i in train:
    if(i[1] == 0):
        train_list.append("France")
    else:
        train_list.append("USA")
sns.set_style("darkgrid")
sns.countplot(train_list)
plt.show()
