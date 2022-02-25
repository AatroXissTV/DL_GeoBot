# tensorflow_check.py
# created 25/02/2022 at 11:17 by Antoine 'AatroXiss' BEAUDESSON
# last modified 25/02/2022 at 11:17 by Antoine 'AatroXiss' BEAUDESSON

""" tensorflow_check.py:
        Todo:
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
import tensorflow as tf

# local application imports

# other imports & constants


def check_tensorflow_version():
    """
    This function is responsible to check the version of tensorflow.
    """
    print("Tensorflow version: {}".format(tf.__version__))


def check_tensorflow_gpu():
    """
    This function is responsible to check the name of the GPU used.
    """
    print("GPU used: {}".format(tf.test.gpu_device_name()))
