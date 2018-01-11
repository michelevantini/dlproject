"""Script to create a normalised dataset"""

import numpy as np
import os
from scipy import misc

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, os.path.pardir, "dataset")
DATA_SUB_DIRS = ["benign", "insitu", "invasive", "normal"]
assert os.path.exists(DATA_DIR)
NORM_DATA_DIR = os.path.join(CURRENT_DIR, os.path.pardir, "norm_dataset")
if not os.path.exists(NORM_DATA_DIR):
    os.makedirs(NORM_DATA_DIR)


def load_and_normalise(image_path):
    print(image_path)
    image_matrix = misc.imread(image_path)
    image_name = os.path.basename(image_path)
    print(image_matrix)
    print(type(image_matrix))
    print(image_name)


def batch_normalise():
    for subdir in DATA_SUB_DIRS:
        current_subdir = os.path.join(DATA_DIR, subdir)
        for i, file_name in enumerate(os.listdir(current_subdir)):
            if file_name.endswith(".jpeg"):
                print(i, ": ", end="")
                load_and_normalise(os.path.join(current_subdir, file_name))
                break
        break


if __name__ == "__main__":
    batch_normalise()




