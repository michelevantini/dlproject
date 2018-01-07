import tensorflow as tf
from scipy.misc import imread
import numpy as np
import os
from PIL import Image

classes = ["Benign", "InSitu", "Invasive", "Normal"]
dataset_path = os.path.join(os.path.pardir, 'dataset')
dataset = np.ndarray([400, 1536, 2048, 3])


for i, class_name in enumerate(classes):
    path = os.path.join(dataset_path, class_name)
    files = os.listdir(path)
    for j, img_name in enumerate(files):
        im = Image.open(path + "/" + img_name)
        imarray = np.array(im)
        dataset[i*100 + j] = imarray






