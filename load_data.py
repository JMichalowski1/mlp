from __future__ import unicode_literals
import numpy as np
import os
from PIL import Image

DIRECTORY = '/Users/Jakub/Desktop/sieci_neuronowe_1/neural_network_2/data_set/'
IMAGE_ROW_SHAPE = (70,)


def get_label_from_filename(filename):
    return int(filename.split('_')[0])


def parse_images(path):
    return[(np.asarray(Image.open(path + filename).convert('1')).astype(int).reshape(IMAGE_ROW_SHAPE),
            get_label_from_filename(filename)) for filename in os.listdir(path)]


def list_to_arrays(data):
    images = np.empty((len(data), IMAGE_ROW_SHAPE[0]))
    labels = np.empty((len(data)))
    for i in range(len(data)):
        images[i, :] = data[i][0]
        labels[i] = data[i][1]
    return images, convert_labels_into_vectors(labels, labels.shape[0])


def create_dataset():
    images = parse_images(DIRECTORY)
    return list_to_arrays(images)


def convert_labels_into_vectors(labels, number_of_samples):
    label_output_matrix = np.zeros((number_of_samples, 10), dtype=float)
    sample_number = 0
    for label in labels:
        label_output_matrix[sample_number][round(label)] = 1.0
        sample_number += 1
    return label_output_matrix






